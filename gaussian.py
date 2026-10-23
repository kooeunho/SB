import os
MY_GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(MY_GPU_ID)

import math
import gc
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import (
    set_seed, as_2d, one_hot, digitize_like_go, rmse, r2, mean_std,
    full_joint_w2_sq_from_samples, PositiveMLP, GORJointPaper,
    go_impute_YB_continuous_from_gamma, go_impute_ZA_continuous_from_gamma,
    MixedMethodsImputerFast,
)

SEED = 42
SIGMA_YZ = 0.9
BIN_CONFIG = (4, 3, 4)
SB_LAMBDA = 0.1484968262254465
C_TOTAL = 50
W2_N_SUB = 2000
SB_SWEEP_RANDOM_OFFSET = 60000
GO_LP_TIME_LIMIT = 300.0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
VERBOSE = False
A_TOTAL = 10000
B_TOTAL = 10000
T_TOTAL = 10000
N_TOTAL = A_TOTAL + B_TOTAL + T_TOTAL
DIM_X = DIM_Y = DIM_Z = 1
SIGMA_XY = SIGMA_XZ = 0.5
EPS = 1e-08
BRIDGE_EPOCHS = 4000
BATCH_SIZE = 256
MC_SAMPLES = 256
LR_A = 0.0005
LR_B = 0.0005
SCALE_REG = 0.001
wA = 2.0
wB = 1.0
GRAD_CLIP_NORM = 5.0
IMPUTE_K = 1024
IMPUTE_BATCH = 2000
DENSITY_EPOCHS = 1000
DENSITY_BATCH = 256
DENSITY_LR = 0.001
DENSITY_PATIENCE = 50
DOWN_EPOCHS = 500
DOWN_LR = 0.001
DOWN_BATCH = 256
DOWN_PATIENCE = 30
GO_K_GRAPH = 5
GO_SOLVER = 'HIGHS'
GO_SOLVER_VERBOSE = False


class DownstreamMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


class CondGaussianMLP(nn.Module):

    def __init__(self, x_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        out = self.net(x)
        mu = out[:, [0]]
        log_var = out[:, [1]].clamp(-8.0, 5.0)
        return (mu, log_var)


def gaussian_nll(w, mu, log_var):
    return 0.5 * (math.log(2.0 * math.pi) + log_var + (w - mu).pow(2) / torch.exp(log_var))


def gaussian_logpdf(w, mu, log_var):
    return -gaussian_nll(w, mu, log_var)


def train_cond_density_mlp(X_np, W_np, seed, epochs=DENSITY_EPOCHS, batch_size=DENSITY_BATCH, lr=DENSITY_LR, patience=DENSITY_PATIENCE):
    set_seed(seed)
    X_np = np.asarray(X_np, dtype=np.float32)
    W_np = as_2d(W_np).astype(np.float32)
    X_tr, X_val, W_tr, W_val = train_test_split(X_np, W_np, test_size=0.2, random_state=seed, shuffle=True)
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    W_tr_t = torch.from_numpy(W_tr).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    W_val_t = torch.from_numpy(W_val).float().to(DEVICE)
    ds = torch.utils.data.TensorDataset(X_tr_t, W_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = CondGaussianMLP(x_dim=DIM_X, hidden=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float('inf')
    no_imp = 0
    for _ in range(epochs):
        model.train()
        for xb, wb in loader:
            opt.zero_grad(set_to_none=True)
            mu, log_var = model(xb)
            loss = gaussian_nll(wb, mu, log_var).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_val, log_var_val = model(X_val_t)
            val_loss = gaussian_nll(W_val_t, mu_val, log_var_val).mean().item()
        if val_loss < best_val - 1e-06:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_log_r_ratio(x, y_std, z_std, params):
    y_density = params['y_density']
    z_density = params['z_density']
    with torch.no_grad():
        mu_y, log_var_y = y_density(x)
        mu_z, log_var_z = z_density(x)
        logp_y = gaussian_logpdf(y_std, mu_y, log_var_y)
        logp_z = gaussian_logpdf(z_std, mu_z, log_var_z)
    return (logp_y - logp_z).squeeze(-1)


def make_gaussian_alignment_cost_fn_std(y_scale, z_scale):
    cy = float(SIGMA_XY) * float(y_scale)
    cz = float(SIGMA_XZ) * float(z_scale)

    def cost_fn(y_std, z_std):
        return ((cy * y_std - cz * z_std) ** 2).squeeze(-1)
    return cost_fn


def train_downstream_model(X_train, Y_train, input_dim, seed):
    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = as_2d(Y_train).astype(np.float32)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed, shuffle=True)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_tr_t = torch.from_numpy(X_tr_s).float().to(DEVICE)
    Y_tr_t = torch.from_numpy(Y_tr).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val_s).float().to(DEVICE)
    Y_val_t = torch.from_numpy(Y_val).float().to(DEVICE)
    model = DownstreamMLP(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=DOWN_LR)
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(X_tr_t, Y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=DOWN_BATCH, shuffle=True)
    best_state = None
    best_val = float('inf')
    no_imp = 0
    for _ in range(DOWN_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val_t), Y_val_t).item()
        if val_loss < best_val - 1e-08:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= DOWN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return (model, scaler)


@torch.no_grad()
def predict_downstream_model(model, scaler, X_np):
    X_np = np.asarray(X_np, dtype=np.float32)
    X_s = scaler.transform(X_np).astype(np.float32)
    X_t = torch.from_numpy(X_s).float().to(DEVICE)
    model.eval()
    pred = model(X_t).cpu().numpy().astype(np.float32)
    return pred


def eval_downstream_both_and_brmse(X_train, Y_train, X_strong, Y_strong, X_weak, Y_weak, X_B_eval, YB_true, input_dim, seed):
    model, scaler = train_downstream_model(X_train=X_train, Y_train=Y_train, input_dim=input_dim, seed=seed)
    pred_strong = predict_downstream_model(model, scaler, X_strong)
    pred_weak = predict_downstream_model(model, scaler, X_weak)
    pred_B = predict_downstream_model(model, scaler, X_B_eval)
    out = {'strong_r2': r2(Y_strong, pred_strong), 'weak_r2': r2(Y_weak, pred_weak), 'rmse_B': rmse(YB_true, pred_B)}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def train_sb(lambda_val, data_gpu, density_models, cost_fn):
    XA_dev = data_gpu['XA']
    YA_dev = data_gpu['YA_std']
    XB_dev = data_gpu['XB']
    ZB_dev = data_gpu['ZB_std']
    a_net = PositiveMLP(DIM_X + DIM_Y).to(DEVICE)
    b_net = PositiveMLP(DIM_X + DIM_Z).to(DEVICE)
    opt_a = torch.optim.Adam(a_net.parameters(), lr=LR_A)
    opt_b = torch.optim.Adam(b_net.parameters(), lr=LR_B)
    params = {'lambda': float(lambda_val), 'y_density': density_models['y_density'], 'z_density': density_models['z_density']}
    lam_f = float(lambda_val)
    N_A = XA_dev.size(0)
    N_B = XB_dev.size(0)
    for ep in range(BRIDGE_EPOCHS):
        idx_a = torch.randint(0, N_A, (BATCH_SIZE,), device=DEVICE)
        idx_b = torch.randint(0, N_B, (BATCH_SIZE,), device=DEVICE)
        x_a = XA_dev[idx_a]
        y_a = YA_dev[idx_a]
        x_b = XB_dev[idx_b]
        z_b = ZB_dev[idx_b]
        a_val = a_net(torch.cat([x_a, y_a], dim=-1)).squeeze(-1)
        mc_idx = torch.randint(0, N_B, (BATCH_SIZE, MC_SAMPLES), device=DEVICE)
        z_mc = ZB_dev[mc_idx.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Z)
        x_rep = x_a.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_X)
        y_rep = y_a.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_Y)
        z_rep = z_mc.reshape(-1, DIM_Z)
        with torch.no_grad():
            b_inner = b_net(torch.cat([x_rep, z_rep], dim=-1)).squeeze(-1)
            c1 = cost_fn(y_rep, z_rep)
            logw1 = torch.log(b_inner + EPS) - c1 / lam_f
            lme1 = torch.logsumexp(logw1.view(BATCH_SIZE, MC_SAMPLES), dim=1) - math.log(MC_SAMPLES)
        lossA = torch.mean((a_val * torch.exp(lme1) - 1.0) ** 2)
        if SCALE_REG > 0:
            lossA = lossA + SCALE_REG * (a_val.mean() - 1.0).pow(2)
        opt_a.zero_grad(set_to_none=True)
        (wA * lossA).backward()
        torch.nn.utils.clip_grad_norm_(a_net.parameters(), GRAD_CLIP_NORM)
        opt_a.step()
        b_val = b_net(torch.cat([x_b, z_b], dim=-1)).squeeze(-1)
        mc_idx_y = torch.randint(0, N_A, (BATCH_SIZE, MC_SAMPLES), device=DEVICE)
        y_mc = YA_dev[mc_idx_y.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Y)
        x2 = x_b.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_X)
        z2 = z_b.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_Z)
        y2 = y_mc.reshape(-1, DIM_Y)
        with torch.no_grad():
            a_inner = a_net(torch.cat([x2, y2], dim=-1)).squeeze(-1)
            log_r = get_log_r_ratio(x2, y2, z2, params)
            c2 = cost_fn(y2, z2)
            logw2 = torch.log(a_inner + EPS) + log_r - c2 / lam_f
            lme2 = torch.logsumexp(logw2.view(BATCH_SIZE, MC_SAMPLES), dim=1) - math.log(MC_SAMPLES)
        lossB = torch.mean((b_val * torch.exp(lme2) - 1.0) ** 2)
        if SCALE_REG > 0:
            lossB = lossB + SCALE_REG * (b_val.mean() - 1.0).pow(2)
        opt_b.zero_grad(set_to_none=True)
        (wB * lossB).backward()
        torch.nn.utils.clip_grad_norm_(b_net.parameters(), GRAD_CLIP_NORM)
        opt_b.step()
        if VERBOSE and (ep + 1) % 1000 == 0:
            print(f'      [SB] epoch {ep + 1}/{BRIDGE_EPOCHS} lossA={lossA.item():.3e}, lossB={lossB.item():.3e}')
    return (a_net, b_net)


@torch.no_grad()
def sb_precompute_A_candidates_weights(b_net, data_gpu, cost_fn, lambda_val, scaler_Z):
    b_net.eval()
    X = data_gpu['XA']
    Y = data_gpu['YA_std']
    Z_pool = data_gpu['ZB_std']
    N = X.size(0)
    M = Z_pool.size(0)
    K = min(IMPUTE_K, M)
    lam_f = float(lambda_val)
    z_cand = np.empty((N, K), dtype=np.float32)
    weights = np.empty((N, K), dtype=np.float32)
    z_hat_raw = np.empty((N, 1), dtype=np.float32)
    for i0 in range(0, N, IMPUTE_BATCH):
        x = X[i0:i0 + IMPUTE_BATCH]
        y = Y[i0:i0 + IMPUTE_BATCH]
        bs = x.size(0)
        idx = torch.randint(0, M, (bs, K), device=DEVICE)
        zK = Z_pool[idx]
        x_r = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_X)
        y_r = y.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_Y)
        z_r = zK.reshape(-1, DIM_Z)
        b_val = b_net(torch.cat([x_r, z_r], dim=-1)).squeeze(-1)
        c_val = cost_fn(y_r, z_r)
        logw = (torch.log(b_val + EPS) - c_val / lam_f).view(bs, K)
        w = torch.softmax(logw, dim=1)
        z_cand[i0:i0 + bs, :] = zK.squeeze(-1).cpu().numpy().astype(np.float32)
        weights[i0:i0 + bs, :] = w.cpu().numpy().astype(np.float32)
        sel = torch.multinomial(w, 1, replacement=True)
        z_sel_std = zK.squeeze(-1).gather(1, sel).view(bs, 1)
        z_sel_raw = scaler_Z.inverse_transform(z_sel_std.cpu().numpy().astype(np.float32)).astype(np.float32)
        z_hat_raw[i0:i0 + bs, :] = z_sel_raw
    return (z_cand, weights, z_hat_raw)


def sb_pbin_from_precomputed(z_candidates_std, weights, edges_z_std):
    z_candidates_std = np.asarray(z_candidates_std, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    N, K = z_candidates_std.shape
    nZ = len(edges_z_std) - 1
    boundaries = torch.tensor(edges_z_std[1:-1], dtype=torch.float32)
    p_out = np.zeros((N, nZ), dtype=np.float32)
    for i0 in range(0, N, 2000):
        i1 = min(i0 + 2000, N)
        z = torch.from_numpy(z_candidates_std[i0:i1, :])
        w = torch.from_numpy(weights[i0:i1, :])
        bs = z.size(0)
        bin_idx = torch.bucketize(z, boundaries, right=False)
        p = torch.zeros((bs, nZ), dtype=torch.float32)
        p.scatter_add_(1, bin_idx.long(), w)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p_out[i0:i1, :] = p.numpy()
    z_map = np.argmax(p_out, axis=1).astype(int)
    return z_map


@torch.no_grad()
def sb_impute_YB(a_net, data_gpu, density_models, cost_fn, lambda_val, scaler_Y):
    a_net.eval()
    X = data_gpu['XB']
    Z = data_gpu['ZB_std']
    Y_pool = data_gpu['YA_std']
    N = X.size(0)
    M = Y_pool.size(0)
    K = min(IMPUTE_K, M)
    lam_f = float(lambda_val)
    params = {'y_density': density_models['y_density'], 'z_density': density_models['z_density']}
    y_hat = np.empty((N, 1), dtype=np.float32)
    for i0 in range(0, N, IMPUTE_BATCH):
        x = X[i0:i0 + IMPUTE_BATCH]
        z = Z[i0:i0 + IMPUTE_BATCH]
        bs = x.size(0)
        idx = torch.randint(0, M, (bs, K), device=DEVICE)
        yK = Y_pool[idx]
        x_r = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_X)
        z_r = z.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_Z)
        y_r = yK.reshape(-1, DIM_Y)
        a_val = a_net(torch.cat([x_r, y_r], dim=-1)).squeeze(-1)
        c_val = cost_fn(y_r, z_r)
        log_r = get_log_r_ratio(x_r, y_r, z_r, params)
        logw = (torch.log(a_val + EPS) + log_r - c_val / lam_f).view(bs, K)
        w = torch.softmax(logw, dim=1)
        sel = torch.multinomial(w, 1, replacement=True)
        y_sel_std = yK.squeeze(-1).gather(1, sel).view(bs, 1)
        y_sel_raw = scaler_Y.inverse_transform(y_sel_std.cpu().numpy().astype(np.float32)).astype(np.float32)
        y_hat[i0:i0 + bs, :] = y_sel_raw
    return y_hat


def make_gaussian_data(seed, sigma_yz):
    set_seed(seed)
    mean = [0.0, 0.0, 0.0]
    cov = [[1.0, SIGMA_XY, SIGMA_XZ], [SIGMA_XY, 1.0, float(sigma_yz)], [SIGMA_XZ, float(sigma_yz), 1.0]]
    data = np.random.multivariate_normal(mean, cov, size=N_TOTAL).astype(np.float32)
    X = data[:, [0]]
    Y = data[:, [1]]
    Z = data[:, [2]]
    XA = X[:A_TOTAL]
    YA = Y[:A_TOTAL]
    ZA = Z[:A_TOTAL]
    b0 = A_TOTAL
    b1 = A_TOTAL + B_TOTAL
    XB = X[b0:b1]
    YB_true = Y[b0:b1]
    ZB = Z[b0:b1]
    XT = X[b1:]
    YT = Y[b1:]
    ZT = Z[b1:]
    return (XA, YA, ZA, XB, YB_true, ZB, XT, YT, ZT)


def _w2_joint(D_hat, D_true, seed):
    return full_joint_w2_sq_from_samples(D_hat, D_true, n_sub=W2_N_SUB, seed=seed)


def generate_weak_test_C(sigma_yz):
    mean = [0.0, 0.0, 0.0]
    cov = [[1.0, SIGMA_XY, SIGMA_XZ], [SIGMA_XY, 1.0, float(sigma_yz)], [SIGMA_XZ, float(sigma_yz), 1.0]]
    C = np.random.multivariate_normal(mean, cov, size=C_TOTAL).astype(np.float32)
    return (C[:, [0]], C[:, [1]], C[:, [2]])


def run_experiment():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 0 is not detected. Check CUDA_VISIBLE_DEVICES=0 and the kernel status.')
    start = time.time()
    seed = SEED
    sigma_yz = SIGMA_YZ
    lam = SB_LAMBDA
    XA, YA, ZA, XB, YB_true, ZB, XT, YT, ZT = make_gaussian_data(seed=seed, sigma_yz=sigma_yz)
    XC, YC, ZC = generate_weak_test_C(sigma_yz)
    results = []
    scaler_Y = StandardScaler()
    scaler_Z = StandardScaler()
    YA_std = scaler_Y.fit_transform(YA).astype(np.float32)
    ZB_std = scaler_Z.fit_transform(ZB).astype(np.float32)
    y_scale = float(scaler_Y.scale_[0])
    z_scale = float(scaler_Z.scale_[0])
    data_gpu = {'XA': torch.from_numpy(XA).float().to(DEVICE), 'YA_std': torch.from_numpy(YA_std).float().to(DEVICE), 'XB': torch.from_numpy(XB).float().to(DEVICE), 'ZB_std': torch.from_numpy(ZB_std).float().to(DEVICE)}
    y_density = train_cond_density_mlp(X_np=XA, W_np=YA_std, seed=seed + 7000)
    z_density = train_cond_density_mlp(X_np=XB, W_np=ZB_std, seed=seed + 8000)
    density_models = {'y_density': y_density, 'z_density': z_density}
    cost_fn = make_gaussian_alignment_cost_fn_std(y_scale, z_scale)
    D_true = np.hstack([as_2d(XT), as_2d(YT), as_2d(ZT)]).astype(np.float64)
    naive_metrics = eval_downstream_both_and_brmse(X_train=XA, Y_train=YA, X_strong=XT, Y_strong=YT, X_weak=XC, Y_weak=YC, X_B_eval=XB, YB_true=YB_true, input_dim=DIM_X, seed=seed + 1000)
    results.append(('NAIVE', naive_metrics['strong_r2'], naive_metrics['weak_r2'], naive_metrics['rmse_B'], np.nan))
    mm = MixedMethodsImputerFast(random_state=seed)
    mm.fit(XA, YA_std, XB, ZB_std)
    mm_outputs = {}
    for k in (1, 2, 3, 4, 5):
        if k == 1:
            ZA_hat_std = mm.impute_mm1(XA, XB, ZB_std)
            YB_hat_std = mm.impute_y_mm1(XA, YA_std, XB)
        elif k == 2:
            ZA_hat_std = mm.impute_mm2(XA, ZB_std)
            YB_hat_std = mm.impute_y_mm2(YA_std, XB)
        elif k == 3:
            ZA_hat_std = mm.impute_mm3(XA, ZB_std)
            YB_hat_std = mm.impute_y_mm3(YA_std, XB)
        elif k == 4:
            ZA_hat_std = mm.impute_mm4(XA, ZB_std)
            YB_hat_std = mm.impute_y_mm4(YA_std, XB)
        else:
            ZA_hat_std = mm.impute_mm5(XA, YA_std, XB, ZB_std)
            YB_hat_std = mm.impute_y_mm5(XA, YA_std, XB, ZB_std)
        mm_outputs[k] = {'ZA_hat_raw': scaler_Z.inverse_transform(as_2d(ZA_hat_std)).astype(np.float32), 'YB_hat_raw': scaler_Y.inverse_transform(as_2d(YB_hat_std)).astype(np.float32)}
    n_bins_x, n_bins_y, n_bins_z = BIN_CONFIG
    go = GORJointPaper(n_bins_x=n_bins_x, n_bins_y=n_bins_y, n_bins_z=n_bins_z, standardize_x=True, k_graph=GO_K_GRAPH, solver=GO_SOLVER, solver_verbose=GO_SOLVER_VERBOSE, random_state=seed, cost_sample_cap=512, lp_time_limit=GO_LP_TIME_LIMIT)
    go.fit(XA, YA, XB, ZB)
    edges_z = go.get_binning()['Z']
    nZ_eff = len(edges_z) - 1
    ZA_bin_true = digitize_like_go(ZA, edges_z)
    ZB_bin_true = digitize_like_go(ZB, edges_z)
    ZT_bin_true = digitize_like_go(ZT, edges_z)
    ZC_bin_true = digitize_like_go(ZC, edges_z)
    ZB_oh = one_hot(ZB_bin_true, nZ_eff)
    T_feat = np.hstack([XT, one_hot(ZT_bin_true, nZ_eff)]).astype(np.float32)
    C_feat = np.hstack([XC, one_hot(ZC_bin_true, nZ_eff)]).astype(np.float32)
    B_feat_true = np.hstack([XB, ZB_oh]).astype(np.float32)
    X_orac_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_true, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
    Y_orac_train = np.vstack([YA, YB_true]).astype(np.float32)
    orac_metrics = eval_downstream_both_and_brmse(X_train=X_orac_train, Y_train=Y_orac_train, X_strong=T_feat, Y_strong=YT, X_weak=C_feat, Y_weak=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=1 + nZ_eff, seed=seed + 2000 + nZ_eff)
    results.append(('ORACLE', orac_metrics['strong_r2'], orac_metrics['weak_r2'], orac_metrics['rmse_B'], np.nan))
    pz_go = go.predict_proba(XA, YA)
    ZA_bin_go = np.argmax(pz_go, axis=1).astype(int)
    YB_hat_go = go_impute_YB_continuous_from_gamma(go, XB, ZB, YA)
    X_go_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_go, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
    Y_go_train = np.vstack([YA, YB_hat_go]).astype(np.float32)
    go_metrics = eval_downstream_both_and_brmse(X_train=X_go_train, Y_train=Y_go_train, X_strong=T_feat, Y_strong=YT, X_weak=C_feat, Y_weak=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=1 + nZ_eff, seed=seed + 3000 + nZ_eff)
    ZA_hat_go_cont = go_impute_ZA_continuous_from_gamma(go, XA, YA, ZB)
    D_go = np.vstack([np.hstack([as_2d(XA), as_2d(YA), as_2d(ZA_hat_go_cont)]), np.hstack([as_2d(XB), as_2d(YB_hat_go), as_2d(ZB)])]).astype(np.float64)
    results.append(('GO', go_metrics['strong_r2'], go_metrics['weak_r2'], rmse(YB_true, YB_hat_go), _w2_joint(D_go, D_true, seed)))
    for k in (1, 2, 3, 4, 5):
        ZA_hat_raw = mm_outputs[k]['ZA_hat_raw']
        YB_hat_raw = mm_outputs[k]['YB_hat_raw']
        ZA_bin_mm = digitize_like_go(ZA_hat_raw, edges_z)
        X_mm_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_mm, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
        Y_mm_train = np.vstack([YA, YB_hat_raw]).astype(np.float32)
        mm_metrics = eval_downstream_both_and_brmse(X_train=X_mm_train, Y_train=Y_mm_train, X_strong=T_feat, Y_strong=YT, X_weak=C_feat, Y_weak=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=1 + nZ_eff, seed=seed + 4000 + 10 * k + nZ_eff)
        D_mm = np.vstack([np.hstack([as_2d(XA), as_2d(YA), as_2d(ZA_hat_raw)]), np.hstack([as_2d(XB), as_2d(YB_hat_raw), as_2d(ZB)])]).astype(np.float64)
        results.append((f'MM{k}', mm_metrics['strong_r2'], mm_metrics['weak_r2'], rmse(YB_true, YB_hat_raw), _w2_joint(D_mm, D_true, seed)))
    edges_z_std = (edges_z - float(scaler_Z.mean_[0])) / (float(scaler_Z.scale_[0]) + 1e-12)
    set_seed(seed + SB_SWEEP_RANDOM_OFFSET)
    a_net, b_net = train_sb(lambda_val=lam, data_gpu=data_gpu, density_models=density_models, cost_fn=cost_fn)
    z_cand_A_std, w_A, ZA_hat_sb_raw = sb_precompute_A_candidates_weights(b_net=b_net, data_gpu=data_gpu, cost_fn=cost_fn, lambda_val=lam, scaler_Z=scaler_Z)
    YB_hat_sb = sb_impute_YB(a_net=a_net, data_gpu=data_gpu, density_models=density_models, cost_fn=cost_fn, lambda_val=lam, scaler_Y=scaler_Y)
    ZA_bin_sb = sb_pbin_from_precomputed(z_candidates_std=z_cand_A_std, weights=w_A, edges_z_std=edges_z_std)
    X_sb_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_sb, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
    Y_sb_train = np.vstack([YA, YB_hat_sb]).astype(np.float32)
    sb_metrics = eval_downstream_both_and_brmse(X_train=X_sb_train, Y_train=Y_sb_train, X_strong=T_feat, Y_strong=YT, X_weak=C_feat, Y_weak=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=1 + nZ_eff, seed=seed + 5000 + 100 + nZ_eff)
    D_sb = np.vstack([np.hstack([as_2d(XA), as_2d(YA), as_2d(ZA_hat_sb_raw)]), np.hstack([as_2d(XB), as_2d(YB_hat_sb), as_2d(ZB)])]).astype(np.float64)
    results.append(('SB', sb_metrics['strong_r2'], sb_metrics['weak_r2'], rmse(YB_true, YB_hat_sb), _w2_joint(D_sb, D_true, seed)))
    width = 96
    print('\n' + '=' * width)
    print(f'Gaussian experiment | GPU={MY_GPU_ID} | seed={seed} | sigma_yz={sigma_yz}')
    print(f'BIN_CONFIG={BIN_CONFIG} | SB lambda={lam:.4f} | A={len(YA)} | B={len(ZB)}')
    print(f'Strong R2: T={len(YT)} | Weak R2: C={len(YC)} (same trained models)')
    print(f'RMSE_B: B={len(YB_true)} | W2^2 truth reference: T={len(YT)}')
    print('-' * width)
    print(f"{'Method':<11} {'Strong R2':>16} {'Weak R2':>16} {'RMSE_B':>15} {'W2^2 (XYZ)':>18}")
    print('-' * width)
    for method, strong_r2, weak_r2, rmse_b, w2 in results:
        w2_str = '-' if np.isnan(w2) else f'{w2:.6f}'
        print(f'{method:<11} {strong_r2:>16.6f} {weak_r2:>16.6f} {rmse_b:>15.6f} {w2_str:>18}')
    print('-' * width)
    print(f'Total elapsed: {time.time() - start:.1f} seconds')
    print('=' * width)
    del a_net, b_net, go, y_density, z_density
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    results = run_experiment()
