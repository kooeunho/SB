import os
MY_GPU_ID = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(MY_GPU_ID)

import math
import gc
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import set_seed, safe_auc
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
SB_LAMBDA = 0.453487850812858
C_TOTAL = 50
SB_SWEEP_RANDOM_OFFSET = 60000
PATH_X = 'data/X_embeddings_8d.npy'
PATH_Y = 'data/Y_data.npy'
PATH_Z = 'data/Z_data.npy'
MAX_SAMPLES = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIM_X = 8
DIM_Y = 1
DIM_Z = 5
Z_STATES = 1 << DIM_Z
BRIDGE_EPOCHS = 6000
BATCH_SIZE = 512
LR_A = 0.0005
LR_B = 0.0005
WEIGHT_DECAY = 1e-05
SCALE_REG = 0.001
wA = 2.0
wB = 1.0
CLIP_NORM = 5.0
DOWN_EPOCHS = 300
DOWN_LR = 0.001
DOWN_BATCH = 256
DOWN_PATIENCE = 25
COST_BETA = 1.0
COST_SHRINK_ALPHA = 50.0
EPS = 1e-08
EPS_LOG = 1e-06


def all_binary_states(d: int):
    states = np.zeros((1 << d, d), dtype=np.float32)
    for i in range(1 << d):
        for j in range(d):
            states[i, j] = i >> j & 1
    return states


def z_to_codes(z_bin):
    z_bin = np.asarray(z_bin).astype(np.int64)
    powers = (1 << np.arange(DIM_Z)).astype(np.int64)
    return (z_bin * powers[None, :]).sum(axis=1).astype(np.int64)


class PositiveMLP(nn.Module):

    def __init__(self, input_dim, hidden=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):
        return self.net(x) + 1e-06


class RatioNetBinary(nn.Module):

    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PzCodeNet(nn.Module):

    def __init__(self, input_dim, num_codes=32, hidden=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_codes))

    def forward(self, x):
        return self.net(x)


class DownstreamMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data_npy(max_samples=None):
    X = np.load(PATH_X).astype(np.float32)
    Y = np.load(PATH_Y).astype(np.float32)
    Z = np.load(PATH_Z).astype(np.float32)
    if max_samples is not None and max_samples < X.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        Z = Z[idx]
    return (X, Y, Z)


def split_data_abct(X, Y, Z, seed, nC=C_TOTAL):
    set_seed(seed)
    n = X.shape[0]
    perm = np.random.permutation(n)
    X = X[perm]
    Y = Y[perm]
    Z = Z[perm]
    n5 = n // 5
    s1 = 2 * n5
    s3 = 4 * n5
    XA, YA, ZA = (X[:s1], Y[:s1], Z[:s1])
    XB, YB, ZB = (X[s1:s3], Y[s1:s3], Z[s1:s3])
    XT_full, YT_full, ZT_full = (X[s3:], Y[s3:], Z[s3:])
    if XT_full.shape[0] <= nC:
        raise ValueError(f'T_full has only {XT_full.shape[0]} samples, cannot reserve C={nC}.')
    y_t_bin = (YT_full.reshape(-1) > 0.5).astype(np.int64)
    if np.unique(y_t_bin).size < 2:
        raise ValueError('T_full contains only one Y class; C-AUC selection is impossible.')
    idx_local = np.arange(XT_full.shape[0])
    idx_C, idx_T = train_test_split(idx_local, train_size=nC, stratify=y_t_bin, random_state=seed + 2, shuffle=True)
    return {'XA': XA, 'YA': YA, 'ZA': ZA, 'XB': XB, 'YB': YB, 'ZB': ZB, 'XC': XT_full[idx_C], 'YC': YT_full[idx_C], 'ZC': ZT_full[idx_C], 'XT': XT_full[idx_T], 'YT': YT_full[idx_T], 'ZT': ZT_full[idx_T]}


def train_downstream(X_train, Y_train, input_dim, seed):
    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32).reshape(-1)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed, shuffle=True)
    Xtr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    Ytr_t = torch.from_numpy(Y_tr).float().to(DEVICE)
    Xv_t = torch.from_numpy(X_val).float().to(DEVICE)
    Yv_t = torch.from_numpy(Y_val).float().to(DEVICE)
    ds = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=DOWN_BATCH, shuffle=True, drop_last=False)
    model = DownstreamMLP(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=DOWN_LR)
    best_loss = float('inf')
    best_state = None
    patience = 0
    for _ in range(DOWN_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = F.binary_cross_entropy_with_logits(model(Xv_t), Yv_t).item()
        if v_loss < best_loss - 1e-06:
            best_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= DOWN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_score(model, X):
    X = np.asarray(X, dtype=np.float32)
    Xt = torch.from_numpy(X).float().to(DEVICE)
    model.eval()
    return model(Xt).detach().cpu().numpy().reshape(-1)


def fit_code_mean_cost_matrix(XA_std, YA_raw, XB_std, ZB_raw):
    YA_i = (YA_raw.reshape(-1) > 0.5).astype(np.int64)
    ZB_i = (ZB_raw > 0.5).astype(np.int64)
    z_codes = z_to_codes(ZB_i)
    X_all = np.concatenate([XA_std, XB_std], axis=0).astype(np.float64)
    mu_all = X_all.mean(axis=0)
    std_all = X_all.std(axis=0) + 1e-06
    mu_y = np.zeros((2, DIM_X), dtype=np.float64)
    for y in [0, 1]:
        m = XA_std[YA_i == y]
        mu_y[y] = mu_all if m.shape[0] == 0 else m.mean(axis=0)
    mu_z = np.zeros((Z_STATES, DIM_X), dtype=np.float64)
    counts = np.bincount(z_codes, minlength=Z_STATES).astype(np.float64)
    for code in range(Z_STATES):
        idx = z_codes == code
        n = counts[code]
        mu_hat = mu_all if n <= 0 else XB_std[idx].mean(axis=0)
        w = n / (n + COST_SHRINK_ALPHA)
        mu_z[code] = w * mu_hat + (1.0 - w) * mu_all
    cost = np.zeros((2, Z_STATES), dtype=np.float64)
    for y in [0, 1]:
        diff = (mu_y[y][None, :] - mu_z) / std_all[None, :]
        cost[y] = COST_BETA * np.sum(diff * diff, axis=1)
    cost_t = torch.tensor(cost.astype(np.float32), device=DEVICE)
    powers_t = torch.tensor((1 << np.arange(DIM_Z)).astype(np.int64), device=DEVICE)
    return {'cost_matrix': cost_t, 'powers': powers_t}


def make_cost_fn(stats, cost_pack):
    y_mean = stats['y_mean']
    y_scale = stats['y_scale']
    z_mean = stats['z_mean']
    z_scale = stats['z_scale']
    cost_matrix = cost_pack['cost_matrix']
    powers = cost_pack['powers']

    def cost_fn(y_std, z_std):
        y_raw = y_std.view(-1, 1) * y_scale + y_mean
        z_raw = z_std.view(-1, DIM_Z) * z_scale + z_mean
        y_bin = (y_raw > 0.5).long().view(-1)
        z_bin = (z_raw > 0.5).long()
        z_code = torch.sum(z_bin * powers.view(1, -1), dim=1).long()
        return cost_matrix[y_bin, z_code]
    return cost_fn


def train_ratio_models(XA_std, YA_raw, XB_std, ZB_raw, epochs=10):
    pyx = RatioNetBinary(DIM_X).to(DEVICE)
    pzx = PzCodeNet(DIM_X, num_codes=Z_STATES).to(DEVICE)
    opt_y = torch.optim.Adam(pyx.parameters(), lr=0.001)
    opt_z = torch.optim.Adam(pzx.parameters(), lr=0.001)
    XA_t = torch.from_numpy(XA_std).float().to(DEVICE)
    YA_t = torch.from_numpy((YA_raw.reshape(-1) > 0.5).astype(np.float32)).float().to(DEVICE)
    XB_t = torch.from_numpy(XB_std).float().to(DEVICE)
    Z_codes = z_to_codes((ZB_raw > 0.5).astype(np.int64))
    Z_t = torch.from_numpy(Z_codes).long().to(DEVICE)
    bs = 2048
    for _ in range(epochs):
        pyx.train()
        perm = torch.randperm(XA_t.size(0), device=DEVICE)
        for i in range(0, XA_t.size(0), bs):
            idx = perm[i:i + bs]
            opt_y.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(pyx(XA_t[idx]), YA_t[idx])
            loss.backward()
            opt_y.step()
        pzx.train()
        perm = torch.randperm(XB_t.size(0), device=DEVICE)
        for i in range(0, XB_t.size(0), bs):
            idx = perm[i:i + bs]
            opt_z.zero_grad(set_to_none=True)
            loss = F.cross_entropy(pzx(XB_t[idx]), Z_t[idx])
            loss.backward()
            opt_z.step()
    pyx.eval()
    pzx.eval()
    return (pyx, pzx)


@torch.no_grad()
def log_pz_given_x_code(pzx_model, x_t, z_code_t):
    logits = pzx_model(x_t)
    logp = F.log_softmax(logits, dim=1)
    idx = torch.arange(x_t.size(0), device=x_t.device)
    return logp[idx, z_code_t].clamp(min=-20.0, max=0.0)


@torch.no_grad()
def log_pz_all(pzx_model, x_t):
    logits = pzx_model(x_t)
    return F.log_softmax(logits, dim=1).clamp(min=-20.0, max=0.0)


def train_bridge(lam, train_gpu, cost_fn, pyx, pzx):
    a_net = PositiveMLP(DIM_X + DIM_Y).to(DEVICE)
    b_net = PositiveMLP(DIM_X + DIM_Z, dropout_rate=0.0).to(DEVICE)
    opt_a = torch.optim.Adam(a_net.parameters(), lr=LR_A, weight_decay=WEIGHT_DECAY)
    opt_b = torch.optim.Adam(b_net.parameters(), lr=LR_B, weight_decay=WEIGHT_DECAY)
    XA = train_gpu['XA']
    YA = train_gpu['YA']
    XB = train_gpu['XB']
    ZB = train_gpu['ZB']
    ZB_code = train_gpu['ZB_code']
    y_states = train_gpu['y_states']
    z_states = train_gpu['z_states']
    nA = XA.size(0)
    nB = XB.size(0)
    S = z_states.size(0)
    for ep in range(BRIDGE_EPOCHS):
        ia = torch.randint(0, nA, (BATCH_SIZE,), device=DEVICE)
        ib = torch.randint(0, nB, (BATCH_SIZE,), device=DEVICE)
        x_a = XA[ia]
        y_a = YA[ia]
        x_b = XB[ib]
        z_b = ZB[ib]
        z_code = ZB_code[ib]
        a_val = a_net(torch.cat([x_a, y_a], dim=-1)).squeeze(-1)
        x_rep = x_a.unsqueeze(1).expand(-1, S, -1).reshape(-1, DIM_X)
        z_rep = z_states.unsqueeze(0).expand(BATCH_SIZE, -1, -1).reshape(-1, DIM_Z)
        y_rep = y_a.unsqueeze(1).expand(-1, S, -1).reshape(-1, DIM_Y)
        with torch.no_grad():
            b_inner = b_net(torch.cat([x_rep, z_rep], dim=-1)).squeeze(-1)
        c1 = cost_fn(y_rep, z_rep)
        logpz_cond_A = log_pz_all(pzx, x_a)
        logw1 = (torch.log(b_inner + EPS) - c1 / lam).view(BATCH_SIZE, S) + logpz_cond_A
        lse1 = torch.logsumexp(logw1, dim=1)
        lossA = torch.mean((a_val * torch.exp(lse1) - 1.0) ** 2)
        lossA = lossA + SCALE_REG * (a_val.mean() - 1.0).pow(2)
        opt_a.zero_grad(set_to_none=True)
        (wA * lossA).backward()
        torch.nn.utils.clip_grad_norm_(a_net.parameters(), CLIP_NORM)
        opt_a.step()
        bq = b_net(torch.cat([x_b, z_b], dim=-1)).squeeze(-1)
        log_pz_obs = log_pz_given_x_code(pzx, x_b, z_code)
        b_val = bq * torch.exp(log_pz_obs)
        S_y = 2
        y2 = y_states.unsqueeze(0).expand(BATCH_SIZE, -1, -1).reshape(-1, DIM_Y)
        x2 = x_b.unsqueeze(1).expand(-1, S_y, -1).reshape(-1, DIM_X)
        z2 = z_b.unsqueeze(1).expand(-1, S_y, -1).reshape(-1, DIM_Z)
        with torch.no_grad():
            a_inner = a_net(torch.cat([x2, y2], dim=-1)).squeeze(-1).view(BATCH_SIZE, S_y)
            lpz_cond = log_pz_given_x_code(pzx, x_b, z_code).view(BATCH_SIZE, 1)
            py1 = torch.sigmoid(pyx(x_b)).clamp(EPS_LOG, 1.0 - EPS_LOG)
            lpy = torch.stack([torch.log(1.0 - py1), torch.log(py1)], dim=1)
        c2 = cost_fn(y2, z2).view(BATCH_SIZE, S_y)
        log_r = lpy - lpz_cond
        logw2 = torch.log(a_inner + EPS) + log_r - c2 / lam
        lse2 = torch.logsumexp(logw2, dim=1)
        lossB = torch.mean((b_val * torch.exp(lse2) - 1.0) ** 2)
        lossB = lossB + SCALE_REG * (bq.mean() - 1.0).pow(2)
        opt_b.zero_grad(set_to_none=True)
        (wB * lossB).backward()
        torch.nn.utils.clip_grad_norm_(b_net.parameters(), CLIP_NORM)
        opt_b.step()
    a_net.eval()
    b_net.eval()
    return (a_net, b_net)


@torch.no_grad()
def sb_impute_z(b_net, train_gpu, cost_fn, lam, pzx):
    XA = train_gpu['XA']
    YA = train_gpu['YA']
    z_states = train_gpu['z_states']
    z_raw_states = train_gpu['z_raw_states']
    nA = XA.size(0)
    S = z_states.size(0)
    out = np.zeros((nA, DIM_Z), dtype=np.float32)
    bs = 2048
    for i0 in range(0, nA, bs):
        i1 = min(i0 + bs, nA)
        x = XA[i0:i1]
        y = YA[i0:i1]
        curr = x.size(0)
        x_rep = x.unsqueeze(1).expand(-1, S, -1).reshape(-1, DIM_X)
        z_rep = z_states.unsqueeze(0).expand(curr, -1, -1).reshape(-1, DIM_Z)
        y_rep = y.unsqueeze(1).expand(-1, S, -1).reshape(-1, DIM_Y)
        b_val = b_net(torch.cat([x_rep, z_rep], dim=-1)).squeeze(-1).view(curr, S)
        c_val = cost_fn(y_rep, z_rep).view(curr, S)
        logpz_cond = log_pz_all(pzx, x)
        logw = torch.log(b_val + EPS) - c_val / lam + logpz_cond
        p = torch.softmax(logw, dim=1)
        z_mean = p @ z_raw_states
        out[i0:i1] = z_mean.cpu().numpy().astype(np.float32)
    return out


@torch.no_grad()
def sb_impute_y(a_net, pyx, train_gpu, cost_fn, lam):
    XB = train_gpu['XB']
    ZB = train_gpu['ZB']
    y_states = train_gpu['y_states']
    nB = XB.size(0)
    S_y = 2
    out = np.zeros((nB,), dtype=np.float32)
    bs = 4096
    for i0 in range(0, nB, bs):
        i1 = min(i0 + bs, nB)
        x = XB[i0:i1]
        z = ZB[i0:i1]
        curr = x.size(0)
        x_rep = x.unsqueeze(1).expand(-1, S_y, -1).reshape(-1, DIM_X)
        y_rep = y_states.unsqueeze(0).expand(curr, -1, -1).reshape(-1, DIM_Y)
        z_rep = z.unsqueeze(1).expand(-1, S_y, -1).reshape(-1, DIM_Z)
        a_val = a_net(torch.cat([x_rep, y_rep], dim=-1)).squeeze(-1).view(curr, S_y)
        c_val = cost_fn(y_rep, z_rep).view(curr, S_y)
        py1 = torch.sigmoid(pyx(x)).clamp(EPS_LOG, 1.0 - EPS_LOG)
        lpy = torch.stack([torch.log(1.0 - py1), torch.log(py1)], dim=1)
        logw = torch.log(a_val + EPS) + lpy - c_val / lam
        p = torch.softmax(logw, dim=1)
        out[i0:i1] = p[:, 1].cpu().numpy().astype(np.float32)
    return np.clip(out, 1e-06, 1.0 - 1e-06)


def prepare_seed_context_weak(seed):
    set_seed(seed)
    X_raw, Y_raw, Z_raw = load_data_npy(MAX_SAMPLES)
    sp = split_data_abct(X_raw, Y_raw, Z_raw, seed, nC=C_TOTAL)
    scX = StandardScaler()
    scX.fit(np.concatenate([sp['XA'], sp['XB']], axis=0))
    XA_std = scX.transform(sp['XA']).astype(np.float32)
    XB_std = scX.transform(sp['XB']).astype(np.float32)
    XC_std = scX.transform(sp['XC']).astype(np.float32)
    XT_std = scX.transform(sp['XT']).astype(np.float32)
    scY = StandardScaler().fit(sp['YA'])
    YA_std = scY.transform(sp['YA']).astype(np.float32)
    scZ = StandardScaler().fit(sp['ZB'])
    ZB_std = scZ.transform(sp['ZB']).astype(np.float32)
    stats = {'y_mean': torch.tensor(float(scY.mean_[0]), device=DEVICE), 'y_scale': torch.tensor(float(scY.scale_[0]), device=DEVICE), 'z_mean': torch.tensor(scZ.mean_.astype(np.float32), device=DEVICE), 'z_scale': torch.tensor(scZ.scale_.astype(np.float32), device=DEVICE)}
    cost_pack = fit_code_mean_cost_matrix(XA_std, sp['YA'], XB_std, sp['ZB'])
    cost_fn = make_cost_fn(stats, cost_pack)
    cost_matrix_np = cost_pack['cost_matrix'].detach().cpu().numpy().astype(np.float32)
    pyx, pzx = train_ratio_models(XA_std, sp['YA'], XB_std, sp['ZB'], epochs=10)
    y_states = torch.from_numpy(scY.transform(np.array([[0.0], [1.0]], dtype=np.float32)).astype(np.float32)).float().to(DEVICE)
    z_raw_states_np = all_binary_states(DIM_Z)
    z_states = torch.from_numpy(scZ.transform(z_raw_states_np).astype(np.float32)).float().to(DEVICE)
    ZB_bits = (sp['ZB'] > 0.5).astype(np.int64)
    ZB_codes = z_to_codes(ZB_bits)
    train_gpu = {'XA': torch.from_numpy(XA_std).float().to(DEVICE), 'YA': torch.from_numpy(YA_std).float().to(DEVICE), 'XB': torch.from_numpy(XB_std).float().to(DEVICE), 'ZB': torch.from_numpy(ZB_std).float().to(DEVICE), 'ZB_code': torch.from_numpy(ZB_codes).long().to(DEVICE), 'y_states': y_states, 'z_states': z_states, 'z_raw_states': torch.from_numpy(z_raw_states_np).float().to(DEVICE)}
    ZA_codes = z_to_codes((sp['ZA'] > 0.5).astype(np.int64))
    ZC_codes = z_to_codes((sp['ZC'] > 0.5).astype(np.int64))
    ZT_codes = z_to_codes((sp['ZT'] > 0.5).astype(np.int64))
    return {'seed': seed, 'XA_std': XA_std, 'XB_std': XB_std, 'XC_std': XC_std, 'XT_std': XT_std, 'YA': sp['YA'], 'YB': sp['YB'], 'YC': sp['YC'], 'YT': sp['YT'], 'ZA': sp['ZA'], 'ZB': sp['ZB'], 'ZC': sp['ZC'], 'ZT': sp['ZT'], 'ZA_codes': ZA_codes, 'ZB_codes': ZB_codes, 'ZC_codes': ZC_codes, 'ZT_codes': ZT_codes, 'train_gpu': train_gpu, 'cost_fn': cost_fn, 'cost_matrix_np': cost_matrix_np, 'pyx': pyx, 'pzx': pzx}


def make_completed_downstream_training(ctx, zA_feat_bits, p_y1_B):
    XA_feat = np.concatenate([ctx['XA_std'], zA_feat_bits], axis=1).astype(np.float32)
    XB_feat = np.concatenate([ctx['XB_std'], ctx['ZB']], axis=1).astype(np.float32)
    X_train = np.concatenate([XA_feat, XB_feat], axis=0).astype(np.float32)
    Y_train = np.concatenate([ctx['YA'].reshape(-1), p_y1_B.reshape(-1)], axis=0).astype(np.float32)
    return (X_train, Y_train)


def run_lambda_completion(ctx, lam):
    a_net, b_net = train_bridge(lam=lam, train_gpu=ctx['train_gpu'], cost_fn=ctx['cost_fn'], pyx=ctx['pyx'], pzx=ctx['pzx'])
    zA_feat = sb_impute_z(b_net=b_net, train_gpu=ctx['train_gpu'], cost_fn=ctx['cost_fn'], lam=lam, pzx=ctx['pzx'])
    p_y1_B = sb_impute_y(a_net=a_net, pyx=ctx['pyx'], train_gpu=ctx['train_gpu'], cost_fn=ctx['cost_fn'], lam=lam)
    return (a_net, b_net, zA_feat, p_y1_B)


def evaluate_both(ctx, zA_feat_bits, p_y1_B, downstream_seed):
    X_train, Y_train = make_completed_downstream_training(ctx, zA_feat_bits, p_y1_B)
    model = train_downstream(X_train=X_train, Y_train=Y_train, input_dim=DIM_X + DIM_Z, seed=downstream_seed)
    C_feat = np.concatenate([ctx['XC_std'], ctx['ZC']], axis=1).astype(np.float32)
    T_feat = np.concatenate([ctx['XT_std'], ctx['ZT']], axis=1).astype(np.float32)
    weak_auc = safe_auc(ctx['YC'], predict_score(model, C_feat))
    strong_auc = safe_auc(ctx['YT'], predict_score(model, T_feat))
    auc_B = safe_auc(ctx['YB'], p_y1_B)
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return {'strong_down_auc': float(strong_auc), 'weak_down_auc': float(weak_auc), 'auc_B': float(auc_B)}


def main():
    if SB_LAMBDA is None:
        raise ValueError('Set SB_LAMBDA near the top to your fixed CelebA lambda (e.g., SB_LAMBDA = 1.0) before running.')
    lam = float(SB_LAMBDA)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f'SB_LAMBDA must be finite and > 0, got {lam!r}')
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA GPU {MY_GPU_ID} is unavailable. Use a fresh Jupyter kernel and check CUDA_VISIBLE_DEVICES.')
    seed = int(SEED)
    t0 = time.time()
    print('=' * 90)
    print('CelebA SB | one seed, one fixed lambda | console only')
    print(f'physical GPU={MY_GPU_ID} ({torch.cuda.get_device_name(0)})')
    print(f'seed={seed} | lambda={lam:.12g} | C={C_TOTAL}')
    print('Strong = held-out T; Weak = held-out C; one model fitted for both')
    print('baselines: none | CSV/log writes: none')
    print('=' * 90)
    ctx = prepare_seed_context_weak(seed)
    yC = (ctx['YC'].reshape(-1) > 0.5).astype(np.int64)
    n0 = int((yC == 0).sum())
    n1 = int((yC == 1).sum())
    print(f"A={len(ctx['YA'])} | B={len(ctx['YB'])} | C={len(ctx['YC'])} (0:{n0}, 1:{n1}) | T={len(ctx['YT'])}")
    print(f'X dim={DIM_X} | Z states={Z_STATES}')
    set_seed(seed + SB_SWEEP_RANDOM_OFFSET)
    print('[SB] training one bridge ...', flush=True)
    a_net, b_net, zA_feat, p_y1_B = run_lambda_completion(ctx=ctx, lam=lam)
    print('[SB] imputations complete; training ONE downstream model ...', flush=True)
    downstream_seed = seed + 3000 + 100
    metrics = evaluate_both(ctx, zA_feat, p_y1_B, downstream_seed)
    print('\n' + '=' * 90)
    print(f'CelebA | seed={seed} | lambda={lam:.12g}')
    print(f"{'Method':<12} {'Strong AUC (T)':>19} {'Weak AUC (C=50)':>19} {'AUC_B':>14}")
    print('-' * 90)
    print(f"{'SB':<12} {metrics['strong_down_auc']:>19.6f} {metrics['weak_down_auc']:>19.6f} {metrics['auc_B']:>14.6f}")
    print('-' * 90)
    print(f'elapsed={time.time() - t0:.1f}s')
    print('No files were saved.')
    print('=' * 90)
    del a_net, b_net, zA_feat, p_y1_B, ctx
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return metrics


if __name__ == '__main__':
    results = main()
