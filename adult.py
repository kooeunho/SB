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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
SB_LAMBDA = 3.199267137797384
C_TOTAL = 50
SB_SWEEP_RANDOM_OFFSET = 60000
PATH_X = 'data/adultX.txt'
PATH_Y = 'data/adultY.txt'
PATH_Z = 'data/adultZ.txt'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_A = 16000
N_B = 16000
Z_CODE_K = 32
KMEANS_BATCH = 2048
COST_SHRINK_ALPHA = 200.0
COST_BETA = 1.0
BRIDGE_EPOCHS = 4000
BATCH_SIZE = 512
LR_A = 0.0005
LR_B = 0.0005
WEIGHT_DECAY = 1e-05
SCALE_REG = 0.001
wA = 2.0
wB = 1.0
CLIP_NORM = 5.0
RATIO_EPOCHS = 10
RATIO_BS = 1024
RATIO_LR = 0.001
DOWN_EPOCHS = 250
DOWN_LR = 0.001
DOWN_BATCH = 512
DOWN_PATIENCE = 25
EPS = 1e-08
EPS_LOG = 1e-06


def one_hot_from_codes(codes, K):
    codes = np.asarray(codes).astype(np.int64).reshape(-1)
    out = np.zeros((codes.shape[0], K), dtype=np.float32)
    out[np.arange(codes.shape[0]), codes] = 1.0
    return out


def to_torch(x, dtype=torch.float32):
    return torch.from_numpy(np.asarray(x)).to(DEVICE, dtype=dtype)


class PositiveMLP(nn.Module):

    def __init__(self, input_dim, hidden=256, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):
        return self.net(x).squeeze(-1) + 1e-06


class RatioNetBinary(nn.Module):

    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RatioNetMultiClass(nn.Module):

    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


class DownstreamMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_adult_txt(path_x, path_y, path_z):
    if not (os.path.exists(path_x) and os.path.exists(path_y) and os.path.exists(path_z)):
        raise FileNotFoundError(f'Cannot find required files:\n{path_x}\n{path_y}\n{path_z}')
    X_df = pd.read_csv(path_x)
    Y_df = pd.read_csv(path_y)
    Z_df = pd.read_csv(path_z)
    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = X_df[c].astype(str).str.strip()
    for c in Z_df.columns:
        if Z_df[c].dtype == object:
            Z_df[c] = Z_df[c].astype(str).str.strip()
    y_raw = Y_df.iloc[:, 0].astype(str).str.strip().values
    y = (y_raw == '>50K').astype(np.int64)
    return (X_df, y, Z_df)


def build_preprocessor(num_cols, cat_cols):
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return ColumnTransformer(transformers=[('num', StandardScaler(), num_cols), ('cat', ohe, cat_cols)], remainder='drop')


def split_indices_stratified_weak(y, seed, nA, nB, nC):
    idx_all = np.arange(y.shape[0])
    idx_A, idx_rem = train_test_split(idx_all, train_size=nA, stratify=y, random_state=seed, shuffle=True)
    y_rem = y[idx_rem]
    idx_B, idx_T_full = train_test_split(idx_rem, train_size=nB, stratify=y_rem, random_state=seed + 1, shuffle=True)
    if len(idx_T_full) <= nC:
        raise ValueError(f'T_full has only {len(idx_T_full)} samples, cannot reserve C={nC}.')
    y_T_full = y[idx_T_full]
    idx_C, idx_T = train_test_split(idx_T_full, train_size=nC, stratify=y_T_full, random_state=seed + 2, shuffle=True)
    return (idx_A, idx_B, idx_C, idx_T)


def compute_cost_matrix(XA, YA, XB, codeB, K):
    Xall = np.concatenate([XA, XB], axis=0)
    stdX = Xall.std(axis=0) + 1e-06
    meanX_y = []
    for yv in [0, 1]:
        m = XA[YA.reshape(-1) == yv]
        if m.shape[0] == 0:
            meanX_y.append(Xall.mean(axis=0))
        else:
            meanX_y.append(m.mean(axis=0))
    meanX_y = np.stack(meanX_y, axis=0)
    dx = XB.shape[1]
    mean_all = XB.mean(axis=0)
    mean_code = np.zeros((K, dx), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        mask = codeB == k
        nk = float(mask.sum())
        counts[k] = nk
        if nk > 0:
            mean_code[k] = XB[mask].mean(axis=0)
        else:
            mean_code[k] = mean_all
    w = counts / (counts + COST_SHRINK_ALPHA)
    mean_code = w[:, None] * mean_code + (1.0 - w)[:, None] * mean_all[None, :]
    diff0 = (meanX_y[0][None, :] - mean_code) / stdX[None, :]
    diff1 = (meanX_y[1][None, :] - mean_code) / stdX[None, :]
    c0 = np.sum(diff0 * diff0, axis=1)
    c1 = np.sum(diff1 * diff1, axis=1)
    C = np.stack([c0, c1], axis=0).astype(np.float32) * COST_BETA
    return C


def train_ratio_models(XA, YA, XB, codeB, input_dim, K):
    pyx = RatioNetBinary(input_dim).to(DEVICE)
    pzx = RatioNetMultiClass(input_dim, K).to(DEVICE)
    opt_y = torch.optim.Adam(pyx.parameters(), lr=RATIO_LR)
    opt_z = torch.optim.Adam(pzx.parameters(), lr=RATIO_LR)
    XA_t = to_torch(XA)
    YA_t = to_torch(YA.reshape(-1).astype(np.float32))
    XB_t = to_torch(XB)
    codeB_t = to_torch(codeB.astype(np.int64), dtype=torch.long)
    for _ in range(RATIO_EPOCHS):
        pyx.train()
        perm = torch.randperm(XA_t.size(0), device=DEVICE)
        for i in range(0, XA_t.size(0), RATIO_BS):
            idx = perm[i:i + RATIO_BS]
            opt_y.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(pyx(XA_t[idx]), YA_t[idx])
            loss.backward()
            opt_y.step()
        pzx.train()
        perm = torch.randperm(XB_t.size(0), device=DEVICE)
        for i in range(0, XB_t.size(0), RATIO_BS):
            idx = perm[i:i + RATIO_BS]
            opt_z.zero_grad(set_to_none=True)
            loss = F.cross_entropy(pzx(XB_t[idx]), codeB_t[idx])
            loss.backward()
            opt_z.step()
    pyx.eval()
    pzx.eval()
    return (pyx, pzx)


@torch.no_grad()
def log_pz_given_x_code(pzx, x, z_code):
    logits = pzx(x)
    logp = F.log_softmax(logits, dim=1)
    return logp[torch.arange(x.size(0), device=DEVICE), z_code].clamp(min=-50.0, max=0.0)


@torch.no_grad()
def log_pz_all(pzx, x):
    logits = pzx(x)
    return F.log_softmax(logits, dim=1).clamp(min=-50.0, max=0.0)


def train_downstream(X_train, Y_train, input_dim, seed):
    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32).reshape(-1)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed, shuffle=True)
    Xtr_t = to_torch(X_tr)
    Ytr_t = to_torch(Y_tr.reshape(-1).astype(np.float32))
    Xv_t = to_torch(X_val)
    Yv_t = to_torch(Y_val.reshape(-1).astype(np.float32))
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
    Xt = to_torch(np.asarray(X, dtype=np.float32))
    model.eval()
    return model(Xt).detach().cpu().numpy().reshape(-1)


def make_completed_downstream_training(ctx, z_probs_A, p_y1_B):
    XA_feat = np.concatenate([ctx['XA'], z_probs_A], axis=1).astype(np.float32)
    XB_feat = np.concatenate([ctx['XB'], ctx['ZB_oh']], axis=1).astype(np.float32)
    X_train = np.concatenate([XA_feat, XB_feat], axis=0).astype(np.float32)
    Y_train = np.concatenate([ctx['YA'].reshape(-1), p_y1_B.reshape(-1)], axis=0).astype(np.float32)
    return (X_train, Y_train)


def train_bridge(lam, train_gpu, pyx, pzx, cost_y_code):
    dx = train_gpu['XA'].size(1)
    K = train_gpu['z_st'].size(0)
    a_net = PositiveMLP(dx + 1, hidden=256, dropout_rate=0.2).to(DEVICE)
    b_net = PositiveMLP(dx + K, hidden=256, dropout_rate=0.0).to(DEVICE)
    opt_a = torch.optim.Adam(a_net.parameters(), lr=LR_A, weight_decay=WEIGHT_DECAY)
    opt_b = torch.optim.Adam(b_net.parameters(), lr=LR_B, weight_decay=WEIGHT_DECAY)
    XA = train_gpu['XA']
    YA = train_gpu['YA']
    XB = train_gpu['XB']
    ZB_code = train_gpu['ZB_code']
    ZB_oh = train_gpu['ZB_oh']
    z_st = train_gpu['z_st']
    nA = XA.size(0)
    nB = XB.size(0)
    for _ in range(BRIDGE_EPOCHS):
        ia = torch.randint(0, nA, (BATCH_SIZE,), device=DEVICE)
        ib = torch.randint(0, nB, (BATCH_SIZE,), device=DEVICE)
        x_a = XA[ia]
        y_a = YA[ia].view(-1, 1)
        y_bin = YA[ia].long()
        x_b = XB[ib]
        z_code = ZB_code[ib]
        z_oh = ZB_oh[ib]
        a_val = a_net(torch.cat([x_a, y_a], dim=1))
        x_rep = x_a.unsqueeze(1).expand(-1, K, -1).reshape(-1, dx)
        z_rep = z_st.unsqueeze(0).expand(BATCH_SIZE, -1, -1).reshape(-1, K)
        b_in = b_net(torch.cat([x_rep, z_rep], dim=1)).view(BATCH_SIZE, K)
        c = cost_y_code[y_bin, :]
        logpz_cond_A = log_pz_all(pzx, x_a)
        logw = torch.log(b_in + EPS) - c / lam + logpz_cond_A
        lse = torch.logsumexp(logw, dim=1)
        lossA = torch.mean((a_val * torch.exp(lse) - 1.0) ** 2)
        lossA = lossA + SCALE_REG * (a_val.mean() - 1.0).pow(2)
        opt_a.zero_grad(set_to_none=True)
        (wA * lossA).backward()
        torch.nn.utils.clip_grad_norm_(a_net.parameters(), CLIP_NORM)
        opt_a.step()
        bq = b_net(torch.cat([x_b, z_oh], dim=1))
        log_pz_obs = log_pz_given_x_code(pzx, x_b, z_code)
        b_val = bq * torch.exp(log_pz_obs)
        y_states = torch.tensor([[0.0], [1.0]], device=DEVICE, dtype=torch.float32)
        x2 = x_b.unsqueeze(1).expand(-1, 2, -1).reshape(-1, dx)
        y2 = y_states.unsqueeze(0).expand(BATCH_SIZE, -1, -1).reshape(-1, 1)
        a_in = a_net(torch.cat([x2, y2], dim=1)).view(BATCH_SIZE, 2)
        with torch.no_grad():
            py1 = torch.sigmoid(pyx(x_b)).clamp(EPS_LOG, 1.0 - EPS_LOG)
            log_py = torch.stack([torch.log(1.0 - py1), torch.log(py1)], dim=1)
            log_pz = log_pz_given_x_code(pzx, x_b, z_code).view(BATCH_SIZE, 1)
        log_r = log_py - log_pz
        c0 = cost_y_code[0, z_code]
        c1 = cost_y_code[1, z_code]
        c2 = torch.stack([c0, c1], dim=1)
        logw2 = torch.log(a_in + EPS) + log_r - c2 / lam
        lse2 = torch.logsumexp(logw2, dim=1) - math.log(2.0)
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
def impute_z_probs_A(b_net, train_gpu, cost_y_code, lam, pzx):
    XA = train_gpu['XA']
    YA = train_gpu['YA'].long()
    z_st = train_gpu['z_st']
    dx = XA.size(1)
    K = z_st.size(0)
    nA = XA.size(0)
    bs = 2048
    out = []
    for i0 in range(0, nA, bs):
        i1 = min(i0 + bs, nA)
        x = XA[i0:i1]
        y = YA[i0:i1]
        curr = x.size(0)
        x_rep = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, dx)
        z_rep = z_st.unsqueeze(0).expand(curr, -1, -1).reshape(-1, K)
        bq = b_net(torch.cat([x_rep, z_rep], dim=1)).view(curr, K)
        c = cost_y_code[y, :]
        logpz_cond = log_pz_all(pzx, x)
        logw = torch.log(bq + EPS) - c / lam + logpz_cond
        probs = torch.softmax(logw, dim=1)
        out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def impute_y_probs_B(a_net, train_gpu, pyx, pzx, cost_y_code, lam):
    XB = train_gpu['XB']
    ZB_code = train_gpu['ZB_code']
    dx = XB.size(1)
    nB = XB.size(0)
    bs = 4096
    out = []
    for i0 in range(0, nB, bs):
        i1 = min(i0 + bs, nB)
        x = XB[i0:i1]
        z_code = ZB_code[i0:i1]
        curr = x.size(0)
        y_states = torch.tensor([[0.0], [1.0]], device=DEVICE, dtype=torch.float32)
        x2 = x.unsqueeze(1).expand(-1, 2, -1).reshape(-1, dx)
        y2 = y_states.unsqueeze(0).expand(curr, -1, -1).reshape(-1, 1)
        a_in = a_net(torch.cat([x2, y2], dim=1)).view(curr, 2)
        py1 = torch.sigmoid(pyx(x)).clamp(EPS_LOG, 1.0 - EPS_LOG)
        log_py = torch.stack([torch.log(1.0 - py1), torch.log(py1)], dim=1)
        log_pz = log_pz_given_x_code(pzx, x, z_code).view(curr, 1)
        log_r = log_py - log_pz
        c0 = cost_y_code[0, z_code]
        c1 = cost_y_code[1, z_code]
        c2 = torch.stack([c0, c1], dim=1)
        logw = torch.log(a_in + EPS) + log_r - c2 / lam
        probs = torch.softmax(logw, dim=1)
        out.append(probs[:, 1].cpu().numpy().astype(np.float32))
    p1 = np.concatenate(out, axis=0)
    return np.clip(p1, 1e-06, 1.0 - 1e-06)


def prepare_seed_context_weak(seed, X_df, y_all, Z_df):
    set_seed(seed)
    idx_A, idx_B, idx_C, idx_T = split_indices_stratified_weak(y=y_all, seed=seed, nA=N_A, nB=N_B, nC=C_TOTAL)
    X_A_df = X_df.iloc[idx_A].copy()
    X_B_df = X_df.iloc[idx_B].copy()
    X_C_df = X_df.iloc[idx_C].copy()
    X_T_df = X_df.iloc[idx_T].copy()
    Z_A_df = Z_df.iloc[idx_A].copy()
    Z_B_df = Z_df.iloc[idx_B].copy()
    Z_C_df = Z_df.iloc[idx_C].copy()
    Z_T_df = Z_df.iloc[idx_T].copy()
    YA = y_all[idx_A].astype(np.int64)
    YB = y_all[idx_B].astype(np.int64)
    YC = y_all[idx_C].astype(np.int64)
    YT = y_all[idx_T].astype(np.int64)
    cols_X_num = [c for c in ['age'] if c in X_df.columns]
    if len(cols_X_num) == 0:
        cols_X_num = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    cols_X_cat = [c for c in X_df.columns if c not in cols_X_num]
    X_pre = build_preprocessor(cols_X_num, cols_X_cat)
    X_pre.fit(pd.concat([X_A_df, X_B_df], axis=0))
    XA = X_pre.transform(X_A_df).astype(np.float32)
    XB = X_pre.transform(X_B_df).astype(np.float32)
    XC = X_pre.transform(X_C_df).astype(np.float32)
    XT = X_pre.transform(X_T_df).astype(np.float32)
    dx = XA.shape[1]
    cols_Z_num = [c for c in ['education-num', 'hours-per-week', 'capital-gain', 'capital-loss'] if c in Z_df.columns]
    if len(cols_Z_num) == 0:
        cols_Z_num = [c for c in Z_df.columns if pd.api.types.is_numeric_dtype(Z_df[c])]
    cols_Z_cat = [c for c in Z_df.columns if c not in cols_Z_num]
    Z_pre = build_preprocessor(cols_Z_num, cols_Z_cat)
    Z_pre.fit(Z_B_df)
    ZA_proc = Z_pre.transform(Z_A_df).astype(np.float32)
    ZB_proc = Z_pre.transform(Z_B_df).astype(np.float32)
    ZC_proc = Z_pre.transform(Z_C_df).astype(np.float32)
    ZT_proc = Z_pre.transform(Z_T_df).astype(np.float32)
    kmZ = MiniBatchKMeans(n_clusters=Z_CODE_K, random_state=seed, batch_size=KMEANS_BATCH, n_init=10, max_iter=200, reassignment_ratio=0.01)
    kmZ.fit(ZB_proc)
    codeA = kmZ.predict(ZA_proc).astype(np.int64)
    codeB = kmZ.predict(ZB_proc).astype(np.int64)
    codeC = kmZ.predict(ZC_proc).astype(np.int64)
    codeT = kmZ.predict(ZT_proc).astype(np.int64)
    ZA_oh = one_hot_from_codes(codeA, Z_CODE_K)
    ZB_oh = one_hot_from_codes(codeB, Z_CODE_K)
    ZC_oh = one_hot_from_codes(codeC, Z_CODE_K)
    ZT_oh = one_hot_from_codes(codeT, Z_CODE_K)
    Cmat = compute_cost_matrix(XA, YA, XB, codeB, Z_CODE_K)
    pyx, pzx = train_ratio_models(XA=XA, YA=YA, XB=XB, codeB=codeB, input_dim=dx, K=Z_CODE_K)
    train_gpu = {'XA': to_torch(XA), 'YA': to_torch(YA.astype(np.float32)), 'XB': to_torch(XB), 'ZB_code': to_torch(codeB.astype(np.int64), dtype=torch.long), 'ZB_oh': to_torch(ZB_oh), 'z_st': torch.eye(Z_CODE_K, device=DEVICE, dtype=torch.float32)}
    cost_y_code = to_torch(Cmat)
    return {'seed': seed, 'K': Z_CODE_K, 'dx': dx, 'XA': XA, 'XB': XB, 'XC': XC, 'XT': XT, 'YA': YA, 'YB': YB, 'YC': YC, 'YT': YT, 'codeA_true': codeA, 'codeB': codeB, 'codeC': codeC, 'codeT': codeT, 'ZA_oh_true': ZA_oh, 'ZB_oh': ZB_oh, 'ZC_oh': ZC_oh, 'ZT_oh': ZT_oh, 'train_gpu': train_gpu, 'cost_y_code': cost_y_code, 'cost_y_code_np': Cmat, 'pyx': pyx, 'pzx': pzx, 'idx_A': idx_A, 'idx_B': idx_B, 'idx_C': idx_C, 'idx_T': idx_T}


def run_lambda_completion(ctx, lam):
    a_net, b_net = train_bridge(lam=lam, train_gpu=ctx['train_gpu'], pyx=ctx['pyx'], pzx=ctx['pzx'], cost_y_code=ctx['cost_y_code'])
    z_probs_A = impute_z_probs_A(b_net=b_net, train_gpu=ctx['train_gpu'], cost_y_code=ctx['cost_y_code'], lam=lam, pzx=ctx['pzx'])
    p_y1_B = impute_y_probs_B(a_net=a_net, train_gpu=ctx['train_gpu'], pyx=ctx['pyx'], pzx=ctx['pzx'], cost_y_code=ctx['cost_y_code'], lam=lam)
    return (a_net, b_net, z_probs_A, p_y1_B)


def evaluate_both(ctx, z_probs_A, p_y1_B, downstream_seed):
    X_train, Y_train = make_completed_downstream_training(ctx, z_probs_A, p_y1_B)
    model = train_downstream(X_train=X_train, Y_train=Y_train, input_dim=ctx['dx'] + ctx['K'], seed=downstream_seed)
    C_feat = np.concatenate([ctx['XC'], ctx['ZC_oh']], axis=1).astype(np.float32)
    T_feat = np.concatenate([ctx['XT'], ctx['ZT_oh']], axis=1).astype(np.float32)
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
        raise ValueError('Set SB_LAMBDA near the top to your fixed Adult lambda (e.g., SB_LAMBDA = 1.0) before running.')
    lam = float(SB_LAMBDA)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f'SB_LAMBDA must be finite and > 0, got {lam!r}')
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA GPU {MY_GPU_ID} is unavailable. Use a fresh Jupyter kernel and check CUDA_VISIBLE_DEVICES.')
    seed = int(SEED)
    t0 = time.time()
    print('=' * 90)
    print('Adult SB | one seed, one fixed lambda | console only')
    print(f'physical GPU={MY_GPU_ID} ({torch.cuda.get_device_name(0)})')
    print(f'seed={seed} | lambda={lam:.12g} | C={C_TOTAL}')
    print('Strong = held-out T; Weak = held-out C; one model fitted for both')
    print('baselines: none | CSV/log writes: none')
    print('=' * 90)
    X_df, y_all, Z_df = load_adult_txt(PATH_X, PATH_Y, PATH_Z)
    ctx = prepare_seed_context_weak(seed, X_df, y_all, Z_df)
    n0 = int((ctx['YC'] == 0).sum())
    n1 = int((ctx['YC'] == 1).sum())
    print(f"A={len(ctx['YA'])} | B={len(ctx['YB'])} | C={len(ctx['YC'])} (0:{n0}, 1:{n1}) | T={len(ctx['YT'])}")
    print(f"X dim={ctx['dx']} | Z codes={ctx['K']}")
    set_seed(seed + SB_SWEEP_RANDOM_OFFSET)
    print('[SB] training one bridge ...', flush=True)
    a_net, b_net, z_probs_A, p_y1_B = run_lambda_completion(ctx=ctx, lam=lam)
    print('[SB] imputations complete; training ONE downstream model ...', flush=True)
    downstream_seed = seed + 3000 + 100 + ctx['K']
    metrics = evaluate_both(ctx, z_probs_A, p_y1_B, downstream_seed)
    print('\n' + '=' * 90)
    print(f'Adult | seed={seed} | lambda={lam:.12g}')
    print(f"{'Method':<12} {'Strong AUC (T)':>19} {'Weak AUC (C=50)':>19} {'AUC_B':>14}")
    print('-' * 90)
    print(f"{'SB':<12} {metrics['strong_down_auc']:>19.6f} {metrics['weak_down_auc']:>19.6f} {metrics['auc_B']:>14.6f}")
    print('-' * 90)
    print(f'elapsed={time.time() - t0:.1f}s')
    print('No files were saved.')
    print('=' * 90)
    del a_net, b_net, z_probs_A, p_y1_B, ctx
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return metrics


if __name__ == '__main__':
    results = main()
