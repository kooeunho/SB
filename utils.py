import math
import time
import copy
import gc
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    return a


def one_hot(idx, n_classes):
    idx = np.asarray(idx).astype(int).ravel()
    out = np.zeros((idx.size, n_classes), dtype=np.float32)
    out[np.arange(idx.size), idx] = 1.0
    return out


def digitize_like_go(x_1d, edges):
    x_1d = np.asarray(x_1d).ravel()
    return np.digitize(x_1d, edges[1:-1], right=False).astype(int)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(as_2d(y_true), as_2d(y_pred))))


def r2(y_true, y_pred):
    return float(r2_score(as_2d(y_true), as_2d(y_pred)))


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    if len(vals) == 1:
        return (float(vals[0]), 0.0)
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)))


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, scores))


def _matrix_sqrt_psd(M):
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w) @ V.T


def gaussian_w2_sq(mu1, cov1, mu2, cov2):
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    mu2 = np.asarray(mu2, dtype=np.float64).ravel()
    C1 = np.asarray(cov1, dtype=np.float64)
    C2 = np.asarray(cov2, dtype=np.float64)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    C1_half = _matrix_sqrt_psd(C1)
    inner = C1_half @ C2 @ C1_half
    inner_half = _matrix_sqrt_psd(inner)
    cross = float(np.trace(C1 + C2 - 2.0 * inner_half))
    return float(max(mean_term + cross, 0.0))


def full_joint_w2_sq_from_samples(D_hat, D_true, n_sub=None, seed=0):
    D_hat = np.asarray(D_hat, dtype=np.float64)
    D_true = np.asarray(D_true, dtype=np.float64)
    if n_sub is not None:
        rng = np.random.RandomState(seed)
        if D_hat.shape[0] > n_sub:
            idx = rng.choice(D_hat.shape[0], n_sub, replace=False)
            D_hat = D_hat[idx]
        if D_true.shape[0] > n_sub:
            idx = rng.choice(D_true.shape[0], n_sub, replace=False)
            D_true = D_true[idx]
    mu_t = D_true.mean(axis=0)
    sd_t = D_true.std(axis=0) + 1e-12
    Dh = (D_hat - mu_t) / sd_t
    Dt = (D_true - mu_t) / sd_t
    mu_h = Dh.mean(axis=0)
    mu_tt = Dt.mean(axis=0)
    cov_h = np.cov(Dh, rowvar=False)
    cov_t = np.cov(Dt, rowvar=False)
    return gaussian_w2_sq(mu_h, cov_h, mu_tt, cov_t)


class PositiveMLP(nn.Module):

    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):
        return self.net(x) + 1e-06


class SynDownstreamMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class CondMeanMLP(nn.Module):

    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def _build_original_go_lp(nX, nY, nZ, mu_XY, mu_XZ_tilde, cost_yz, mu_X, lam, alpha, graph_edges):
    n_gamma = nX * nY * nZ
    n_eXY = nX * nY
    n_eXZ = nX * nZ
    n_edges = len(graph_edges)
    n_r = n_edges * nY * nZ
    off_g = 0
    off_eXY = off_g + n_gamma
    off_eXZ = off_eXY + n_eXY
    off_abs_eXY = off_eXZ + n_eXZ
    off_abs_eXZ = off_abs_eXY + n_eXY
    off_r = off_abs_eXZ + n_eXZ
    n_total = off_r + n_r
    c_obj = np.zeros(n_total, dtype=np.float64)
    c_obj[off_g:off_g + n_gamma] = np.broadcast_to(cost_yz[None, :, :], (nX, nY, nZ)).ravel()
    if n_edges > 0:
        edge_weights = np.asarray([e[2] for e in graph_edges], dtype=np.float64)
        c_obj[off_r:off_r + n_r] = float(lam) * np.repeat(edge_weights, nY * nZ)
    eq_rows = []
    eq_cols = []
    eq_data = []
    b_eq = []
    row = 0
    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                gidx = off_g + x * nY * nZ + y * nZ + z
                eq_rows.append(row)
                eq_cols.append(gidx)
                eq_data.append(1.0)
            eidx = off_eXY + x * nY + y
            eq_rows.append(row)
            eq_cols.append(eidx)
            eq_data.append(-1.0)
            b_eq.append(float(mu_XY[x, y]))
            row += 1
    for x in range(nX):
        for z in range(nZ):
            for y in range(nY):
                gidx = off_g + x * nY * nZ + y * nZ + z
                eq_rows.append(row)
                eq_cols.append(gidx)
                eq_data.append(1.0)
            eidx = off_eXZ + x * nZ + z
            eq_rows.append(row)
            eq_cols.append(eidx)
            eq_data.append(-1.0)
            b_eq.append(float(mu_XZ_tilde[x, z]))
            row += 1
    for i in range(n_eXY):
        eq_rows.append(row)
        eq_cols.append(off_eXY + i)
        eq_data.append(1.0)
    b_eq.append(0.0)
    row += 1
    for i in range(n_eXZ):
        eq_rows.append(row)
        eq_cols.append(off_eXZ + i)
        eq_data.append(1.0)
    b_eq.append(0.0)
    row += 1
    A_eq = coo_matrix((np.asarray(eq_data), (np.asarray(eq_rows), np.asarray(eq_cols))), shape=(row, n_total)).tocsr()
    b_eq = np.asarray(b_eq, dtype=np.float64)
    ub_rows = []
    ub_cols = []
    ub_data = []
    b_ub = []
    row = 0
    for i in range(n_eXY):
        eidx = off_eXY + i
        aidx = off_abs_eXY + i
        ub_rows += [row, row]
        ub_cols += [eidx, aidx]
        ub_data += [1.0, -1.0]
        b_ub.append(0.0)
        row += 1
        ub_rows += [row, row]
        ub_cols += [eidx, aidx]
        ub_data += [-1.0, -1.0]
        b_ub.append(0.0)
        row += 1
    for i in range(n_eXZ):
        eidx = off_eXZ + i
        aidx = off_abs_eXZ + i
        ub_rows += [row, row]
        ub_cols += [eidx, aidx]
        ub_data += [1.0, -1.0]
        b_ub.append(0.0)
        row += 1
        ub_rows += [row, row]
        ub_cols += [eidx, aidx]
        ub_data += [-1.0, -1.0]
        b_ub.append(0.0)
        row += 1
    for i in range(n_eXY):
        ub_rows.append(row)
        ub_cols.append(off_abs_eXY + i)
        ub_data.append(1.0)
    b_ub.append(float(alpha))
    row += 1
    for i in range(n_eXZ):
        ub_rows.append(row)
        ub_cols.append(off_abs_eXZ + i)
        ub_data.append(1.0)
    b_ub.append(float(alpha))
    row += 1
    mu_safe = np.maximum(np.asarray(mu_X, dtype=np.float64), 1e-12)
    for e_idx, (i, j, _w) in enumerate(graph_edges):
        inv_i = 1.0 / mu_safe[i]
        inv_j = 1.0 / mu_safe[j]
        for y in range(nY):
            for z in range(nZ):
                g_i = off_g + i * nY * nZ + y * nZ + z
                g_j = off_g + j * nY * nZ + y * nZ + z
                r_idx = off_r + e_idx * nY * nZ + y * nZ + z
                ub_rows += [row, row, row]
                ub_cols += [g_i, g_j, r_idx]
                ub_data += [inv_i, -inv_j, -1.0]
                b_ub.append(0.0)
                row += 1
                ub_rows += [row, row, row]
                ub_cols += [g_i, g_j, r_idx]
                ub_data += [-inv_i, inv_j, -1.0]
                b_ub.append(0.0)
                row += 1
    A_ub = coo_matrix((np.asarray(ub_data), (np.asarray(ub_rows), np.asarray(ub_cols))), shape=(row, n_total)).tocsr()
    b_ub = np.asarray(b_ub, dtype=np.float64)
    bounds = []
    bounds += [(0.0, None)] * n_gamma
    bounds += [(None, None)] * n_eXY
    bounds += [(None, None)] * n_eXZ
    bounds += [(0.0, None)] * n_eXY
    bounds += [(0.0, None)] * n_eXZ
    bounds += [(0.0, None)] * n_r
    return (c_obj, A_eq, b_eq, A_ub, b_ub, bounds, n_gamma)


class GORJointPaper:

    def __init__(self, n_bins_x=4, n_bins_y=3, n_bins_z=4, standardize_x=True, k_graph=5, solver='HIGHS', solver_verbose=False, random_state=0, cost_sample_cap=512, lp_time_limit=300.0):
        self.alpha = 0.4
        self.lam = 0.1
        self.n_bins_x = int(n_bins_x)
        self.n_bins_y = int(n_bins_y)
        self.n_bins_z = int(n_bins_z)
        self.standardize_x = bool(standardize_x)
        self.k_graph = int(k_graph)
        self.solver = str(solver)
        self.solver_verbose = bool(solver_verbose)
        self.cost_sample_cap = int(cost_sample_cap)
        self.lp_time_limit = float(lp_time_limit)
        self.rng = np.random.default_rng(int(random_state))
        self.x_scaler = StandardScaler() if self.standardize_x else None
        self.edges_ = {}
        self.nX_ = None
        self.nY_ = None
        self.nZ_ = None
        self.gamma_ = None
        self.cond_prob_ = None
        self.x_state_repr_ = None
        self.graph_edges_ = None
        self.status_ = None

    @staticmethod
    def _make_quantile_edges(arr_1d, n_bins):
        arr_1d = np.asarray(arr_1d).ravel()
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(arr_1d, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges = np.unique(edges)
        if edges.size < 3:
            mn = float(np.min(arr_1d))
            mx = float(np.max(arr_1d))
            if mn == mx:
                mn -= 1.0
                mx += 1.0
            edges = np.array([-np.inf, 0.5 * (mn + mx), np.inf])
        return edges.astype(float)

    @staticmethod
    def _to_bin_1d(x_1d, edges):
        return np.digitize(np.asarray(x_1d).ravel(), edges[1:-1], right=False).astype(int)

    def _fit_binning(self, XA, YA, XB, ZB):
        XA = as_2d(XA)
        XB = as_2d(XB)
        YA = as_2d(YA)
        ZB = as_2d(ZB)
        if self.standardize_x:
            self.x_scaler.fit(np.vstack([XA, XB]))
            XA_s = self.x_scaler.transform(XA)
            XB_s = self.x_scaler.transform(XB)
        else:
            XA_s = XA
            XB_s = XB
        d = XA_s.shape[1]
        self.edges_['X'] = []
        for j in range(d):
            edges_j = self._make_quantile_edges(np.concatenate([XA_s[:, j], XB_s[:, j]]), self.n_bins_x)
            self.edges_['X'].append(edges_j)
        self.edges_['Y'] = self._make_quantile_edges(YA.ravel(), self.n_bins_y)
        self.edges_['Z'] = self._make_quantile_edges(ZB.ravel(), self.n_bins_z)
        return (XA_s, XB_s)

    def _x_to_state(self, X_s):
        X_s = as_2d(X_s)
        d = X_s.shape[1]
        binned = []
        bins_per_dim = []
        for j in range(d):
            edges_j = self.edges_['X'][j]
            bj = self._to_bin_1d(X_s[:, j], edges_j)
            binned.append(bj)
            bins_per_dim.append(len(edges_j) - 1)
        state = np.zeros(X_s.shape[0], dtype=int)
        mult = 1
        for j in reversed(range(d)):
            state += binned[j] * mult
            mult *= bins_per_dim[j]
        return (state, int(mult), bins_per_dim)

    @staticmethod
    def _state_representatives(X_s_all, x_state_all, nX):
        d = X_s_all.shape[1]
        reps = np.zeros((nX, d), dtype=float)
        counts = np.zeros(nX, dtype=int)
        for xs, x in zip(x_state_all, X_s_all):
            reps[xs] += x
            counts[xs] += 1
        for i in range(nX):
            if counts[i] > 0:
                reps[i] /= counts[i]
        return reps

    @staticmethod
    def _mean_pairwise_euclidean(A, B, chunk=2000):
        if A.size == 0 or B.size == 0:
            return 0.0
        s = 0.0
        cnt = 0
        for i0 in range(0, A.shape[0], chunk):
            Ab = A[i0:i0 + chunk]
            A2 = np.sum(Ab * Ab, axis=1, keepdims=True)
            B2 = np.sum(B * B, axis=1, keepdims=True).T
            D2 = np.maximum(A2 + B2 - 2.0 * (Ab @ B.T), 0.0)
            D = np.sqrt(D2)
            s += float(D.sum())
            cnt += int(D.size)
        return s / max(cnt, 1)

    def _estimate_cost_yz(self, XA_s, yA_bin, XB_s, zB_bin):
        nY = self.nY_
        nZ = self.nZ_
        c = np.zeros((nY, nZ), dtype=float)
        idxA_by_y = [np.where(yA_bin == y)[0] for y in range(nY)]
        idxB_by_z = [np.where(zB_bin == z)[0] for z in range(nZ)]
        for y in range(nY):
            IA = idxA_by_y[y]
            if IA.size == 0:
                continue
            XA_y_full = XA_s[IA]
            if XA_y_full.shape[0] > self.cost_sample_cap:
                selA = self.rng.choice(XA_y_full.shape[0], size=self.cost_sample_cap, replace=False)
                XA_y = XA_y_full[selA]
            else:
                XA_y = XA_y_full
            for z in range(nZ):
                IB = idxB_by_z[z]
                if IB.size == 0:
                    c[y, z] = 0.0
                    continue
                XB_z_full = XB_s[IB]
                if XB_z_full.shape[0] > self.cost_sample_cap:
                    selB = self.rng.choice(XB_z_full.shape[0], size=self.cost_sample_cap, replace=False)
                    XB_z = XB_z_full[selB]
                else:
                    XB_z = XB_z_full
                c[y, z] = self._mean_pairwise_euclidean(XA_y, XB_z)
        return c

    def _build_knn_graph(self, x_reps):
        nX = x_reps.shape[0]
        if nX <= 1:
            return []
        k_eff = min(self.k_graph + 1, nX)
        nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean')
        nn.fit(x_reps)
        dist, ind = nn.kneighbors(x_reps, return_distance=True)
        edges = {}
        for i in range(nX):
            for d, j in zip(dist[i, 1:], ind[i, 1:]):
                ii, jj = (i, j) if i < j else (j, i)
                w = 1.0 / max(float(d), 1e-12)
                edges[ii, jj] = max(edges.get((ii, jj), 0.0), w)
        return [(i, j, w) for (i, j), w in edges.items()]

    def fit(self, XA, YA, XB, ZB):
        XA = as_2d(XA)
        YA = as_2d(YA)
        XB = as_2d(XB)
        ZB = as_2d(ZB)
        XA_s, XB_s = self._fit_binning(XA, YA, XB, ZB)
        xA_state, nX, _ = self._x_to_state(XA_s)
        xB_state, _, _ = self._x_to_state(XB_s)
        yA_bin = self._to_bin_1d(YA, self.edges_['Y'])
        zB_bin = self._to_bin_1d(ZB, self.edges_['Z'])
        nY = len(self.edges_['Y']) - 1
        nZ = len(self.edges_['Z']) - 1
        self.nX_ = nX
        self.nY_ = nY
        self.nZ_ = nZ
        X_all = np.vstack([XA_s, XB_s])
        x_state_all = np.concatenate([xA_state, xB_state])
        self.x_state_repr_ = self._state_representatives(X_all, x_state_all, nX)
        self.graph_edges_ = self._build_knn_graph(self.x_state_repr_)
        mu_XY = np.zeros((nX, nY), dtype=float)
        mu_XZ_B = np.zeros((nX, nZ), dtype=float)
        for xs, y in zip(xA_state, yA_bin):
            mu_XY[xs, y] += 1.0
        for xs, z in zip(xB_state, zB_bin):
            mu_XZ_B[xs, z] += 1.0
        mu_XY /= max(mu_XY.sum(), 1.0)
        mu_XZ_B /= max(mu_XZ_B.sum(), 1.0)
        mu_XA = mu_XY.sum(axis=1)
        mu_XB = mu_XZ_B.sum(axis=1)
        mu_XB_sm = mu_XB + 1e-12
        mu_XB_sm /= mu_XB_sm.sum()
        mu_XZ_tilde = np.zeros((nX, nZ), dtype=float)
        for x in range(nX):
            mu_XZ_tilde[x, :] = mu_XZ_B[x, :] * (mu_XA[x] / mu_XB_sm[x])
        s_tilde = mu_XZ_tilde.sum()
        if s_tilde > 0:
            mu_XZ_tilde /= s_tilde
        cost_yz = self._estimate_cost_yz(XA_s, yA_bin, XB_s, zB_bin)
        c_obj, A_eq, b_eq, A_ub, b_ub, bounds, n_gamma = _build_original_go_lp(nX=nX, nY=nY, nZ=nZ, mu_XY=mu_XY, mu_XZ_tilde=mu_XZ_tilde, cost_yz=cost_yz, mu_X=mu_XA, lam=self.lam, alpha=self.alpha, graph_edges=self.graph_edges_)
        options = {'presolve': True, 'time_limit': self.lp_time_limit, 'disp': bool(self.solver_verbose)}
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options=options)
        if not res.success or res.x is None:
            raise RuntimeError(f'Original-GO LP failed under SciPy/HiGHS: status={res.status}, message={res.message}')
        gamma_val = res.x[:n_gamma].reshape(nX, nY, nZ)
        gamma_val = np.maximum(gamma_val, 0.0)
        self.gamma_ = gamma_val
        self.status_ = 'optimal'
        denom = mu_XY[:, :, None]
        cond = np.divide(gamma_val, denom, out=np.zeros_like(gamma_val), where=denom > 0)
        unseen = denom <= 0
        if np.any(unseen):
            cond[unseen.repeat(nZ, axis=2)] = 1.0 / nZ
        s = cond.sum(axis=2, keepdims=True)
        cond = np.divide(cond, np.maximum(s, 1e-12))
        self.cond_prob_ = cond
        return self

    def predict_proba(self, XA, YA):
        XA = as_2d(XA)
        YA = as_2d(YA)
        XA_s = self.x_scaler.transform(XA) if self.standardize_x else XA
        x_state, _, _ = self._x_to_state(XA_s)
        y_bin = self._to_bin_1d(YA, self.edges_['Y'])
        probs = self.cond_prob_[x_state, y_bin, :]
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs

    def get_binning(self):
        return self.edges_


def go_impute_YB_continuous_from_gamma(go, XB_raw, ZB_raw, YA_raw):
    edges = go.get_binning()
    edges_y = edges['Y']
    edges_z = edges['Z']
    yA_bin = digitize_like_go(YA_raw, edges_y)
    nY = len(edges_y) - 1
    rep_y = np.zeros((nY,), dtype=np.float64)
    global_mean = float(np.mean(YA_raw))
    for k in range(nY):
        mask = yA_bin == k
        if np.any(mask):
            rep_y[k] = float(np.mean(YA_raw[mask]))
        else:
            rep_y[k] = global_mean
    XB_raw = as_2d(XB_raw)
    ZB_raw = as_2d(ZB_raw)
    XB_s = go.x_scaler.transform(XB_raw) if go.standardize_x else XB_raw
    x_state, _, _ = go._x_to_state(XB_s)
    z_bin = go._to_bin_1d(ZB_raw, edges_z)
    gamma = go.gamma_
    gamma_xz = gamma.sum(axis=1)
    gamma_sel = gamma[x_state, :, z_bin]
    denom = gamma_xz[x_state, z_bin].reshape(-1, 1)
    p_y = np.divide(gamma_sel, denom, out=np.zeros_like(gamma_sel), where=denom > 0)
    unseen = (denom <= 0).ravel()
    if np.any(unseen):
        p_y[unseen, :] = 1.0 / nY
    y_hat = (p_y @ rep_y.reshape(-1, 1)).astype(np.float32)
    return y_hat


def go_impute_ZA_continuous_from_gamma(go, XA_raw, YA_raw, ZB_raw):
    edges = go.get_binning()
    edges_z = edges['Z']
    zB_bin = digitize_like_go(ZB_raw, edges_z)
    nZ = len(edges_z) - 1
    rep_z = np.zeros((nZ,), dtype=np.float64)
    global_mean = float(np.mean(ZB_raw))
    for k in range(nZ):
        mask = zB_bin == k
        rep_z[k] = float(np.mean(ZB_raw[mask])) if np.any(mask) else global_mean
    pz = go.predict_proba(XA_raw, YA_raw)
    z_hat = (pz @ rep_z.reshape(-1, 1)).astype(np.float32)
    return z_hat


class MixedMethodsImputerFast:

    def __init__(self, random_state=1):
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.reg_Z = None
        self.sigma_Z = None
        self.reg_Y = None
        self.sigma_Y = None

    def fit(self, XA, YA_std, XB, ZB_std):
        XA = as_2d(XA)
        YA_std = as_2d(YA_std)
        XB = as_2d(XB)
        ZB_std = as_2d(ZB_std)
        self.reg_Z = LinearRegression()
        self.reg_Z.fit(XB, ZB_std.ravel())
        resid_Z = ZB_std.ravel() - self.reg_Z.predict(XB)
        self.sigma_Z = float(np.sqrt(max(np.var(resid_Z), 1e-12)))
        self.reg_Y = LinearRegression()
        self.reg_Y.fit(XA, YA_std.ravel())
        resid_Y = YA_std.ravel() - self.reg_Y.predict(XA)
        self.sigma_Y = float(np.sqrt(max(np.var(resid_Y), 1e-12)))
        return self

    def _noisy_pred(self, X, reg, sigma, scale=1.0):
        X = as_2d(X)
        mean = reg.predict(X).reshape(-1, 1)
        noise = self.rng.normal(0.0, sigma * scale, size=mean.shape)
        return mean + noise

    @staticmethod
    def _nn_1d(query, candidates):
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(as_2d(candidates))
        _, ind = nn.kneighbors(as_2d(query), n_neighbors=1)
        return ind.ravel()

    def impute_mm1(self, XA, XB, ZB):
        ztilde_A = self.reg_Z.predict(as_2d(XA)).reshape(-1, 1)
        ztilde_B = self.reg_Z.predict(as_2d(XB)).reshape(-1, 1)
        idx = self._nn_1d(ztilde_A, ztilde_B)
        return as_2d(ZB)[idx]

    def impute_mm2(self, XA, ZB):
        ztilde_A = self.reg_Z.predict(as_2d(XA)).reshape(-1, 1)
        idx = self._nn_1d(ztilde_A, ZB)
        return as_2d(ZB)[idx]

    def impute_mm3(self, XA, ZB, noise_scale=1.0):
        ztilde_A = self._noisy_pred(XA, self.reg_Z, self.sigma_Z, scale=noise_scale)
        idx = self._nn_1d(ztilde_A, ZB)
        return as_2d(ZB)[idx]

    def impute_mm4(self, XA, ZB, noise_scale=1.0):
        XA = as_2d(XA)
        ZB = as_2d(ZB)
        nA = XA.shape[0]
        nB = ZB.shape[0]
        if nA != nB:
            return self.impute_mm3(XA, ZB, noise_scale=noise_scale)
        ztilde_A = self._noisy_pred(XA, self.reg_Z, self.sigma_Z, scale=noise_scale).ravel()
        zB = ZB.ravel()
        orderA = np.argsort(ztilde_A)
        orderB = np.argsort(zB)
        assign = np.empty(nA, dtype=int)
        assign[orderA] = orderB
        return ZB[assign]

    def impute_mm5(self, XA, YA, XB, ZB, noise_scale=1.0):
        XA = as_2d(XA)
        YA = as_2d(YA)
        XB = as_2d(XB)
        ZB = as_2d(ZB)
        ztilde_A = self._noisy_pred(XA, self.reg_Z, self.sigma_Z, scale=noise_scale)
        ytilde_B = self._noisy_pred(XB, self.reg_Y, self.sigma_Y, scale=noise_scale)
        A_feat = np.concatenate([YA, ztilde_A], axis=1)
        B_feat = np.concatenate([ytilde_B, ZB], axis=1)
        mean = B_feat.mean(axis=0, keepdims=True)
        std = B_feat.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        A_scaled = (A_feat - mean) / std
        B_scaled = (B_feat - mean) / std
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(B_scaled)
        _, ind = nn.kneighbors(A_scaled, n_neighbors=1)
        idx = ind.ravel()
        return ZB[idx]

    def impute_y_mm1(self, XA, YA, XB):
        ytilde_A = self.reg_Y.predict(as_2d(XA)).reshape(-1, 1)
        ytilde_B = self.reg_Y.predict(as_2d(XB)).reshape(-1, 1)
        idx = self._nn_1d(ytilde_B, ytilde_A)
        return as_2d(YA)[idx]

    def impute_y_mm2(self, YA, XB):
        ytilde_B = self.reg_Y.predict(as_2d(XB)).reshape(-1, 1)
        idx = self._nn_1d(ytilde_B, YA)
        return as_2d(YA)[idx]

    def impute_y_mm3(self, YA, XB, noise_scale=1.0):
        ytilde_B = self._noisy_pred(XB, self.reg_Y, self.sigma_Y, scale=noise_scale)
        idx = self._nn_1d(ytilde_B, YA)
        return as_2d(YA)[idx]

    def impute_y_mm4(self, YA, XB, noise_scale=1.0):
        YA = as_2d(YA)
        XB = as_2d(XB)
        nA = YA.shape[0]
        nB = XB.shape[0]
        if nA != nB:
            return self.impute_y_mm3(YA, XB, noise_scale=noise_scale)
        ytilde_B = self._noisy_pred(XB, self.reg_Y, self.sigma_Y, scale=noise_scale).ravel()
        yA = YA.ravel()
        orderB = np.argsort(ytilde_B)
        orderA = np.argsort(yA)
        assign = np.empty(nB, dtype=int)
        assign[orderB] = orderA
        return YA[assign]

    def impute_y_mm5(self, XA, YA, XB, ZB, noise_scale=1.0):
        XA = as_2d(XA)
        YA = as_2d(YA)
        XB = as_2d(XB)
        ZB = as_2d(ZB)
        ztilde_A = self._noisy_pred(XA, self.reg_Z, self.sigma_Z, scale=noise_scale)
        ytilde_B = self._noisy_pred(XB, self.reg_Y, self.sigma_Y, scale=noise_scale)
        A_feat = np.concatenate([ztilde_A, YA], axis=1)
        B_feat = np.concatenate([ZB, ytilde_B], axis=1)
        mean = A_feat.mean(axis=0, keepdims=True)
        std = A_feat.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        A_scaled = (A_feat - mean) / std
        B_scaled = (B_feat - mean) / std
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(A_scaled)
        _, ind = nn.kneighbors(B_scaled, n_neighbors=1)
        idx = ind.ravel()
        return YA[idx]


@dataclass(frozen=True)
class SyntheticConfig:
    name: str
    sb_lambda: float
    seed: int = 42
    noise_var: float = 0.5
    noise_rho: float = 0.9
    gpu_id: int = 0
    bin_config: tuple = (4, 3, 4)
    bridge_epochs: int = 4000
    batch_size: int = 256
    mc_samples: int = 256
    lr_a: float = 0.0005
    lr_b: float = 0.0005
    scale_reg: float = 0.001
    w_a: float = 2.0
    w_b: float = 1.0
    grad_clip_norm: float = 5.0
    impute_k: int = 1024
    impute_batch: int = 2000
    cond_mean_epochs: int = 300
    cond_mean_lr: float = 0.001
    cond_mean_batch: int = 256
    cond_mean_patience: int = 30
    down_epochs: int = 500
    down_lr: float = 0.001
    down_batch: int = 64
    down_patience: int = 30
    w2_n_sub: int = 500
    sb_sweep_random_offset: int = 60000
    go_k_graph: int = 5
    go_solver_verbose: bool = False
    go_lp_time_limit: float = 300.0
    verbose: bool = False
    dim_x: int = 2
    dim_y: int = 1
    dim_z: int = 1
    eps: float = 1e-08


def make_synthetic_runner(config: SyntheticConfig, make_data, make_test_c):
    try:
        import ot
    except ImportError as e:
        raise ImportError('SYN experiments require POT: pip install POT') from e
    NAME = config.name
    SEED = config.seed
    NOISE_VAR = config.noise_var
    NOISE_RHO = config.noise_rho
    MY_GPU_ID = config.gpu_id
    SB_LAMBDA = config.sb_lambda
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DownstreamMLP = SynDownstreamMLP
    generate_weak_test_C = make_test_c
    BIN_CONFIG = config.bin_config
    BRIDGE_EPOCHS = config.bridge_epochs
    BATCH_SIZE = config.batch_size
    MC_SAMPLES = config.mc_samples
    LR_A = config.lr_a
    LR_B = config.lr_b
    SCALE_REG = config.scale_reg
    wA = config.w_a
    wB = config.w_b
    GRAD_CLIP_NORM = config.grad_clip_norm
    IMPUTE_K = config.impute_k
    IMPUTE_BATCH = config.impute_batch
    COND_MEAN_EPOCHS = config.cond_mean_epochs
    COND_MEAN_LR = config.cond_mean_lr
    COND_MEAN_BATCH = config.cond_mean_batch
    COND_MEAN_PATIENCE = config.cond_mean_patience
    DOWN_EPOCHS = config.down_epochs
    DOWN_LR = config.down_lr
    DOWN_BATCH = config.down_batch
    DOWN_PATIENCE = config.down_patience
    W2_N_SUB = config.w2_n_sub
    SB_SWEEP_RANDOM_OFFSET = config.sb_sweep_random_offset
    GO_K_GRAPH = config.go_k_graph
    GO_SOLVER_VERBOSE = config.go_solver_verbose
    GO_LP_TIME_LIMIT = config.go_lp_time_limit
    VERBOSE = config.verbose
    DIM_X = config.dim_x
    DIM_Y = config.dim_y
    DIM_Z = config.dim_z
    EPS = config.eps

    def full_joint_w2_sq_empirical(D_hat, D_true, n_sub=W2_N_SUB, seed=0):
        D_hat = np.asarray(D_hat, dtype=np.float64)
        D_true = np.asarray(D_true, dtype=np.float64)
        rng = np.random.RandomState(seed)
        if n_sub is not None:
            if D_hat.shape[0] > n_sub:
                idx_h = rng.choice(D_hat.shape[0], n_sub, replace=False)
                D_hat = D_hat[idx_h]
            if D_true.shape[0] > n_sub:
                idx_t = rng.choice(D_true.shape[0], n_sub, replace=False)
                D_true = D_true[idx_t]
        mu_t = D_true.mean(axis=0)
        sd_t = D_true.std(axis=0) + 1e-12
        Dh = (D_hat - mu_t) / sd_t
        Dt = (D_true - mu_t) / sd_t
        a = np.ones(Dh.shape[0], dtype=np.float64) / Dh.shape[0]
        b = np.ones(Dt.shape[0], dtype=np.float64) / Dt.shape[0]
        M = ot.dist(Dh, Dt, metric='sqeuclidean')
        w2_sq = ot.emd2(a, b, M, numItermax=1000000)
        return float(w2_sq)

    @torch.no_grad()
    def ensure_valid_posterior_probs(w, logw, stage, lam, batch_start):
        valid = torch.isfinite(w).all() & (w >= 0).all() & (w.sum(dim=1) > 0).all()
        if bool(valid.item()):
            return
        n_nan = int(torch.isnan(logw).sum().item())
        n_inf = int(torch.isinf(logw).sum().item())
        n_bad_w = int((~torch.isfinite(w)).sum().item())
        raise FloatingPointError(f'{NAME} invalid posterior before multinomial: stage={stage}, lambda={float(lam):.12g}, batch_start={batch_start}, NaN logits={n_nan}, Inf logits={n_inf}, non-finite probabilities={n_bad_w}. Possible SB instability; no probabilities were repaired or replaced.')

    def ensure_finite_numpy(arr, label):
        if not np.isfinite(np.asarray(arr)).all():
            raise FloatingPointError(f'{NAME} non-finite values in {label}.')

    def train_cond_mean_mlp(X_np, W_np, seed, epochs=COND_MEAN_EPOCHS, batch_size=COND_MEAN_BATCH, lr=COND_MEAN_LR, patience=COND_MEAN_PATIENCE):
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
        model = CondMeanMLP(input_dim=DIM_X, hidden=64).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        best_state = None
        best_val = float('inf')
        no_imp = 0
        for _ in range(epochs):
            model.train()
            for xb, wb in loader:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), wb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = crit(model(X_val_t), W_val_t).item()
            if val_loss < best_val - 1e-08:
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

    @torch.no_grad()
    def predict_cond_mean(model, X_np):
        model.eval()
        X_t = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).float().to(DEVICE)
        return model(X_t).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_log_r_ratio(x, y_std, z_std, params):
        mY = params['mY_net']
        mZ = params['mZ_net']
        dtype = x.dtype
        device = x.device
        var_y = torch.tensor(params['var_y_mle'], device=device, dtype=dtype).clamp(min=1e-12)
        var_z = torch.tensor(params['var_z_mle'], device=device, dtype=dtype).clamp(min=1e-12)
        mu_y = mY(x)
        mu_z = mZ(x)
        logp_y = -0.5 * (torch.log(2.0 * math.pi * var_y) + (y_std - mu_y).pow(2) / var_y)
        logp_z = -0.5 * (torch.log(2.0 * math.pi * var_z) + (z_std - mu_z).pow(2) / var_z)
        return (logp_y - logp_z).squeeze(-1)

    def cost_fn_res_torch(y_res, z_res, sign_res, cost_std):
        diff = y_res - sign_res * z_res
        return (diff ** 2 / cost_std).squeeze(-1)

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
        return model(X_t).cpu().numpy().astype(np.float32)

    def prepare_context(seed, noise_var, noise_rho):
        XA, YA, ZA_true, XB, YB_true, ZB, XT, YT, ZT = make_data(seed=seed, noise_var=noise_var, noise_rho=noise_rho)
        XC, YC, ZC = generate_weak_test_C(noise_var, noise_rho)
        scY = StandardScaler().fit(YA)
        scZ = StandardScaler().fit(ZB)
        YA_std = scY.transform(YA).astype(np.float32)
        ZB_std = scZ.transform(ZB).astype(np.float32)
        mY_net = train_cond_mean_mlp(X_np=XA, W_np=YA_std, seed=seed + 7000)
        mZ_net = train_cond_mean_mlp(X_np=XB, W_np=ZB_std, seed=seed + 8000)
        muY_A = predict_cond_mean(mY_net, XA)
        muZ_B = predict_cond_mean(mZ_net, XB)
        resY_A = (YA_std - muY_A).astype(np.float32)
        resZ_B = (ZB_std - muZ_B).astype(np.float32)
        muZ_A = predict_cond_mean(mZ_net, XA)
        muY_B = predict_cond_mean(mY_net, XB)
        var_y_mle = float(np.mean(resY_A ** 2))
        var_z_mle = float(np.mean(resZ_B ** 2))
        var_y_mle = max(var_y_mle, 1e-12)
        var_z_mle = max(var_z_mle, 1e-12)
        sign_res = 1.0 if noise_rho >= 0 else -1.0
        rng = np.random.default_rng(seed)
        iy = rng.integers(0, resY_A.shape[0], size=20000)
        iz = rng.integers(0, resZ_B.shape[0], size=20000)
        diffs = resY_A[iy] - sign_res * resZ_B[iz]
        cost_std = float(np.var(diffs)) + 1e-06
        train_torch = {'XA': torch.from_numpy(XA).float().to(DEVICE), 'YA_std': torch.from_numpy(YA_std).float().to(DEVICE), 'YA_res': torch.from_numpy(resY_A).float().to(DEVICE), 'XB': torch.from_numpy(XB).float().to(DEVICE), 'ZB_std': torch.from_numpy(ZB_std).float().to(DEVICE), 'ZB_res': torch.from_numpy(resZ_B).float().to(DEVICE)}
        pool_torch = {'ya_pool_std': torch.from_numpy(YA_std).float().to(DEVICE), 'ya_pool_res': torch.from_numpy(resY_A).float().to(DEVICE), 'zb_pool_std': torch.from_numpy(ZB_std).float().to(DEVICE), 'zb_pool_res': torch.from_numpy(resZ_B).float().to(DEVICE)}
        params = {'sign_res': float(sign_res), 'cost_std': float(cost_std), 'mY_net': mY_net.to(DEVICE), 'mZ_net': mZ_net.to(DEVICE), 'var_y_mle': float(var_y_mle), 'var_z_mle': float(var_z_mle)}
        aux = {'XA': XA, 'YA': YA, 'ZA_true': ZA_true, 'XB': XB, 'YB_true': YB_true, 'ZB': ZB, 'XT': XT, 'YT': YT, 'ZT': ZT, 'YA_std': YA_std, 'ZB_std': ZB_std, 'muZ_A': muZ_A.astype(np.float32), 'muY_B': muY_B.astype(np.float32), 'scY': scY, 'scZ': scZ}
        aux.update({'XC': XC, 'YC': YC, 'ZC': ZC})
        return {'seed': seed, 'noise_var': float(noise_var), 'noise_rho': float(noise_rho), 'train_torch': train_torch, 'pool_torch': pool_torch, 'params': params, 'aux': aux}

    def train_sb(lambda_val, ctx):
        td = ctx['train_torch']
        pool = ctx['pool_torch']
        params = ctx['params']
        XA_t = td['XA']
        YA_std_t = td['YA_std']
        YA_res_t = td['YA_res']
        XB_t = td['XB']
        ZB_std_t = td['ZB_std']
        ZB_res_t = td['ZB_res']
        ya_pool_std = pool['ya_pool_std']
        ya_pool_res = pool['ya_pool_res']
        zb_pool_std = pool['zb_pool_std']
        zb_pool_res = pool['zb_pool_res']
        sign_res = float(params['sign_res'])
        cost_std = float(params['cost_std'])
        lam_f = float(lambda_val)
        a_net = PositiveMLP(DIM_X + DIM_Y, hidden=64).to(DEVICE)
        b_net = PositiveMLP(DIM_X + DIM_Z, hidden=64).to(DEVICE)
        opt_a = torch.optim.Adam(a_net.parameters(), lr=LR_A)
        opt_b = torch.optim.Adam(b_net.parameters(), lr=LR_B)
        N_A = XA_t.size(0)
        N_B = XB_t.size(0)
        for ep in range(BRIDGE_EPOCHS):
            idx_a = torch.randint(0, N_A, (BATCH_SIZE,), device=DEVICE)
            idx_b = torch.randint(0, N_B, (BATCH_SIZE,), device=DEVICE)
            x_a = XA_t[idx_a]
            y_a = YA_std_t[idx_a]
            y_res_a = YA_res_t[idx_a]
            x_b = XB_t[idx_b]
            z_b = ZB_std_t[idx_b]
            z_res_b = ZB_res_t[idx_b]
            a_val = a_net(torch.cat([x_a, y_a], dim=-1)).squeeze(-1)
            idx_z = torch.randint(0, zb_pool_std.size(0), (BATCH_SIZE, MC_SAMPLES), device=DEVICE)
            z_mc_std = zb_pool_std[idx_z.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Z)
            z_mc_res = zb_pool_res[idx_z.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Z)
            x_rep = x_a.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_X)
            y_res_rep = y_res_a.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_Y)
            z_std_rep = z_mc_std.reshape(-1, DIM_Z)
            z_res_rep = z_mc_res.reshape(-1, DIM_Z)
            with torch.no_grad():
                b_inner = b_net(torch.cat([x_rep, z_std_rep], dim=-1)).squeeze(-1)
            c1 = cost_fn_res_torch(y_res_rep, z_res_rep, sign_res, cost_std)
            logw1 = torch.log(b_inner + EPS) - c1 / lam_f
            lme1 = torch.logsumexp(logw1.view(BATCH_SIZE, MC_SAMPLES), dim=1) - math.log(MC_SAMPLES)
            lossA = torch.mean((a_val * torch.exp(lme1) - 1.0) ** 2)
            if SCALE_REG > 0:
                lossA = lossA + SCALE_REG * (a_val.mean() - 1.0).pow(2)
            if not bool(torch.isfinite(lossA.detach()).item()):
                raise FloatingPointError(f'{NAME} SB lossA is NaN/Inf at epoch={ep + 1}/{BRIDGE_EPOCHS}, lambda={float(lambda_val):.12g}.')
            opt_a.zero_grad(set_to_none=True)
            (wA * lossA).backward()
            torch.nn.utils.clip_grad_norm_(a_net.parameters(), GRAD_CLIP_NORM)
            opt_a.step()
            b_val = b_net(torch.cat([x_b, z_b], dim=-1)).squeeze(-1)
            idx_y = torch.randint(0, ya_pool_std.size(0), (BATCH_SIZE, MC_SAMPLES), device=DEVICE)
            y_mc_std = ya_pool_std[idx_y.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Y)
            y_mc_res = ya_pool_res[idx_y.reshape(-1)].view(BATCH_SIZE, MC_SAMPLES, DIM_Y)
            x2_rep = x_b.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_X)
            z2_std_rep = z_b.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_Z)
            z2_res_rep = z_res_b.unsqueeze(1).expand(-1, MC_SAMPLES, -1).reshape(-1, DIM_Z)
            y2_std = y_mc_std.reshape(-1, DIM_Y)
            y2_res = y_mc_res.reshape(-1, DIM_Y)
            with torch.no_grad():
                a_inner = a_net(torch.cat([x2_rep, y2_std], dim=-1)).squeeze(-1)
            log_r = get_log_r_ratio(x2_rep, y2_std, z2_std_rep, params)
            c2 = cost_fn_res_torch(y2_res, z2_res_rep, sign_res, cost_std)
            logw2 = torch.log(a_inner + EPS) + log_r - c2 / lam_f
            lme2 = torch.logsumexp(logw2.view(BATCH_SIZE, MC_SAMPLES), dim=1) - math.log(MC_SAMPLES)
            lossB = torch.mean((b_val * torch.exp(lme2) - 1.0) ** 2)
            if SCALE_REG > 0:
                lossB = lossB + SCALE_REG * (b_val.mean() - 1.0).pow(2)
            if not bool(torch.isfinite(lossB.detach()).item()):
                raise FloatingPointError(f'{NAME} SB lossB is NaN/Inf at epoch={ep + 1}/{BRIDGE_EPOCHS}, lambda={float(lambda_val):.12g}.')
            opt_b.zero_grad(set_to_none=True)
            (wB * lossB).backward()
            torch.nn.utils.clip_grad_norm_(b_net.parameters(), GRAD_CLIP_NORM)
            opt_b.step()
            if VERBOSE and (ep + 1) % 1000 == 0:
                print(f'      [SB] epoch {ep + 1}/{BRIDGE_EPOCHS} lossA={lossA.item():.3e}, lossB={lossB.item():.3e}')
        a_net.eval()
        b_net.eval()
        for net_name, net in (('a_net', a_net), ('b_net', b_net)):
            bad = [name for name, parameter in net.named_parameters() if not bool(torch.isfinite(parameter).all().item())]
            if bad:
                raise FloatingPointError(f'{NAME} SB diverged at lambda={float(lambda_val):.12g}: {net_name} has non-finite parameters: {bad}.')
        return (a_net, b_net)

    @torch.no_grad()
    def sb_precompute_A_zres_weights(ctx, b_net, lambda_val):
        td = ctx['train_torch']
        pool = ctx['pool_torch']
        aux = ctx['aux']
        params = ctx['params']
        XA_t = td['XA']
        YA_res_t = td['YA_res']
        zb_pool_std = pool['zb_pool_std']
        zb_pool_res = pool['zb_pool_res']
        z_mean_std = torch.from_numpy(aux['muZ_A']).float().to(DEVICE)
        N = XA_t.size(0)
        M = zb_pool_std.size(0)
        K = min(IMPUTE_K, M)
        lam_f = float(lambda_val)
        sign_res = float(params['sign_res'])
        cost_std = float(params['cost_std'])
        z_res_cand = np.empty((N, K), dtype=np.float32)
        weights = np.empty((N, K), dtype=np.float32)
        z_hat_raw = np.empty((N, 1), dtype=np.float32)
        scZ = aux['scZ']
        b_net.eval()
        for i0 in range(0, N, IMPUTE_BATCH):
            i1 = min(i0 + IMPUTE_BATCH, N)
            x = XA_t[i0:i1]
            y_res = YA_res_t[i0:i1]
            bs = x.size(0)
            idx = torch.randint(0, M, (bs, K), device=DEVICE)
            zk_std = zb_pool_std[idx]
            zk_res = zb_pool_res[idx]
            x_rep = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_X)
            z_std_rep = zk_std.reshape(-1, DIM_Z)
            y_res_rep = y_res.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_Y)
            z_res_rep = zk_res.reshape(-1, DIM_Z)
            b_val = b_net(torch.cat([x_rep, z_std_rep], dim=-1)).squeeze(-1)
            c_val = cost_fn_res_torch(y_res_rep, z_res_rep, sign_res, cost_std)
            logw = (torch.log(b_val + EPS) - c_val / lam_f).view(bs, K)
            w = torch.softmax(logw, dim=1)
            ensure_valid_posterior_probs(w, logw, 'A-side Z imputation', lam_f, i0)
            z_res_cand[i0:i1, :] = zk_res.squeeze(-1).cpu().numpy().astype(np.float32)
            weights[i0:i1, :] = w.cpu().numpy().astype(np.float32)
            sel = torch.multinomial(w, 1, replacement=True)
            z_sel_res = zk_res.squeeze(-1).gather(1, sel).view(bs, 1)
            z_imp_std = z_mean_std[i0:i1] + z_sel_res
            z_imp_raw = scZ.inverse_transform(z_imp_std.cpu().numpy().astype(np.float32)).astype(np.float32)
            z_hat_raw[i0:i1, :] = z_imp_raw
        return (z_res_cand, weights, z_hat_raw)

    def sb_pbin_from_precomputed(z_mean_std_np, z_res_candidates, weights, edges_z_std):
        z_mean_std_np = np.asarray(z_mean_std_np, dtype=np.float32)
        z_res_candidates = np.asarray(z_res_candidates, dtype=np.float32)
        ensure_finite_numpy(z_res_candidates, 'SB residual candidates')
        ensure_finite_numpy(z_mean_std_np, 'SB conditional mean for A-side Z')
        ensure_finite_numpy(weights, 'SB posterior weights')
        weights = np.asarray(weights, dtype=np.float32)
        N, K = z_res_candidates.shape
        nZ = len(edges_z_std) - 1
        boundaries = torch.tensor(edges_z_std[1:-1], dtype=torch.float32)
        p_out = np.zeros((N, nZ), dtype=np.float32)
        for i0 in range(0, N, 2000):
            i1 = min(i0 + 2000, N)
            z_mean = torch.from_numpy(z_mean_std_np[i0:i1, :])
            z_res = torch.from_numpy(z_res_candidates[i0:i1, :])
            w = torch.from_numpy(weights[i0:i1, :])
            z_std = z_mean + z_res
            bin_idx = torch.bucketize(z_std, boundaries, right=True)
            p = torch.zeros((z_res.size(0), nZ), dtype=torch.float32)
            p.scatter_add_(1, bin_idx.long(), w)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p_out[i0:i1, :] = p.numpy()
        z_map = np.argmax(p_out, axis=1).astype(int)
        return z_map

    @torch.no_grad()
    def sb_impute_YB(ctx, a_net, lambda_val):
        td = ctx['train_torch']
        pool = ctx['pool_torch']
        aux = ctx['aux']
        params = ctx['params']
        XB_t = td['XB']
        ZB_std_t = td['ZB_std']
        ZB_res_t = td['ZB_res']
        ya_pool_std = pool['ya_pool_std']
        ya_pool_res = pool['ya_pool_res']
        y_mean_std = torch.from_numpy(aux['muY_B']).float().to(DEVICE)
        N = XB_t.size(0)
        M = ya_pool_std.size(0)
        K = min(IMPUTE_K, M)
        lam_f = float(lambda_val)
        sign_res = float(params['sign_res'])
        cost_std = float(params['cost_std'])
        scY = aux['scY']
        y_hat_raw = np.empty((N, 1), dtype=np.float32)
        a_net.eval()
        for i0 in range(0, N, IMPUTE_BATCH):
            i1 = min(i0 + IMPUTE_BATCH, N)
            x = XB_t[i0:i1]
            z_std = ZB_std_t[i0:i1]
            z_res = ZB_res_t[i0:i1]
            bs = x.size(0)
            idx = torch.randint(0, M, (bs, K), device=DEVICE)
            yk_std = ya_pool_std[idx]
            yk_res = ya_pool_res[idx]
            x_rep = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_X)
            y_std_rep = yk_std.reshape(-1, DIM_Y)
            y_res_rep = yk_res.reshape(-1, DIM_Y)
            z_std_rep = z_std.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_Z)
            z_res_rep = z_res.unsqueeze(1).expand(-1, K, -1).reshape(-1, DIM_Z)
            a_val = a_net(torch.cat([x_rep, y_std_rep], dim=-1)).squeeze(-1)
            c_val = cost_fn_res_torch(y_res_rep, z_res_rep, sign_res, cost_std)
            log_r = get_log_r_ratio(x_rep, y_std_rep, z_std_rep, params)
            logw = (torch.log(a_val + EPS) + log_r - c_val / lam_f).view(bs, K)
            w = torch.softmax(logw, dim=1)
            ensure_valid_posterior_probs(w, logw, 'B-side Y imputation', lam_f, i0)
            sel = torch.multinomial(w, 1, replacement=True)
            y_sel_res = yk_res.squeeze(-1).gather(1, sel).view(bs, 1)
            y_imp_std = y_mean_std[i0:i1] + y_sel_res
            y_imp_raw = scY.inverse_transform(y_imp_std.cpu().numpy().astype(np.float32)).astype(np.float32)
            y_hat_raw[i0:i1, :] = y_imp_raw
        return y_hat_raw

    def eval_both_tests_and_brmse(X_train, Y_train, T_feat, YT, C_feat, YC, X_B_eval, YB_true, input_dim, seed):
        model, scaler = train_downstream_model(X_train=X_train, Y_train=Y_train, input_dim=input_dim, seed=seed)
        pred_T = predict_downstream_model(model, scaler, T_feat)
        pred_C = predict_downstream_model(model, scaler, C_feat)
        pred_B = predict_downstream_model(model, scaler, X_B_eval)
        ensure_finite_numpy(pred_T, 'Strong downstream predictions')
        ensure_finite_numpy(pred_C, 'Weak downstream predictions')
        ensure_finite_numpy(pred_B, 'B downstream predictions')
        scores = {'strong_r2': r2(YT, pred_T), 'weak_r2': r2(YC, pred_C), 'downstream_rmse_B': rmse(YB_true, pred_B)}
        del model, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores

    def empirical_joint_w2(D_hat, D_true, seed):
        ensure_finite_numpy(D_hat, 'completed joint (W2 input)')
        ensure_finite_numpy(D_true, 'true joint (W2 input)')
        result = full_joint_w2_sq_empirical(D_hat, D_true, n_sub=W2_N_SUB, seed=seed)
        ensure_finite_numpy(result, 'empirical W2^2')
        return result

    def run_experiment():
        if not torch.cuda.is_available():
            raise RuntimeError('GPU 0 is not detected. Check CUDA_VISIBLE_DEVICES=0 and the kernel state.')
        start = time.time()
        seed, noise_var, noise_rho, lam = (SEED, NOISE_VAR, NOISE_RHO, SB_LAMBDA)
        ctx = prepare_context(seed=seed, noise_var=noise_var, noise_rho=noise_rho)
        aux = ctx['aux']
        XA, YA, ZA_true = (aux['XA'], aux['YA'], aux['ZA_true'])
        XB, YB_true, ZB = (aux['XB'], aux['YB_true'], aux['ZB'])
        XT, YT, ZT = (aux['XT'], aux['YT'], aux['ZT'])
        XC, YC, ZC = (aux['XC'], aux['YC'], aux['ZC'])
        YA_std, ZB_std = (aux['YA_std'], aux['ZB_std'])
        results = []
        D_true = np.hstack([np.asarray(XT, dtype=np.float64), as_2d(YT).astype(np.float64), as_2d(ZT).astype(np.float64)])
        n_bins_x, n_bins_y, n_bins_z = BIN_CONFIG
        go = GORJointPaper(n_bins_x=n_bins_x, n_bins_y=n_bins_y, n_bins_z=n_bins_z, standardize_x=True, k_graph=GO_K_GRAPH, solver_verbose=GO_SOLVER_VERBOSE, random_state=seed, cost_sample_cap=512, lp_time_limit=GO_LP_TIME_LIMIT)
        go.fit(XA, YA, XB, ZB)
        edges_z = go.get_binning()['Z']
        nZ_eff = len(edges_z) - 1
        ZA_bin_true = digitize_like_go(ZA_true, edges_z)
        ZB_oh = one_hot(digitize_like_go(ZB, edges_z), nZ_eff)
        ZT_oh = one_hot(digitize_like_go(ZT, edges_z), nZ_eff)
        ZC_oh = one_hot(digitize_like_go(ZC, edges_z), nZ_eff)
        T_feat = np.hstack([XT, ZT_oh]).astype(np.float32)
        C_feat = np.hstack([XC, ZC_oh]).astype(np.float32)
        B_feat_true = np.hstack([XB, ZB_oh]).astype(np.float32)

        def record(method, scores, rmse_b, w2_sq=np.nan):
            for score_name in ('strong_r2', 'weak_r2'):
                ensure_finite_numpy(scores[score_name], f'{method} {score_name}')
            ensure_finite_numpy(rmse_b, f'{method} RMSE_B')
            if method not in ('NAIVE', 'ORACLE'):
                ensure_finite_numpy(w2_sq, f'{method} W2^2')
            results.append({'method': method, 'strong_r2': float(scores['strong_r2']), 'weak_r2': float(scores['weak_r2']), 'rmse_B': float(rmse_b), 'w2_sq': float(w2_sq)})
        naive = eval_both_tests_and_brmse(X_train=XA, Y_train=YA, T_feat=XT, YT=YT, C_feat=XC, YC=YC, X_B_eval=XB, YB_true=YB_true, input_dim=DIM_X, seed=seed + 1000)
        record('NAIVE', naive, naive['downstream_rmse_B'])
        X_oracle_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_true, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
        Y_oracle_train = np.vstack([YA, YB_true]).astype(np.float32)
        oracle = eval_both_tests_and_brmse(X_train=X_oracle_train, Y_train=Y_oracle_train, T_feat=T_feat, YT=YT, C_feat=C_feat, YC=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=DIM_X + nZ_eff, seed=seed + 2000 + nZ_eff)
        record('ORACLE', oracle, oracle['downstream_rmse_B'])
        ZA_bin_go = np.argmax(go.predict_proba(XA, YA), axis=1).astype(int)
        YB_hat_go = go_impute_YB_continuous_from_gamma(go, XB, ZB, YA)
        ZA_hat_go_cont = go_impute_ZA_continuous_from_gamma(go, XA, YA, ZB)
        X_go_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_go, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
        Y_go_train = np.vstack([YA, YB_hat_go]).astype(np.float32)
        go_scores = eval_both_tests_and_brmse(X_train=X_go_train, Y_train=Y_go_train, T_feat=T_feat, YT=YT, C_feat=C_feat, YC=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=DIM_X + nZ_eff, seed=seed + 3000 + nZ_eff)
        D_go = np.vstack([np.hstack([XA, YA, ZA_hat_go_cont]), np.hstack([XB, YB_hat_go, ZB])]).astype(np.float64)
        record('GO', go_scores, rmse(YB_true, YB_hat_go), empirical_joint_w2(D_go, D_true, seed))
        del D_go
        mm = MixedMethodsImputerFast(random_state=seed)
        mm.fit(XA, YA_std, XB, ZB_std)
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
            ZA_hat_mm = aux['scZ'].inverse_transform(as_2d(ZA_hat_std)).astype(np.float32)
            YB_hat_mm = aux['scY'].inverse_transform(as_2d(YB_hat_std)).astype(np.float32)
            ZA_bin_mm = digitize_like_go(ZA_hat_mm, edges_z)
            X_mm_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_mm, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
            Y_mm_train = np.vstack([YA, YB_hat_mm]).astype(np.float32)
            mm_scores = eval_both_tests_and_brmse(X_train=X_mm_train, Y_train=Y_mm_train, T_feat=T_feat, YT=YT, C_feat=C_feat, YC=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=DIM_X + nZ_eff, seed=seed + 4000 + 10 * k + nZ_eff)
            D_mm = np.vstack([np.hstack([XA, YA, ZA_hat_mm]), np.hstack([XB, YB_hat_mm, ZB])]).astype(np.float64)
            record(f'MM{k}', mm_scores, rmse(YB_true, YB_hat_mm), empirical_joint_w2(D_mm, D_true, seed))
            del D_mm
        edges_z_std = (edges_z - float(aux['scZ'].mean_[0])) / (float(aux['scZ'].scale_[0]) + 1e-12)
        set_seed(seed + SB_SWEEP_RANDOM_OFFSET)
        a_net, b_net = train_sb(lambda_val=lam, ctx=ctx)
        z_res_cand, weights, ZA_hat_sb = sb_precompute_A_zres_weights(ctx=ctx, b_net=b_net, lambda_val=lam)
        YB_hat_sb = sb_impute_YB(ctx=ctx, a_net=a_net, lambda_val=lam)
        ZA_bin_sb = sb_pbin_from_precomputed(z_mean_std_np=aux['muZ_A'].astype(np.float32), z_res_candidates=z_res_cand, weights=weights, edges_z_std=edges_z_std)
        X_sb_train = np.vstack([np.hstack([XA, one_hot(ZA_bin_sb, nZ_eff)]), np.hstack([XB, ZB_oh])]).astype(np.float32)
        Y_sb_train = np.vstack([YA, YB_hat_sb]).astype(np.float32)
        sb_scores = eval_both_tests_and_brmse(X_train=X_sb_train, Y_train=Y_sb_train, T_feat=T_feat, YT=YT, C_feat=C_feat, YC=YC, X_B_eval=B_feat_true, YB_true=YB_true, input_dim=DIM_X + nZ_eff, seed=seed + 5000 + 100 + nZ_eff)
        D_sb = np.vstack([np.hstack([XA, YA, ZA_hat_sb]), np.hstack([XB, YB_hat_sb, ZB])]).astype(np.float64)
        record('SB', sb_scores, rmse(YB_true, YB_hat_sb), empirical_joint_w2(D_sb, D_true, seed))
        width = 94
        print('\n' + '=' * width)
        print(f'{NAME} | GPU={MY_GPU_ID} | seed={seed} | noise_var={noise_var} | noise_rho={noise_rho}')
        print(f'BIN_CONFIG={BIN_CONFIG} | fixed SB lambda={lam:.15g}')
        print(f'A={len(XA)} | B={len(XB)} | Strong T={len(YT)} | Weak C={len(YC)}')
        print(f'Strong/Weak R2: T/C tests; RMSE_B: B={len(XB)}; W2^2: completed A+B vs T')
        print('-' * width)
        print(f"{'Method':<11} {'Strong R2':>14} {'Weak R2':>14} {'RMSE_B':>14} {'W2^2 (4D)':>18}")
        print('-' * width)
        for row in results:
            w2 = '-' if np.isnan(row['w2_sq']) else f"{row['w2_sq']:.6f}"
            print(f"{row['method']:<11} {row['strong_r2']:>14.6f} {row['weak_r2']:>14.6f} {row['rmse_B']:>14.6f} {w2:>18}")
        print('-' * width)
        print(f'Elapsed time: {time.time() - start:.1f} seconds')
        print('=' * width)
        del a_net, b_net, z_res_cand, weights, go, ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    return run_experiment
