import os
MY_GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(MY_GPU_ID)

import numpy as np
from utils import SyntheticConfig, make_synthetic_runner, set_seed

A_TOTAL = 10000
B_TOTAL = 10000
T_TOTAL = 10000
C_TOTAL = 50
N_TOTAL = A_TOTAL + B_TOTAL + T_TOTAL
DIM_X = 2
SEED = 42
NOISE_VAR = 0.5
NOISE_RHO = 0.9
SB_LAMBDA = 0.1963040650040271


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def make_syn3_data(seed, noise_var, noise_rho):
    set_seed(seed)
    cov = float(noise_rho) * float(noise_var)
    Sigma_eps = np.array([[noise_var, cov], [cov, noise_var]], dtype=np.float32)
    X_all = np.random.randn(N_TOTAL, DIM_X).astype(np.float32)
    eps = np.random.multivariate_normal(mean=[0.0, 0.0], cov=Sigma_eps, size=N_TOTAL).astype(np.float32)
    X1 = X_all[:, [0]]
    X2 = X_all[:, [1]]
    eps1 = eps[:, [0]]
    eps2 = eps[:, [1]]
    Y_all = (softplus_np(X1) + X2 + eps1).astype(np.float32)
    Z_all = (0.5 * softplus_np(X1 + X2) + X2 ** 2 + eps2).astype(np.float32)
    XA = X_all[:A_TOTAL]
    YA = Y_all[:A_TOTAL]
    ZA_true = Z_all[:A_TOTAL]
    b0 = A_TOTAL
    b1 = A_TOTAL + B_TOTAL
    XB = X_all[b0:b1]
    YB_true = Y_all[b0:b1]
    ZB = Z_all[b0:b1]
    XT = X_all[b1:]
    YT = Y_all[b1:]
    ZT = Z_all[b1:]
    return (XA, YA, ZA_true, XB, YB_true, ZB, XT, YT, ZT)


def generate_weak_test_C(noise_var, noise_rho):
    cov = float(noise_rho) * float(noise_var)
    Sigma_eps = np.array([[noise_var, cov], [cov, noise_var]], dtype=np.float32)
    XC = np.random.randn(C_TOTAL, DIM_X).astype(np.float32)
    epsC = np.random.multivariate_normal(mean=[0.0, 0.0], cov=Sigma_eps, size=C_TOTAL).astype(np.float32)
    YC = (softplus_np(XC[:, [0]]) + XC[:, [1]] + epsC[:, [0]]).astype(np.float32)
    ZC = (0.5 * softplus_np(XC[:, [0]] + XC[:, [1]]) + XC[:, [1]] ** 2 + epsC[:, [1]]).astype(np.float32)
    return (XC, YC, ZC)


def run_experiment():
    config = SyntheticConfig(
        name='SYN3', seed=SEED, noise_var=NOISE_VAR, noise_rho=NOISE_RHO,
        sb_lambda=SB_LAMBDA, gpu_id=MY_GPU_ID,
    )
    return make_synthetic_runner(config, make_syn3_data, generate_weak_test_C)()


if __name__ == '__main__':
    try:
        results = run_experiment()
    except FloatingPointError as exc:
        print(f'[SYN3 NUMERICAL FAILURE] {type(exc).__name__}: {exc}')
        print('No metrics were saved; the unsuccessful run must be investigated.')
        raise
