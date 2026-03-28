import numpy as np
from core_models import standard_diffusion
from constants import U_ALPHA, DRIFT_SCALE

CONFIG = {
    'T': 1.0,
    'COARSE': 40,
    'MULTIPLIER': 2,
    'M': 200,
    'transaction_cost': 0.0003,
    'alpha': U_ALPHA,
    'drift_scale': DRIFT_SCALE
}

def vectorized_sim(mu, sigma, S0, t, M, alpha=U_ALPHA, drift_scale=DRIFT_SCALE):
    return standard_diffusion(mu, sigma, S0, t, M, alpha, drift_scale)

def accelerated_diffusion(mu, sigma, S0, T, M, n_coarse=40, fine_scale_multiplier=2):
    t_f = np.linspace(0, T, n_coarse * fine_scale_multiplier + 1)
    S_mean = vectorized_sim(mu, sigma, S0, t_f, M)
    return S_mean, t_f

def ablation_diffusion(mu, sigma, S0, T, M, n_coarse=40, fine_scale_multiplier=2, method='milstein'):
    t_f = np.linspace(0, T, n_coarse * fine_scale_multiplier + 1)
    if method == 'paper_accel':
        S_mean = vectorized_sim(mu, sigma, S0, t_f, M)
    else:
        S_mean = standard_diffusion(mu, sigma, S0, t_f, M, alpha=0, drift_scale=0)
    return S_mean, t_f