import numpy as np
import warnings
warnings.filterwarnings('ignore')

from constants import U_ALPHA, DRIFT_SCALE, DIV_TOL
np.random.seed(27)
BASELINE_TIME_STEPS = 1000
MONTE_CARLO_PATHS = 200
FINE_SCALE_MULTIPLIER = 2

# ===================== 论文：安全版势能与漂移项 =====================
def potential_U(x, alpha=1.0):
    x_safe = np.clip(np.abs(x), 1e-8, 1e8)
    return 0.5 * alpha * np.square(x_safe)

def grad_U(x, alpha=1.0):
    return alpha * x  # 移除冗余clip，已在主函数保护

# 【关键优化】删除无意义的np.gradient，论文漂移直接计算，速度暴增
def drift_C(x, alpha=1.0, scale=0.1):
    x_safe = np.clip(x, 1e-8, 1e8)
    return scale * alpha * x_safe

# ===================== 向量化核心：Milstein/Euler全向量化 =====================
def milstein_step_vec(S_prev, mu, sigma, dt, dW):
    return S_prev * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + 0.5 * sigma**2 * (dW**2 - dt))

def euler_maruyama_step_vec(S_prev, mu, sigma, dt, dW):
    return S_prev + mu * S_prev * dt + sigma * S_prev * dW

# ===================== 向量化扩散：核心提速 =====================
def standard_diffusion_vec(mu, sigma, S0, t, M, alpha=U_ALPHA, drift_scale=DRIFT_SCALE):
    dt = np.diff(t)
    Nt = len(t)
    dW = np.random.randn(M, Nt-1) * np.sqrt(dt[np.newaxis, :])
    S = np.full((M, Nt), S0, dtype=np.float64)
    
    for i in range(Nt-1):
        # 【优化】极简漂移修正，删除冗余计算
        g_u = grad_U(S[:, i], alpha)
        c = drift_C(S[:, i], alpha, drift_scale)
        mu_corrected = mu - 1e-4 * g_u + 1e-4 * c
        S[:, i+1] = milstein_step_vec(S[:, i], mu_corrected, sigma, dt[i], dW[:, i])
        S[:, i+1] = np.clip(S[:, i+1], 1e-8, 1e10)
    
    return np.mean(S, axis=0)

def normal_diffusion_vec(mu, sigma, S0, t, M):
    dt = np.diff(t)
    Nt = len(t)
    dW = np.random.randn(M, Nt-1) * np.sqrt(dt[np.newaxis, :])
    S = np.full((M, Nt), S0, dtype=np.float64)
    
    for i in range(Nt-1):
        S[:, i+1] = milstein_step_vec(S[:, i], mu, sigma, dt[i], dW[:, i])
        S[:, i+1] = np.clip(S[:, i+1], 1e-8, 1e10)
    
    return np.mean(S, axis=0)

def euler_maruyama_diffusion_vec(mu, sigma, S0, t, M, alpha=U_ALPHA, drift_scale=DRIFT_SCALE):
    dt = np.diff(t)
    Nt = len(t)
    dW = np.random.randn(M, Nt-1) * np.sqrt(dt[np.newaxis, :])
    S = np.full((M, Nt), S0, dtype=np.float64)
    
    for i in range(Nt-1):
        g_u = grad_U(S[:, i], alpha)
        c = drift_C(S[:, i], alpha, drift_scale)
        mu_corrected = mu - 1e-4 * g_u + 1e-4 * c
        S[:, i+1] = euler_maruyama_step_vec(S[:, i], mu_corrected, sigma, dt[i], dW[:, i])
        S[:, i+1] = np.clip(S[:, i+1], 1e-8, 1e10)
    
    return np.mean(S, axis=0)

def control_variate_diffusion_vec(mu, sigma, S0, t, M, alpha=U_ALPHA, drift_scale=DRIFT_SCALE):
    S_mean = standard_diffusion_vec(mu, sigma, S0, t, M, alpha, drift_scale)
    S_exact = exact_geometric_brownian(mu, sigma, S0, t)
    f_cent = S_mean - np.mean(S_mean)
    c_cent = S_exact - np.mean(S_exact)
    theta = -np.cov(f_cent, c_cent)[0,1] / (np.var(c_cent) + 1e-8)
    return S_mean + theta * (S_exact - np.mean(S_exact))

# ===================== 原有函数简化 =====================
def fast_multiscale_brownian(T, n_coarse, multiplier=2):
    dt_c = T / n_coarse
    t_c = np.linspace(0, T, n_coarse+1)
    dW_c = np.random.randn(n_coarse) * np.sqrt(dt_c)
    W_c = np.concatenate([[0], np.cumsum(dW_c)])
    n_fine = n_coarse * multiplier
    t_f = np.linspace(0, T, n_fine+1)
    W_f = np.interp(t_f, t_c, W_c)
    corr = np.random.randn(n_fine) * np.sqrt(dt_c / multiplier)
    W_f[1:] += np.cumsum(corr)
    return W_f, t_f, W_c, t_c

def multiscale_time_generator(T, N_coarse, N_fine_per_coarse=2):
    N_fine_per_coarse = int(N_fine_per_coarse)
    t_coarse = np.linspace(0, T, N_coarse+1)
    N_fine = N_coarse * N_fine_per_coarse
    t_fine = np.linspace(0, T, N_fine+1)
    return t_fine, t_coarse

def exact_geometric_brownian(mu, sigma, S0, t):
    return S0 * np.exp((mu - 0.5*sigma**2)*t)

# ===================== 统计函数 =====================
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0.0, 0.0
    m = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    return float(m - 1.96*se), float(m + 1.96*se)

def calculate_max_drawdown(prices):
    if len(prices) == 0:
        return 0.0
    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / peak
    return float(dd.max())

def calculate_trading_metrics(df):
    ret = df['portfolio_value'].pct_change().dropna()
    if len(ret) == 0:
        return {'sharpe_ratio':0,'max_drawdown':0,'win_rate':0,'profit_factor':0}
    rf = (1+0.02)**(1/252)-1
    ex = ret - rf
    sharpe = np.sqrt(252) * ex.mean() / (ex.std()+1e-8)
    dd = calculate_max_drawdown(df['portfolio_value'].values)
    win = len(ret[ret>0])/len(ret) if len(ret)>0 else 0
    profit = ret[ret>0].sum()
    loss = -ret[ret<0].sum()
    pf = profit/(loss+1e-8)
    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': dd,
        'win_rate': win,
        'profit_factor': pf
    }

def calculate_statistical_significance(a,b,alpha=0.05):
    from scipy import stats
    t,p = stats.ttest_ind(a,b, equal_var=False)
    return {'t':t, 'p':p, 'significant': p<alpha}

standard_diffusion = standard_diffusion_vec
normal_diffusion = normal_diffusion_vec
euler_maruyama_diffusion = euler_maruyama_diffusion_vec
control_variate_diffusion = control_variate_diffusion_vec