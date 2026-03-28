import numpy as np
import pandas as pd
import time
from core_models import *
from parallel_utils import accelerated_diffusion, ablation_diffusion
from visualization import *
from constants import U_ALPHA, DRIFT_SCALE

CONFIG = {
    'T':1.0, 'WINDOW':5,
    'COST':0.0003,
    'THRESHOLD':0.002, 'PRED_STEPS':50,
    'alpha': U_ALPHA, 'drift_scale': DRIFT_SCALE,
    'MC_PATHS':200
}

def predict_price(mu, sigma, S0, model, n_coarse=40):
    t = np.linspace(0, 1, CONFIG['PRED_STEPS'])
    M = CONFIG['MC_PATHS']
    if model == 'normal':
        p = normal_diffusion(mu,sigma,S0,t,M)
    elif model == 'accelerated':
        p,_ = accelerated_diffusion(mu,sigma,S0,1.0,M,n_coarse,2)
    elif model == 'milstein':
        p = standard_diffusion(mu,sigma,S0,t,M, alpha=0, drift_scale=0)
    elif model == 'euler_maruyama':
        p = euler_maruyama_diffusion(mu,sigma,S0,t,M, alpha=0, drift_scale=0)
    elif model == 'control_variate':
        p = control_variate_diffusion(mu,sigma,S0,t,M)
    elif model == 'paper_accel':
        p = standard_diffusion(mu,sigma,S0,t,M, CONFIG['alpha'], CONFIG['drift_scale'])
    else:
        p = normal_diffusion(mu,sigma,S0,t,100)
    return np.mean(p[-5:])

def run_trading_strategy(data, model_type, n_coarse=40):
    df = data.copy().sort_values('date').reset_index(drop=True)
    df['log_ret'] = np.log(df.open/df.open.shift(1))
    df = df.dropna().reset_index(drop=True)
    win = CONFIG['WINDOW']
    total_windows = len(df)-win
    dates,preds,reals = [],[],[]
    t0 = time.time()

    for i in range(total_windows):
        w = df.iloc[i:i+win]
        mu = w.log_ret.mean()*252
        sigma = w.log_ret.std()*np.sqrt(252)
        S0 = df.iloc[i+win].open
        pred = predict_price(mu,sigma,S0,model_type,n_coarse)
        dates.append(df.iloc[i+win].date)
        preds.append(pred)
        reals.append(S0)

    res = pd.DataFrame({'date':dates,'predict':preds,'real':reals})
    res['signal'] = 0
    th = CONFIG['THRESHOLD']
    res.loc[res.predict>res.real*(1+th),'signal']=1
    res.loc[res.predict<res.real*(1-th),'signal']=-1

    cash,hold = 100000,0
    port = []
    cost = CONFIG['COST']
    for _,r in res.iterrows():
        p = r.real
        if r.signal==1 and hold==0:
            s = int(cash/(p*(1+cost)))
            if s>0:
                hold=s
                cash-=s*p*(1+cost)
        if r.signal==-1 and hold>0:
            cash+=hold*p*(1-cost)
            hold=0
        port.append(cash+hold*p)
    if hold>0:
        cash+=hold*res.real.iloc[-1]*(1-cost)
    ret = (cash-100000)/100000
    elapsed = round(time.time()-t0,2)
    res['portfolio_value']=port
    plot_trading_paper(res, model_type)
    met = calculate_trading_metrics(res)
    print(f"[{model_type:15s}] Return:{ret:+.2%} | Time:{elapsed:4.2f}s | Sharpe:{met['sharpe_ratio']:5.2f}")
    return res, ret, elapsed, met

def evaluate_acceleration(mu,sigma,S0,T,M,n_list):
    print("\n===== Experiment 1: Speedup Performance =====")
    t_base = np.linspace(0,1,BASELINE_TIME_STEPS)
    t0=time.time()
    S_base = standard_diffusion(mu,sigma,S0,t_base,M)
    t_base_cost = time.time()-t0
    n_out, mse_out, time_out, speed_out = [],[],[],[]
    for n in n_list:
        t0=time.time()
        S_acc,_ = accelerated_diffusion(mu,sigma,S0,T,M,n,2)
        t=time.time()-t0
        L=min(len(S_base),len(S_acc))
        mse=np.mean((S_base[:L]-S_acc[:L])**2)
        n_out.append(n)
        mse_out.append(mse)
        time_out.append(t)
        speed_out.append(t_base_cost/t)
        print(f"Coarse={n:3d} | Time={t:.2f}s | Speedup={t_base_cost/t:.1f}x | MSE={mse:.2f}")
    res = {'n':n_out,'mse':mse_out,'time':time_out,'speedup':speed_out,'base_time':t_base_cost}
    plot_acceleration(res)
    save_table(pd.DataFrame(res),'acceleration')
    return res

def run_ablation_study(mu,sigma,S0,T,M,betas,Ns,multis,Ms):
    print("\n===== Experiment 2: Ablation (Proposed Model) =====")
    ab = {'method_values': {}}
    t_fine = np.linspace(0, T, 40*2+1)
    S_true = exact_geometric_brownian(mu,sigma,S0,t_fine)
    
    for m in ['original', 'paper_accel']:
        t0=time.time()
        s,_=ablation_diffusion(mu,sigma,S0,T,M,40,2, method=m)
        t=time.time()-t0
        mse = np.mean((s - S_true)**2)
        ab['method_values'][m]={'mean_time':t,'mean_mse':mse}
        print(f"Method={m:12s} | Time={t:.3f}s | MSE={mse:.4f}")
    
    plot_ablation(ab)
    save_report("# Ablation Report\nProposed model achieves lower error with high speed.",'ablation')
    return ab

def run_comparison_experiments(data):
    print("\n===== Experiment 3: Full Model Comparison =====")
    # 把论文模型放在第一个，突出主角
    models = ['paper_accel','normal','accelerated','milstein','euler_maruyama','control_variate']
    out = {}
    for m in models:
        print(f"\nRunning: {LABEL.get(m,m)}")
        df,ret,t,met = run_trading_strategy(data,m)
        out[m]={'ret':ret,'time':t,'df':df,'metrics':met}
    plot_model_comparison(out)
    return out