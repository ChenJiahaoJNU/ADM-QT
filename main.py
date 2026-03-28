import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from trading_strategy import *
from technical_strategies import (
    run_ma_strategy, run_macd_strategy,
    run_rsi_strategy, run_buy_hold_strategy
)
from visualization import *

EXPERIMENT = {
    'coarse_list':[40,50,80,100],
    'beta':[8,10,12,15],
    'N':[40,50,60,77],
    'multi':[2],
    'M':[200]
}

def main():
    print("="*80)
    print("       PAPER VERSION: Accelerated Diffusion Model for Trading")
    print("="*80)

    try:
        spx = pd.read_csv('SP500.csv')
        spx['date'] = pd.to_datetime(spx['date'])
        spx = spx[(spx['date'] >= '2021-01-01') & (spx['date'] <= '2022-12-31')].copy()
        print(f"Data loaded: {len(spx)} rows (2021-2022)")
    except:
        print("Error: SP500.csv not found.")
        return

    spx['ret'] = np.log(spx.close/spx.close.shift(1))
    mu = spx.ret.mean()*252
    sigma= spx.ret.std()*np.sqrt(252)
    print(f"Fitted: mu={mu:.3f}, sigma={sigma:.3f}")

    evaluate_acceleration(mu, sigma, 100, 1.0, 200, EXPERIMENT['coarse_list'])
    run_ablation_study(mu,sigma,100,1.0,200,
                       EXPERIMENT['beta'],
                       EXPERIMENT['N'],
                       EXPERIMENT['multi'],
                       EXPERIMENT['M'])
    comp = run_comparison_experiments(spx)

    print("\n===== Traditional Strategies =====")
    traditional = {}
    for name, func in [
        ('buy_hold', run_buy_hold_strategy),
        ('ma', run_ma_strategy),
        ('macd', run_macd_strategy),
        ('rsi', run_rsi_strategy)
    ]:
        df, ret, t = func(spx)
        met = calculate_trading_metrics(df)
        traditional[name] = {'ret':ret,'time':t,'df':df,'metrics':met}
        print(f"[{name:10s}] Return:{ret:+.2%} | Time:{t:.2f}s | Sharpe:{met['sharpe_ratio']:.2f}")

    all_results = {**comp,**traditional}
    plot_all_strategy_comparison(all_results)
    plot_final_summary(all_results)

    lines = ["# Paper: Final Performance Report\n"]
    for name, info in all_results.items():
        lines.append(f"- {LABEL.get(name,name):25s} | Return:{info['ret']:6.2%} | Sharpe:{info['metrics']['sharpe_ratio']:5.2f} | Time:{info['time']:5.2f}s")
    save_report('\n'.join(lines), 'final_report')

    print("\n===== PAPER CONCLUSION =====")
    r_paper = comp['paper_accel']['ret']
    t_paper = comp['paper_accel']['time']
    r_normal = comp['normal']['ret']
    t_normal = comp['normal']['time']
    print(f"✅ Proposed Model: Return={r_paper:.2%}, Time={t_paper:.2f}s")
    print(f"✅ Standard GBM  : Return={r_normal:.2%}, Time={t_normal:.2f}s")
    print(f"\n✅ All paper figures & tables saved!")

if __name__ == '__main__':
    main()