import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'savefig.bbox': 'tight'
    # 已删除 Arial 字体
})

COLOR = {
    'paper_accel': '#E63946',    # 亮红（论文模型高亮）
    'normal': '#457B9D',         # 蓝
    'accelerated': '#1D3557',    # 深蓝
    'milstein': '#F4A261',       # 橙
    'euler_maruyama': '#2A9D8F', # 绿
    'control_variate': '#9C89B6',# 紫
    'buy_hold': '#000000',
    'ma': '#FFB300',
    'macd': '#FF5E7D',
    'rsi': '#00B4D8'
}

LABEL = {
    'paper_accel': 'Proposed Accelerated Model (Our Paper)',
    'normal': 'Standard GBM',
    'accelerated': 'Multiscale Accelerated',
    'milstein': 'Milstein',
    'euler_maruyama': 'Euler-Maruyama',
    'control_variate': 'Control Variate'
}

for d in ['Plots', 'Tables', 'Reports']:
    os.makedirs(d, exist_ok=True)

# ---------------------------
# 论文图1：加速性能 vs 粗步数
# ---------------------------
def plot_acceleration(results, path='Plots/paper_acceleration.png'):
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.set_xlabel('Coarse Time Steps')
    ax1.set_ylabel('Speedup Ratio', color='#2E86AB')
    ax1.plot(results['n'], results['speedup'], 'o-', linewidth=2.5, markersize=6, color='#2E86AB', label='Speedup Ratio')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax2 = ax1.twinx()
    ax2.set_ylabel('MSE Error', color='#C73E1D')
    ax2.plot(results['n'], results['mse'], 's-', linewidth=2.5, markersize=6, color='#C73E1D', label='MSE')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')
    plt.title('Speedup vs Accuracy (Multiscale Acceleration)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------------------
# 论文图2：消融实验（论文模型优势）
# ---------------------------
def plot_ablation(ab, path='Plots/paper_ablation.png'):
    methods = ['Original', 'Proposed (Paper)']
    mse = [ab['method_values']['original']['mean_mse'],
           ab['method_values']['paper_accel']['mean_mse']]
    time_cost = [ab['method_values']['original']['mean_time'],
                 ab['method_values']['paper_accel']['mean_time']]
    
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, mse, width, label='MSE', color='#F4A261')
    ax.bar(x + width/2, time_cost, width, label='Time (s)', color='#2A9D8F')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Value')
    ax.set_title('Ablation Study: Proposed Model vs Baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------------------
# 论文图3：全模型收益对比（高亮论文模型）
# ---------------------------
def plot_model_comparison(comp, path='Plots/paper_model_comparison.png'):
    models = list(comp.keys())
    rets = [comp[m]['ret'] for m in models]
    times = [comp[m]['time'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,4.5))
    
    colors = [COLOR[m] for m in models]
    ax1.bar([LABEL.get(m,m) for m in models], rets, color=colors)
    ax1.set_ylabel('Total Return')
    ax1.set_title('Total Return Comparison')
    ax1.tick_params(axis='x', rotation=25)
    
    ax2.bar([LABEL.get(m,m) for m in models], times, color=colors)
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Computation Speed')
    ax2.tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------------------
# 论文图4：交易曲线（论文模型单独高亮）
# ---------------------------
def plot_trading_paper(df, model_name, path_prefix='Plots/paper_trading_'):
    plt.figure(figsize=(12,5))
    plt.plot(df.date, df.real, color='#666666', linewidth=1.2, label='Real Price')
    plt.plot(df.date, df.predict, '--', color='#0077B6', linewidth=1.5, label='Predicted Price')
    plt.plot(df.date, df.portfolio_value, color=COLOR[model_name], linewidth=2.5, label='Portfolio Value')
    
    buy = df[df.signal == 1]
    sell = df[df.signal == -1]
    plt.scatter(buy.date, buy.real, c='#2ECC71', marker='^', s=60, zorder=5, label='Buy')
    plt.scatter(sell.date, sell.real, c='#E74C3C', marker='v', s=60, zorder=5, label='Sell')
    
    plt.title(f'Trading Performance: {LABEL.get(model_name, model_name)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path_prefix}{model_name}.png')
    plt.close()

# ---------------------------
# 论文图5：最终综合对比图（收益/夏普/回撤/速度）
# ---------------------------
def plot_final_summary(all_results, paper_model='paper_accel', path='Plots/paper_final_summary.png'):
    models = [k for k in all_results.keys() if k in COLOR]
    summary = {}
    for m in models:
        ret = all_results[m]['ret'] * 100
        sharpe = all_results[m]['metrics']['sharpe_ratio']
        mdd = -all_results[m]['metrics']['max_drawdown'] * 100
        speed = 1.0 / (all_results[m]['time'] + 1e-6)
        summary[m] = [ret, sharpe, mdd, speed]
    
    keys = ['Return(%)', 'Sharpe', 'MaxDrawdown(-%)', 'Speed']
    fig, ax = plt.subplots(figsize=(9,5))
    for m in models:
        style = '-' if m != paper_model else '--'
        lw = 1.8 if m != paper_model else 4
        alpha = 0.7 if m != paper_model else 1.0
        ax.plot(keys, summary[m], marker='o', linewidth=lw, alpha=alpha, color=COLOR[m],
                label=LABEL.get(m,m), linestyle=style)
    
    ax.set_title('Overall Performance Summary (Proposed Model Highlighted)')
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------------------
# 论文图6：所有策略最终对比
# ---------------------------
def plot_all_strategy_comparison(all_results, path='Plots/paper_all_comparison.png'):
    names = list(all_results.keys())
    returns = [all_results[n]['ret'] for n in names]
    sharpes = [all_results[n]['metrics']['sharpe_ratio'] for n in names]
    mdds = [all_results[n]['metrics']['max_drawdown'] for n in names]
    times = [all_results[n]['time'] for n in names]
    
    fig, axes = plt.subplots(2,2,figsize=(15,8))
    (ax1,ax2),(ax3,ax4)=axes
    colors = [COLOR.get(n,'#999') for n in names]
    
    ax1.bar([LABEL.get(n,n) for n in names], returns, color=colors)
    ax1.set_title('Total Return')
    ax1.tick_params(axis='x', rotation=30)
    
    ax2.bar([LABEL.get(n,n) for n in names], sharpes, color=colors)
    ax2.set_title('Sharpe Ratio')
    ax2.tick_params(axis='x', rotation=30)
    
    ax3.bar([LABEL.get(n,n) for n in names], mdds, color=colors)
    ax3.set_title('Max Drawdown')
    ax3.tick_params(axis='x', rotation=30)
    
    ax4.bar([LABEL.get(n,n) for n in names], times, color=colors)
    ax4.set_title('Computation Time (s)')
    ax4.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_table(df, name):
    df.to_csv(f'Tables/paper_{name}.csv', index=False)

def save_report(text, name):
    with open(f'Reports/paper_{name}.md', 'w', encoding='utf-8') as f:
        f.write(text)