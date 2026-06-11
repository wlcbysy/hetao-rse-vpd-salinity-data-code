import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap
import statsmodels.api as sm
import matplotlib.gridspec as gridspec

# ==============================================================================
# Phase 4.2: 顶刊级别主图绘制 (Premium Journal Aesthetic)
# ==============================================================================

def main():
    print("========== 正在生成 RSE/Nature 级高质量主图 ==========")
    
    # 1. 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'Hetao_Master_Dataset_2000_2023.csv')
    fig_dir = os.path.join(base_dir, 'figures')
    
    # 2. 读取全量真实数据
    df = pd.read_csv(data_file).replace([np.inf, -np.inf], np.nan).dropna()
    features = ['VPD', 'NDSI', 'SM', 'LST', 'Tmax']
    target = 'GPP'
    X = df[features]
    y = df[target]
    
    # 3. 模型重训与 SHAP (使用全量)
    print("拟合模型与计算 SHAP (全量 43,269 数据)...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_model.fit(X, y)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    vpd_idx = features.index('VPD')
    vpd_shap = shap_values[:, vpd_idx]
    vpd_data = X['VPD'].values
    ndsi_data = X['NDSI'].values

    # =========================================================================
    # 顶刊级绘图体系设计 (Premium Journal Aesthetic)
    # 设计理念：
    # 1. 采用纯英文标识，使用国际标准学术字体 (Arial / Helvetica)
    # 2. 边框使用全封闭加粗向内刻度 (inward ticks)，无顶标/边标赘余
    # 3. 散点 + KDE 密度等高线 (完美解决 4 万点过密导致的信息丢失)
    # 4. JointPlot 边缘分布 (Marginal Hist+KDE)，展示 VPD 与 SHAP 分布特征
    # 5. 内嵌悬浮式图例 (Inset Colorbar) 节约空间
    # =========================================================================
    print("开始构建联合密度坐标系...")
    
    # 统一字体字号和格式
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # 创建主网格
    fig = plt.figure(figsize=(10, 8), dpi=400)
    gs = gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.1)
    
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # 清理 Marginal axes 的坐标轴
    ax_top.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_right.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    sns.despine(ax=ax_top, left=True, right=True, top=True)
    sns.despine(ax=ax_right, bottom=True, right=True, top=True)
    
    # --- 主图区 (Main Scatter + Contour) ---
    print("绘制海量数据散点与密度等高线...")
    # 使用 Spectral_r 分散盐分，冷色调为低盐，暖色调(红)为高盐
    scatter = ax_main.scatter(vpd_data, vpd_shap, c=ndsi_data, cmap='Spectral_r', 
                              s=10, alpha=0.5, edgecolors='none', zorder=2)
    
    # 核心高级感设计：添加 KDE 核密度等高线，圈出 4万个点的“引力中心”
    sns.kdeplot(x=vpd_data, y=vpd_shap, ax=ax_main, levels=6, color='black', alpha=0.4, linewidths=0.8, zorder=3)
    
    # --- LOWESS 平滑拐点线 ---
    print("拟合非线性 LOWESS 趋势...")
    lowess = sm.nonparametric.lowess
    z = lowess(vpd_shap, vpd_data, frac=0.1, it=0)
    # 使用亮眼的纯黑或者纯红区分数据
    ax_main.plot(z[:, 0], z[:, 1], color='#FF0000', linewidth=3, zorder=4, label='LOWESS Trend')
    
    # Y=0 基准线
    ax_main.axhline(0, color='#888888', linestyle='--', linewidth=1.5, zorder=1)
    
    # 标签定制
    ax_main.set_xlabel("Vapor Pressure Deficit (VPD, kPa)", fontsize=16, fontweight='bold')
    ax_main.set_ylabel("SHAP Value of VPD (Impact on GPP)", fontsize=16, fontweight='bold')
    ax_main.tick_params(axis='both', which='major', labelsize=14)
    ax_main.legend(loc='lower left', frameon=True, fontsize=13, edgecolor='black')
    
    # --- 顶刊风格：内嵌悬浮 Colorbar ---
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # 将 Colorbar 嵌在主图的右上角空白处
    cax = inset_axes(ax_main, width="30%", height="4%", loc='upper right', borderpad=2)
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.set_label('NDSI (Salinity Index)', fontsize=12, fontweight='bold')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.tick_params(labelsize=10)
    
    # --- 边缘分布图 (Marginal Histograms) ---
    print("绘制边缘分布密度图...")
    sns.histplot(x=vpd_data, ax=ax_top, color='gray', bins=50, stat='density', alpha=0.3, edgecolor='none')
    sns.kdeplot(x=vpd_data, ax=ax_top, color='black', linewidth=1.5)
    
    sns.histplot(y=vpd_shap, ax=ax_right, color='gray', bins=50, stat='density', alpha=0.3, edgecolor='none')
    sns.kdeplot(y=vpd_shap, ax=ax_right, color='black', linewidth=1.5)
    
    ax_top.set_xlabel('')
    ax_top.set_ylabel('')
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    
    out_fig = os.path.join(fig_dir, '06_Premium_Main_Figure.png')
    plt.savefig(out_fig, bbox_inches='tight', dpi=500)  # 500 DPI 高清无损
    plt.close()
    
    print(f"========== 顶刊图表已生成: {out_fig} ==========")

if __name__ == "__main__":
    main()
