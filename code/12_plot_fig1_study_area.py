import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from figure_common import salinity_quantiles

# ==============================================================================
# Phase 7: 构建 Fig 1 (Study Area & Geographical Location)
# 任务:
# 1. 使用 Cartopy + GridSpec 构建完全一致的矢量级排版，保证字体字号绝对统一
# 2. Panel A: 高清地形图+黄河水系+大图套小图 (Inset Map)
# 3. Panel B, C: 2023 真实数据分布与边缘密度
# ==============================================================================

def add_scale_bar(ax, lon, lat, length_km=300):
    """Draw an approximate longitude scale bar at a fixed latitude."""
    length_deg = length_km / (111.32 * np.cos(np.deg2rad(lat)))
    ax.plot(
        [lon, lon + length_deg],
        [lat, lat],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.plot(
        [lon, lon],
        [lat - 0.12, lat + 0.12],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2,
        zorder=8,
    )
    ax.plot(
        [lon + length_deg, lon + length_deg],
        [lat - 0.12, lat + 0.12],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2,
        zorder=8,
    )
    ax.text(
        lon + length_deg / 2,
        lat + 0.28,
        f"{length_km} km",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        zorder=9,
    )


def main():
    print("========== 开始生成 Fig 1: 顶刊级研究区与水土环境地图 ==========")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'Hetao_Master_Dataset_2000_2023.csv')
    fig_dir = os.path.join(base_dir, 'figures')
    
    # 1. 读取数据
    df = pd.read_csv(data_file).replace([np.inf, -np.inf], np.nan).dropna()
    df_2023 = df[df['Year'] == 2023]
    ndsi_33, ndsi_66 = salinity_quantiles(df)
    
    lon_min, lon_max = df_2023['Lon'].min(), df_2023['Lon'].max()
    lat_min, lat_max = df_2023['Lat'].min(), df_2023['Lat'].max()
    
    # 统一风格
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    
    # 创建全局画布 (优化长宽比以适应对齐)
    fig = plt.figure(figsize=(20, 10), dpi=400)
    # 调整宽高比和间距，使左右两边的顶部和底部对齐，并增加 wspace 防止 Y 轴遮挡
    gs = GridSpec(2, 3, width_ratios=[2.2, 2, 0.4], wspace=0.35, hspace=0.25)
    
    # === Panel A: Location Map (Topography + Rivers) ===
    print("-> 正在绘制 Panel A (高程地形与黄河水系)...")
    ax_a = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    
    # 视角聚焦黄河流域/西北 (扩大纬度跨度以增加高度，使其自动撑满上下空间并对齐右侧)
    ax_a.set_extent([98, 115, 30, 47], crs=ccrs.PlateCarree())
    
    # 引入高阶自然底图 (DEM shaded relief)
    ax_a.stock_img()
    
    # 添加水系、国界、海岸线
    ax_a.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=1.5, alpha=0.6)
    ax_a.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.6)
    ax_a.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black')
    ax_a.add_feature(cfeature.COASTLINE, linewidth=1)
    
    # 添加省界 (可选，cartopy 默认无高精度省界，使用 STATES 近似或跳过)
    # ax_a.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray', linestyle=':')
    
    # 画出河套红框
    rect = Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                     linewidth=3.5, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    ax_a.add_patch(rect)
    add_scale_bar(ax_a, 110.2, 31.2, length_km=300)
    
    # 注释说明 (移至右上角空白区域，避免与任何图层重叠)
    ax_a.annotate('Hetao Irrigation\nDistrict', xy=((lon_min+lon_max)/2, lat_max), xytext=((lon_min+lon_max)/2 + 2.5, lat_max + 2.5),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                    fontsize=16, fontweight='bold', color='darkred', ha='center',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'), transform=ccrs.PlateCarree())
    ax_a.text(0.54, 0.28, "Yellow River corridor", transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', color='navy',
              bbox=dict(facecolor='white', alpha=0.65, edgecolor='none'))
    
    gl = ax_a.gridlines(draw_labels=True, linewidth=0.8, color='black', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}
    
    # Inset Map (缩略全图) - 置于左下角 (青藏高原区域，研究区外，绝对空白)
    print("-> 正在绘制 Panel A 嵌套缩略图...")
    ax_inset = inset_axes(ax_a, width="35%", height="35%", loc="lower left",
                          axes_class=cartopy.mpl.geoaxes.GeoAxes,
                          axes_kwargs=dict(projection=ccrs.PlateCarree()))
    ax_inset.set_extent([70, 140, 15, 55])
    ax_inset.add_feature(cfeature.LAND, facecolor='#EAEAEA')
    ax_inset.add_feature(cfeature.OCEAN, facecolor='#D6EAF8')
    ax_inset.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # 在缩略图上用红点标出位置
    ax_inset.plot((lon_min+lon_max)/2, (lat_min+lat_max)/2, 'ro', markersize=8, transform=ccrs.PlateCarree())
    # 增加一个红框框住缩放区域
    inset_rect = Rectangle((98, 32), 115-98, 45-32, linewidth=1.5, edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree())
    ax_inset.add_patch(inset_rect)
    
    ax_a.text(-0.05, 1.02, "(a)", transform=ax_a.transAxes, size=22, weight='bold')
    
    # === Panel B: GPP Map ===
    print("-> 正在绘制 Panel B (GPP 空间地图)...")
    ax_b_map = fig.add_subplot(gs[0, 1])
    sc_b = ax_b_map.scatter(df_2023['Lon'], df_2023['Lat'], c=df_2023['GPP'], cmap='YlGn', s=1.5, alpha=0.9)
    ax_b_map.set_xlabel("Longitude (°E)", fontsize=16, fontweight='bold')
    ax_b_map.set_ylabel("Latitude (°N)", fontsize=16, fontweight='bold')
    ax_b_map.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                  arrowprops=dict(facecolor='black', width=3, headwidth=10),
                  xycoords='axes fraction', textcoords='axes fraction',
                  fontsize=16, fontweight='bold', ha='center', va='center')
    ax_b_map.grid(True, linestyle=':', alpha=0.5)
    sns.despine(ax=ax_b_map, top=True, right=True)
    
    ax_b_map.text(-0.15, 1.05, "(b)", transform=ax_b_map.transAxes, size=22, weight='bold')
    
    # Panel B: Density
    ax_b_den = fig.add_subplot(gs[0, 2])
    sns.kdeplot(y=df_2023['GPP'], ax=ax_b_den, fill=True, color='green', alpha=0.5)
    ax_b_den.set_ylabel("")
    ax_b_den.set_xlabel("Pixel density", fontsize=16, fontweight='bold')
    ax_b_den.tick_params(labelleft=False, labelsize=14)
    sns.despine(ax=ax_b_den, top=True, right=True, left=True)
    
    # 把色带挂在 density plot 右边，防止卡在 map 和 density 中间
    cbar_b = plt.colorbar(sc_b, ax=ax_b_den, fraction=0.2, pad=0.1)
    cbar_b.set_label("GPP (gC/m²/day)", fontsize=16, fontweight='bold')
    cbar_b.ax.tick_params(labelsize=14)
    ax_b_map.tick_params(labelsize=14)
    
    # === Panel C: NDSI Map ===
    print("-> 正在绘制 Panel C (NDSI 空间地图)...")
    ax_c_map = fig.add_subplot(gs[1, 1])
    sc_c = ax_c_map.scatter(df_2023['Lon'], df_2023['Lat'], c=df_2023['NDSI'], cmap='YlOrBr', s=1.5, alpha=0.9)
    ax_c_map.set_xlabel("Longitude (°E)", fontsize=16, fontweight='bold')
    ax_c_map.set_ylabel("Latitude (°N)", fontsize=16, fontweight='bold')
    ax_c_map.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                  arrowprops=dict(facecolor='black', width=3, headwidth=10),
                  xycoords='axes fraction', textcoords='axes fraction',
                  fontsize=16, fontweight='bold', ha='center', va='center')
    ax_c_map.grid(True, linestyle=':', alpha=0.5)
    sns.despine(ax=ax_c_map, top=True, right=True)
    
    ax_c_map.text(-0.15, 1.05, "(c)", transform=ax_c_map.transAxes, size=22, weight='bold')
    
    # Panel C: Density
    ax_c_den = fig.add_subplot(gs[1, 2])
    sns.kdeplot(y=df_2023['NDSI'], ax=ax_c_den, fill=True, color='orange', alpha=0.5)
    ax_c_den.axhline(ndsi_33, color='#2c7fb8', linestyle='--', linewidth=1.4, label='33%')
    ax_c_den.axhline(ndsi_66, color='#d95f0e', linestyle='--', linewidth=1.4, label='66%')
    ax_c_den.set_ylabel("")
    ax_c_den.set_xlabel("Pixel density", fontsize=16, fontweight='bold')
    ax_c_den.tick_params(labelleft=False, labelsize=14)
    sns.despine(ax=ax_c_den, top=True, right=True, left=True)
    
    # 把色带挂在 density plot 右边
    cbar_c = plt.colorbar(sc_c, ax=ax_c_den, fraction=0.2, pad=0.1)
    cbar_c.set_label("NDSI (higher = saltier)", fontsize=16, fontweight='bold')
    cbar_c.ax.tick_params(labelsize=14)
    ax_c_map.tick_params(labelsize=14)
    ax_c_den.text(0.98, ndsi_33, "33%", transform=ax_c_den.get_yaxis_transform(),
                  color='#2c7fb8', fontsize=10, va='bottom', ha='right',
                  bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.0))
    ax_c_den.text(0.98, ndsi_66, "66%", transform=ax_c_den.get_yaxis_transform(),
                  color='#d95f0e', fontsize=10, va='bottom', ha='right',
                  bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.0))
    
    out_final = os.path.join(fig_dir, 'Final_Fig1_StudyArea_Concept.png')
    plt.savefig(out_final, bbox_inches='tight')
    plt.close()
    
    print(f"--> 全新矢量统一版 Fig 1 生成完毕: {out_final}")

if __name__ == "__main__":
    main()
