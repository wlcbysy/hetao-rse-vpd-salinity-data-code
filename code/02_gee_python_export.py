import ee
import geemap
import os

# ==============================================================================
# Phase 2: 河套平原“VPD-盐分复合胁迫” GEE 本地自动化下载脚本 (Python)
# 依赖包: pip install earthengine-api geemap
# ==============================================================================

def main():
    # 1. GEE 认证与初始化
    print("正在初始化 Google Earth Engine...")
    # 注意：最新版 GEE Python API 必须绑定一个 Google Cloud Project。
    # 如果运行报错 "no project found"，请在此处填入您的项目 ID（例如 'ee-您的用户名'）
    PROJECT_ID = 'proud-archery-463807-e6'  
    
    try:
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()
    except Exception as e:
        print("尝试认证...")
        ee.Authenticate()
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()

    # 2. 定义本地保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(current_dir), 'data')
    os.makedirs(out_dir, exist_ok=True)
    print(f"数据输出目录: {out_dir}")

    # 3. 划定研究区边界 (包络框替代)
    # 由于 FAO GAUL 县级名称匹配问题可能导致边界为空（引发 Image.clip 空白错误），
    # 此处先使用覆盖河套平原核心农业区的矩形 Bounding Box 作为测试。
    # 后续阶段可以直接读取本地准备好的 Shapefile： geemap.shp_to_ee("path/to/shp")
    roi = ee.Geometry.Polygon([
        [[106.1, 40.2], [109.4, 40.2], [109.4, 41.3], [106.1, 41.3], [106.1, 40.2]]
    ])

    # 4. 时间参数设定 (以 2023 年为例)
    year = 2023
    spring_start = f'{year}-04-15'
    spring_end = f'{year}-05-15'
    summer_start = f'{year}-06-01'
    summer_end = f'{year}-09-30'

    # =========================================================
    # 提取 Sentinel-2 春季裸土 NDSI (空间分辨率 10m -> 为了下载可能需重采样)
    # =========================================================
    print(f"正在处理 {year} 年春季 Sentinel-2 NDSI...")
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    def compute_ndsi(img):
        ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI')
        return img.addBands(ndsi)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(roi)
          .filterDate(spring_start, spring_end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .map(mask_s2_clouds))

    s2_ndsi = s2.map(compute_ndsi).select('NDSI').median().clip(roi)
    
    ndsi_out = os.path.join(out_dir, f'Hetao_NDSI_Spring_{year}.tif')
    if not os.path.exists(ndsi_out):
        print("开始下载 Sentinel-2 NDSI (由于区域较大，设定下载分辨率为 100m 以避免超出内存限制)...")
        # 注意: 原始 10m 像元直接下载到本地可能会内存溢出(Payload too large)，这里将 scale 设为 100 仅作示例
        # 若需高精度，请使用 geemap.ee_export_image_to_drive 导出到云盘
        geemap.download_ee_image(s2_ndsi, ndsi_out, scale=100, region=roi, crs='EPSG:4326')
    else:
        print("NDSI 文件已存在，跳过下载。")

    # =========================================================
    # 提取 MODIS GPP (MOD17A2H - 生长季平均示例)
    # =========================================================
    print(f"正在处理 {year} 年夏季 MODIS GPP...")
    modis_gpp = (ee.ImageCollection("MODIS/061/MOD17A2H")
                 .filterBounds(roi)
                 .filterDate(summer_start, summer_end)
                 .select('Gpp')
                 .map(lambda img: img.multiply(0.0001)))
                 
    mean_gpp = modis_gpp.mean().clip(roi)
    gpp_out = os.path.join(out_dir, f'Hetao_Mean_GPP_Summer_{year}.tif')
    if not os.path.exists(gpp_out):
        print("开始下载 MODIS 夏季平均 GPP (分辨率 500m)...")
        geemap.download_ee_image(mean_gpp, gpp_out, scale=500, region=roi, crs='EPSG:4326')
    else:
        print("GPP 文件已存在，跳过下载。")

    # =========================================================
    # 提取 ERA5-Land 夏季平均 VPD (饱和水汽压差)
    # =========================================================
    print(f"正在处理 {year} 年夏季 ERA5-Land VPD...")
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterBounds(roi)
            .filterDate(summer_start, summer_end))

    # Tmean, Tdmean 转换为摄氏度
    tmean = era5.select('temperature_2m').mean().subtract(273.15)
    tdmean = era5.select('dewpoint_temperature_2m').mean().subtract(273.15)
    
    # 饱和水汽压 (e_sat) 与 实际水汽压 (e_act) 计算公式 [单位: kPa]
    e_sat = tmean.expression('0.611 * exp(17.27 * T / (T + 237.3))', {'T': tmean})
    e_act = tdmean.expression('0.611 * exp(17.27 * Td / (Td + 237.3))', {'Td': tdmean})
    vpd = e_sat.subtract(e_act).max(0).rename('VPD_Summer_Mean').clip(roi)

    vpd_out = os.path.join(out_dir, f'Hetao_Mean_VPD_Summer_{year}.tif')
    if not os.path.exists(vpd_out):
        print("开始下载 ERA5-Land 夏季平均 VPD (分辨率约 11132m)...")
        geemap.download_ee_image(vpd, vpd_out, scale=11132, region=roi, crs='EPSG:4326')
    else:
        print("VPD 文件已存在，跳过下载。")

    print("\n所有处理与下载任务运行完毕！数据已保存至:", out_dir)

if __name__ == "__main__":
    main()
