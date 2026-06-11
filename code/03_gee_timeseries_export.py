import ee
import geemap
import os

# ==============================================================================
# Phase 2 (拓展): 河套平原“VPD-盐分复合胁迫” 
# 任务：提取长时序生态气象指标 (2000-2023) 及 高纯度农作物掩膜
# ==============================================================================

def main():
    # 1. GEE 认证与初始化
    print("正在初始化 Google Earth Engine...")
    PROJECT_ID = 'proud-archery-463807-e6'
    
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception as e:
        print("初始化失败，请检查网络或授权状态:", e)
        return

    # 2. 定义本地保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将长时序数据放在 data/timeseries/ 下以保持整洁
    data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'timeseries')
    os.makedirs(data_dir, exist_ok=True)
    
    base_data_dir = os.path.dirname(data_dir)

    # 3. 划定研究区边界
    # 采用覆盖河套农业区的 Bounding Box
    roi = ee.Geometry.Polygon([
        [[106.1, 40.2], [109.4, 40.2], [109.4, 41.3], [106.1, 41.3], [106.1, 40.2]]
    ])

    # =========================================================
    # 核心任务 1：提取静态农作物掩膜 (基于 ESA WorldCover)
    # =========================================================
    print("\n========== 正在提取静态农作物掩膜 ==========")
    crop_mask_out = os.path.join(base_data_dir, 'Hetao_CropMask_WorldCover.tif')
    if not os.path.exists(crop_mask_out):
        print("下载 ESA WorldCover 农田掩膜 (代码 40 = Cropland)...")
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
        cropland = worldcover.select('Map').eq(40).clip(roi)
        # 为避免过大，分辨率设为 100m
        geemap.download_ee_image(cropland, crop_mask_out, scale=100, region=roi, crs='EPSG:4326')
    else:
        print("农作物掩膜已存在，跳过。")

    # =========================================================
    # 核心任务 2：循环下载 2000-2023 年夏季（6-9月）生态气象数据
    # 将同一年的 GPP, LST, NIRv, VPD, Tmax, SM 融合成一张多波段 TIFF
    # =========================================================
    print("\n========== 开始提取长时序生态气象指标 (2000-2023) ==========")
    years = range(2000, 2024)
    
    for year in years:
        out_file = os.path.join(data_dir, f'Hetao_EcoMeteo_Summer_{year}.tif')
        if os.path.exists(out_file):
            print(f"{year} 年多波段数据已存在，跳过。")
            continue
            
        print(f"正在构建 {year} 年的数据集...")
        summer_start = f'{year}-06-01'
        summer_end = f'{year}-09-30'

        # [MODIS] GPP
        gpp = (ee.ImageCollection("MODIS/061/MOD17A2H")
               .filterDate(summer_start, summer_end)
               .select('Gpp')
               .map(lambda img: img.multiply(0.0001))
               .mean().rename('GPP'))
        
        # [MODIS] LST (白天)
        lst = (ee.ImageCollection("MODIS/061/MOD11A2")
               .filterDate(summer_start, summer_end)
               .select('LST_Day_1km')
               .map(lambda img: img.multiply(0.02).subtract(273.15))
               .mean().rename('LST'))

        # [MODIS] NIRv (NDVI * NIR)
        def compute_nirv(img):
            ndvi = img.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
            nir = img.select('sur_refl_b02').multiply(0.0001)
            return img.addBands(ndvi.multiply(nir).rename('NIRv'))
            
        nirv = (ee.ImageCollection("MODIS/061/MOD09A1")
                .filterDate(summer_start, summer_end)
                .map(compute_nirv)
                .select('NIRv')
                .mean().rename('NIRv'))

        # [ERA5-Land] Tmax, SM, VPD
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(summer_start, summer_end)
        
        tmax = era5.select('temperature_2m').max().subtract(273.15).rename('Tmax')
        sm = era5.select('volumetric_soil_water_layer_1').mean().rename('SM')
        
        # VPD 计算
        tmean = era5.select('temperature_2m').mean().subtract(273.15)
        tdmean = era5.select('dewpoint_temperature_2m').mean().subtract(273.15)
        e_sat = tmean.expression('0.611 * exp(17.27 * T / (T + 237.3))', {'T': tmean})
        e_act = tdmean.expression('0.611 * exp(17.27 * Td / (Td + 237.3))', {'Td': tdmean})
        vpd = e_sat.subtract(e_act).max(0).rename('VPD')

        # [合并与裁剪]
        # 把今年的 6 个核心变量打包成一个带有 6 个波段的 Image
        yearly_img = ee.Image.cat([gpp, lst, nirv, tmax, sm, vpd]).clip(roi)

        print(f"-> 正在下载 {year} 年多波段 TIFF (包含 GPP/LST/NIRv/Tmax/SM/VPD)...")
        try:
            # 统一输出分辨率设定为 1000m（兼顾运算速度与本地存储）
            geemap.download_ee_image(yearly_img, out_file, scale=1000, region=roi, crs='EPSG:4326')
        except Exception as e:
            print(f"!!! {year} 年下载失败: {e}")

    print("\nPhase 2 长时序与掩膜拓展任务全部执行完毕！")

if __name__ == "__main__":
    main()
