// ==============================================================================
// Phase 2: 河套平原“VPD-盐分复合胁迫” GEE 数据提取脚本 (JavaScript)
// 平台：Google Earth Engine Code Editor (https://code.earthengine.google.com/)
// ==============================================================================

// 1. 研究区边界划定 (ROI: 巴彦淖尔市主要农业旗县 - 临河区, 五原县, 杭锦后旗, 乌拉特前旗)
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level2");
var roi = gaul.filter(ee.Filter.inList('ADM2_NAME', ['Linhe', 'Wuyuan', 'Hanggin Hou', 'Urat Qian']))
              .geometry();

Map.centerObject(roi, 8);
Map.addLayer(roi, {color: 'red'}, 'Hetao Plain ROI (Selected Counties)');

// 时间范围设定
var startDate = '2019-01-01'; // 可根据研究需要调整 (如 2000-2023)
var endDate = '2023-12-31';
var springStart = '04-15'; // 裸土期
var springEnd = '05-15';
var summerStart = '06-01'; // 生长季
var summerEnd = '09-30';

// ==============================================================================
// 2. 核心数据源处理
// ==============================================================================

// ------------------------------------------------------------------------------
// 2.1 Sentinel-2 土壤盐分基底 (NDSI) - 裸土期 (4月中下旬-5月上旬)
// ------------------------------------------------------------------------------
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// 提取指定年份的春季裸土期 NDSI
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(roi)
  .filterDate('2023-' + springStart, '2023-' + springEnd) // 示例：提取2023年
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

// NDSI = (Green - SWIR1) / (Green + SWIR1) 或者其他常用盐分指数配方
var computeNDSI = function(img) {
  var ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI');
  return img.addBands(ndsi);
};

var s2_ndsi = s2.map(computeNDSI).select('NDSI').median().clip(roi);
Map.addLayer(s2_ndsi, {min: -0.2, max: 0.2, palette: ['blue', 'white', 'red']}, 'Spring NDSI (2023)');

// ------------------------------------------------------------------------------
// 2.2 ERA5-Land 气象驱动力 (VPD, Tmax, Soil Moisture)
// ------------------------------------------------------------------------------
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(roi)
  .filterDate(startDate, endDate);

// 计算每日 Tmax 和 VPD (需要将每小时数据聚合成每日)
// ERA5 温度单位是开尔文 (K)，需要转摄氏度 (C)
// 简化的 VPD 计算逻辑：基于 T(气温) 和 Td(露点温度)
var computeDailyMeteo = function(date) {
  var dayStr = ee.Date(date).format('YYYY-MM-dd');
  var dailyImages = era5.filterDate(dayStr, ee.Date(date).advance(1, 'day'));
  
  // Tmax (最高温度)
  var tmax = dailyImages.select('temperature_2m').max().subtract(273.15).rename('Tmax');
  
  // 土壤水分 (第一层 0-7cm)
  var sm = dailyImages.select('volumetric_soil_water_layer_1').mean().rename('SM');
  
  // VPD 近似计算: e_sat - e_act
  var tmean = dailyImages.select('temperature_2m').mean().subtract(273.15);
  var tdmean = dailyImages.select('dewpoint_temperature_2m').mean().subtract(273.15);
  
  var e_sat = tmean.expression('0.611 * exp(17.27 * T / (T + 237.3))', {'T': tmean});
  var e_act = tdmean.expression('0.611 * exp(17.27 * Td / (Td + 237.3))', {'Td': tdmean});
  var vpd = e_sat.subtract(e_act).max(0).rename('VPD'); // kPa
  
  return tmax.addBands(sm).addBands(vpd)
             .set('system:time_start', ee.Date(dayStr).millis());
};

// 获取日期列表并映射计算 (这里仅示例计算10天，实际可按月或导出完整时间序列)
var days = ee.List.sequence(0, 10).map(function(n) {
  return ee.Date('2023-07-01').advance(n, 'day');
});
var daily_meteo_col = ee.ImageCollection.fromImages(days.map(computeDailyMeteo));


// ------------------------------------------------------------------------------
// 2.3 MODIS 植被生理指标 (GPP, NIRv, LST)
// ------------------------------------------------------------------------------
// GPP (MOD17A2H - 8天合成)
var modis_gpp = ee.ImageCollection("MODIS/061/MOD17A2H")
  .filterBounds(roi)
  .filterDate('2023-' + summerStart, '2023-' + summerEnd)
  .select('Gpp')
  .map(function(img){ return img.multiply(0.0001).clip(roi).set('system:time_start', img.get('system:time_start')); });

// LST (MOD11A2 - 8天合成)
var modis_lst = ee.ImageCollection("MODIS/061/MOD11A2")
  .filterBounds(roi)
  .filterDate('2023-' + summerStart, '2023-' + summerEnd)
  .select('LST_Day_1km')
  .map(function(img){ return img.multiply(0.02).subtract(273.15).clip(roi).rename('LST_Day'); });


// ==============================================================================
// 3. 导出任务配置 (Export Tasks)
// ==============================================================================

// 导出 NDSI 图像至 Google Drive
Export.image.toDrive({
  image: s2_ndsi,
  description: 'Hetao_Spring_NDSI_2023',
  folder: 'Hetao_Compound_Stress_Data', // 对应本地的 data/ 文件夹
  region: roi,
  scale: 10, // Sentinel-2分辨率
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 提示：您可以将此脚本复制到 GEE Code Editor 中运行。
// 时序数据（GPP/VPD）由于维度较高，建议针对各个像素导出为 CSV 时序或按波段堆叠的 TIFF。
