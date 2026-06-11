import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import shap

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("========== Phase 4: Random Forest & SHAP 机制解耦 ==========")
    
    # 1. 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'Hetao_Master_Dataset_2000_2023.csv')
    fig_dir = os.path.join(base_dir, 'figures')
    
    # 2. 数据读取与准备
    print("正在加载主数据集...")
    df = pd.read_csv(data_file)
    # 清洗掉可能的无穷大 (Inf) 值，防止 sklearn 报错
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 特征与目标变量定义
    features = ['VPD', 'NDSI', 'SM', 'LST', 'Tmax']
    target = 'GPP'
    
    X = df[features]
    y = df[target]
    
    print(f"数据总样本量: {len(df)}")
    
    # 划分训练集与测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 随机森林模型训练
    print("\n正在训练 Random Forest 机器学习模型 (这可能需要几十秒)...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"--> 模型评估完成 | R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # 4. 特征重要性作图 (Feature Importance)
    print("\n正在绘制特征重要性排名图...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '01_RF_Feature_Importance.png'), dpi=300)
    plt.close()
    print("--> 01_RF_Feature_Importance.png 已保存！")
    
    # 5. SHAP 解释分析
    print("\n正在构建 SHAP 解释器并计算归因值 (请稍作等待)...")
    # 由于原始样本量近 4.5 万，为了在画图时兼顾速度与美观，我们随机采样 8000 个点用于图解
    X_sample = shap.utils.sample(X, 8000)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    # 5.1 SHAP Summary Plot
    print("正在绘制 SHAP 摘要图 (Summary Plot)...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '02_SHAP_Summary_Plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("--> 02_SHAP_Summary_Plot.png 已保存！")
    
    # 5.2 SHAP Dependence Plot (VPD 与 NDSI 复合交互作用)
    print("正在绘制 VPD 与 NDSI(盐分) 复合胁迫机制图 (Dependence Plot)...")
    plt.figure()
    # interaction_index="NDSI" 表示在 VPD 的散点图上，将点的颜色映射为 NDSI，揭示二者的复合效应
    shap.dependence_plot("VPD", shap_values, X_sample, interaction_index="NDSI", show=False, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '03_SHAP_Dependence_VPD_NDSI.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("--> 03_SHAP_Dependence_VPD_NDSI.png 已保存！")

    print("\n========== Phase 4 顺利跑完！ ==========")
    print(f"所有的成果统计图表已存放在: {fig_dir} 文件夹中，您可以点开查看了！")

if __name__ == "__main__":
    main()
