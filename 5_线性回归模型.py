##### 导入库 #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

##### 设置绘图风格 #####
sns.set_theme(style="white")
sns.set_theme(style="ticks")

##### 设置中文 #####
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  
plt.rcParams['font.family'] = ['sans-serif']

##### 创建文件夹 #####
import os
if not os.path.exists("图片"):
    os.makedirs("图片")
    
if not os.path.exists("结果表"):
    os.makedirs("结果表")

##### 1. 读取数据集 #####
print("##### 读取数据集 #####")
data = pd.read_csv("python回归数据.csv", encoding='gbk')
print(f"数据集形状: {data.shape}")

##### 2. 数据准备 #####
print("\n##### 数据准备 #####")
X = data.drop('周期产量', axis=1)
y = data['周期产量']

##### 3. 使用IQR过滤目标变量极端值 #####
print("\n##### 使用IQR过滤目标变量极端值 #####")
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

print(f"IQR = {IQR:.4f}")
print(f"下界 = {lower_bound:.4f}")
print(f"上界 = {upper_bound:.4f}")

# 过滤极端值
mask = (y >= lower_bound) & (y <= upper_bound)
X = X[mask]
y = y[mask]

print(f"过滤前样本数: {len(data)}")
print(f"过滤后样本数: {len(X)}")
print(f"过滤掉的样本数: {len(data) - len(X)}")

##### 4. 定义尾部MAE计算函数 #####
def calculate_tail_mae(y_true, y_pred):
    """
    计算尾部MAE（取周期产量的后30%计算的MAE）
    """
    threshold = np.percentile(y_true, 70)
    mask = y_true >= threshold
    return mean_absolute_error(y_true[mask], y_pred[mask])

##### 5. 读取所有模型的评估结果 #####
print("\n##### 读取所有模型的评估结果 #####")
rf_results = pd.read_csv('结果表/随机森林评估结果.csv', encoding='utf-8-sig')
xgb_results = pd.read_csv('结果表/XGBoost评估结果.csv', encoding='utf-8-sig')
lgb_results = pd.read_csv('结果表/LightGBM评估结果.csv', encoding='utf-8-sig')
nn_results = pd.read_csv('结果表/神经网络评估结果.csv', encoding='utf-8-sig')

all_results = pd.concat([rf_results, xgb_results, lgb_results, nn_results], ignore_index=True)
print("所有模型评估结果:")
print(all_results)

##### 6. 选择最优模型（根据MAE>尾部MAE>R²>RMSE） #####
print("\n##### 选择最优模型 #####")
all_results['排序得分'] = all_results.apply(
    lambda row: (row['MAE'], row['尾部MAE'], -row['R²'], row['RMSE']), axis=1
)
best_model_row = all_results.loc[all_results['排序得分'].idxmin()]
best_model_name = best_model_row['模型']
print(f"最优模型: {best_model_name}")
print(f"最优模型MAE: {best_model_row['MAE']:.4f}")
print(f"最优模型尾部MAE: {best_model_row['尾部MAE']:.4f}")
print(f"最优模型R²: {best_model_row['R²']:.4f}")
print(f"最优模型RMSE: {best_model_row['RMSE']:.4f}")

##### 7. 加载最优模型 #####
print("\n##### 加载最优模型 #####")
model_file_map = {
    '随机森林': '结果表/随机森林模型.pkl',
    'XGBoost': '结果表/XGBoost模型.pkl',
    'LightGBM': '结果表/LightGBM模型.pkl',
    '神经网络': '结果表/神经网络模型.pkl'
}

with open(model_file_map[best_model_name], 'rb') as f:
    best_model = pickle.load(f)
print(f"最优模型 {best_model_name} 已加载")

##### 8. 获取最优模型的重要特征 #####
print("\n##### 获取最优模型的重要特征 #####")

# 获取所有特征的重要性
full_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': best_model.feature_importances_
}).sort_values('重要性', ascending=False)

# 选择前12个最重要的特征
top_features = full_importance.head(12)['特征'].tolist()
print(f"选择的Top12特征: {top_features}")

##### 9. 使用最优模型预测所有数据 #####
print("\n##### 使用最优模型预测所有数据 #####")
y_original = best_model.predict(X)
print(f"预测完成，共 {len(y_original)} 个样本")

##### 10. 数据增强 #####
print("\n##### 数据增强 #####")

# 使用原始数据集
X_original = X[top_features]

# 为每个观测值生成10个邻域值
augmented_X_list = []
augmented_y_list = []

np.random.seed(42)
for i in range(len(X_original)):
    # 添加原始样本
    augmented_X_list.append(X_original.iloc[i].values)
    augmented_y_list.append(y_original[i])
    
    # 生成10个邻域样本
    for j in range(10):
        # 对每个特征添加小的随机扰动（正态分布，标准差为特征标准差的5%）
        noise = np.random.normal(0, X_original.std().values * 0.05)
        augmented_sample = X_original.iloc[i].values + noise
        augmented_X_list.append(augmented_sample)
        
        # 预测增强后的样本
        augmented_sample_full = X.iloc[i].copy()
        augmented_sample_full[top_features] = augmented_sample
        augmented_y_list.append(best_model.predict(augmented_sample_full.values.reshape(1, -1))[0])

X_augmented = pd.DataFrame(augmented_X_list, columns=top_features)
y_augmented = pd.Series(augmented_y_list)

print(f"增强后的数据集大小: {X_augmented.shape}")
print(f"原始数据集大小: {X_original.shape}")
print(f"增强倍数: {len(X_augmented) / len(X_original):.1f}倍")

##### 11. 训练线性回归模型 #####
print("\n##### 训练线性回归模型 #####")

# 划分训练集和测试集
X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42
)

print(f"增强后训练集大小: {X_aug_train.shape}")
print(f"增强后测试集大小: {X_aug_test.shape}")

# 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_aug_train, y_aug_train)

# 预测
lr_pred = lr_model.predict(X_aug_test)

# 计算评估指标
lr_mae = mean_absolute_error(y_aug_test, lr_pred)
lr_r2 = r2_score(y_aug_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_aug_test, lr_pred))
lr_tail_mae = calculate_tail_mae(y_aug_test.values, lr_pred)

print(f"线性回归 MAE: {lr_mae:.4f}")
print(f"线性回归 R²: {lr_r2:.4f}")
print(f"线性回归 RMSE: {lr_rmse:.4f}")
print(f"线性回归 尾部MAE: {lr_tail_mae:.4f}")

##### 12. 保存线性回归评估结果 #####
print("\n##### 保存线性回归评估结果 #####")
lr_results_df = pd.DataFrame({
    '指标': ['MAE', 'R²', 'RMSE', '尾部MAE'],
    '数值': [lr_mae, lr_r2, lr_rmse, lr_tail_mae]
})
lr_results_df.to_csv('结果表/线性回归评估结果.csv', index=False, encoding='utf-8-sig')
print("线性回归评估结果已保存")
print(lr_results_df)

##### 13. 保存线性回归系数 #####
print("\n##### 保存线性回归系数 #####")
lr_coef_df = pd.DataFrame({
    '特征': top_features,
    '系数': lr_model.coef_
})
lr_coef_df = pd.concat([
    lr_coef_df,
    pd.DataFrame({'特征': ['截距'], '系数': [lr_model.intercept_]})
], ignore_index=True)
lr_coef_df.to_csv('结果表/线性回归系数.csv', index=False, encoding='utf-8-sig')
print("线性回归系数已保存")
print(lr_coef_df)

##### 14. 绘制线性回归真实值与预测值对比图 #####
print("\n##### 绘制线性回归真实值与预测值对比图 #####")
fig = plt.figure(figsize=(10, 8))
sns.set_context("talk", font_scale=1.2)
plt.scatter(y_aug_test, lr_pred, alpha=0.6, color='indianred', edgecolors='k', linewidth=0.5)
plt.plot([y_aug_test.min(), y_aug_test.max()], [y_aug_test.min(), y_aug_test.max()], 'r--', lw=2)
plt.xlabel('真实值（最优模型预测）')
plt.ylabel('预测值')
plt.title('线性回归 - 预测值与真实值对比', fontsize=24, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('图片/线性回归预测值与真实值对比.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

##### 15. 保存最优模型信息 #####
print("\n##### 保存最优模型信息 #####")
best_model_info = pd.DataFrame({
    '最优模型名称': [best_model_name],
    '使用的特征数': [len(top_features)],
    '特征列表': [', '.join(top_features)]
})
best_model_info.to_csv('结果表/最优模型信息.csv', index=False, encoding='utf-8-sig')
print("最优模型信息已保存")

print("\n##### 线性回归模型训练完成 #####")
print(f"基于最优模型 {best_model_name} 的线性回归分析已完成")
