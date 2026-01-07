##### 导入库 #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
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

##### 4. 划分训练集和测试集 #####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

##### 5. 数据标准化 #####
print("\n##### 数据标准化 #####")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("数据标准化完成")

##### 6. 定义尾部MAE计算函数 #####
def calculate_tail_mae(y_true, y_pred):
    """
    计算尾部MAE（取周期产量的后30%计算的MAE）
    """
    threshold = np.percentile(y_true, 70)
    mask = y_true >= threshold
    return mean_absolute_error(y_true[mask], y_pred[mask])

##### 7. 神经网络模型训练与评估 #####
print("\n##### 神经网络模型训练与评估 #####")

# 参数空间设置
nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # 隐藏层的层数和每层的神经元数
    'activation': ['relu', 'tanh'],  # 激活函数
    'alpha': [0.001, 0.01],  # L2正则化参数
    'learning_rate_init': [0.01, 0.02]  # 初始学习率
}

nn_model = MLPRegressor(max_iter=100, random_state=1000, early_stopping=True)
nn_grid_search = GridSearchCV(
    nn_model, 
    nn_param_grid, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1,
    verbose=1
)
nn_grid_search.fit(X_train_scaled, y_train)

nn_best_model = nn_grid_search.best_estimator_
nn_best_params = nn_grid_search.best_params_
nn_pred = nn_best_model.predict(X_test_scaled)

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_tail_mae = calculate_tail_mae(y_test.values, nn_pred)

print(f"神经网络最优参数: {nn_best_params}")
print(f"神经网络 MAE: {nn_mae:.4f}")
print(f"神经网络 R²: {nn_r2:.4f}")
print(f"神经网络 RMSE: {nn_rmse:.4f}")
print(f"神经网络 尾部MAE: {nn_tail_mae:.4f}")

##### 8. 保存最优参数表 #####
print("\n##### 保存最优参数表 #####")
params_df = pd.DataFrame({
    '模型': ['神经网络'],
    '最优参数': [str(nn_best_params)]
})
params_df.to_csv('结果表/神经网络最优超参数.csv', index=False, encoding='utf-8-sig')
print("神经网络最优参数表已保存")

##### 9. 保存模型评估结果表 #####
print("\n##### 保存模型评估结果表 #####")
results_df = pd.DataFrame({
    '模型': ['神经网络'],
    'MAE': [nn_mae],
    'R²': [nn_r2],
    'RMSE': [nn_rmse],
    '尾部MAE': [nn_tail_mae]
})
results_df.to_csv('结果表/神经网络评估结果.csv', index=False, encoding='utf-8-sig')
print("神经网络评估结果表已保存")
print(results_df)

##### 10. 绘制真实值与预测值对比图 #####
print("\n##### 绘制真实值与预测值对比图 #####")
fig = plt.figure(figsize=(10, 8))
sns.set_context("talk", font_scale=1.2)
plt.scatter(y_test, nn_pred, alpha=0.6, color='mediumpurple', edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('神经网络 - 真实值与预测值对比', fontsize=24, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('图片/神经网络真实值与预测值对比.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

##### 11. 计算特征重要性（使用排列重要性） #####
print("\n##### 计算特征重要性 #####")
nn_perm_importance = permutation_importance(nn_best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
nn_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': nn_perm_importance.importances_mean
}).sort_values('重要性', ascending=False).head(10)

##### 12. 绘制特征重要性图 #####
print("\n##### 绘制特征重要性图 #####")
fig = plt.figure(figsize=(12, 8))
sns.set_context("talk", font_scale=1.2)
bars = plt.barh(nn_importance['特征'], nn_importance['重要性'], color=sns.color_palette("Blues_r", n_colors=10))
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
             ha='left', va='center', fontsize=12)
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('神经网络 - Top10特征重要性', fontsize=24, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('图片/神经网络Top10特征重要性.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

##### 13. 保存模型和标准化器 #####
print("\n##### 保存模型和标准化器 #####")
import pickle
with open('结果表/神经网络模型.pkl', 'wb') as f:
    pickle.dump(nn_best_model, f)
with open('结果表/标准化器.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("神经网络模型和标准化器已保存")

print("\n##### 神经网络模型训练完成 #####")
