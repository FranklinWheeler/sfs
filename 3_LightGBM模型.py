##### 导入库 #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
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

##### 5. 定义尾部MAE计算函数 #####
def calculate_tail_mae(y_true, y_pred):
    """
    计算尾部MAE（取周期产量的后30%计算的MAE）
    """
    threshold = np.percentile(y_true, 70)
    mask = y_true >= threshold
    return mean_absolute_error(y_true[mask], y_pred[mask])

##### 6. LightGBM模型训练与评估 #####
print("\n##### LightGBM模型训练与评估 #####")

# 参数空间设置
lgb_param_grid = {
    'n_estimators': [180, 200],  # boosting迭代次数
    'max_depth': [6, 12],  # 树的最大深度
    'learning_rate': [0.01, 0.05],  # 学习率
    'num_leaves': [31, 50],  # 一棵树上的叶子数
    'subsample': [0.7, 0.8]  # 训练每棵树时使用的样本比例
}

lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
lgb_grid_search = GridSearchCV(
    lgb_model, 
    lgb_param_grid, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1,
    verbose=1
)
lgb_grid_search.fit(X_train, y_train)

lgb_best_model = lgb_grid_search.best_estimator_
lgb_best_params = lgb_grid_search.best_params_
lgb_pred = lgb_best_model.predict(X_test)

lgb_mae = mean_absolute_error(y_test, lgb_pred)
lgb_r2 = r2_score(y_test, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
lgb_tail_mae = calculate_tail_mae(y_test.values, lgb_pred)

print(f"LightGBM最优参数: {lgb_best_params}")
print(f"LightGBM MAE: {lgb_mae:.4f}")
print(f"LightGBM R²: {lgb_r2:.4f}")
print(f"LightGBM RMSE: {lgb_rmse:.4f}")
print(f"LightGBM 尾部MAE: {lgb_tail_mae:.4f}")

##### 7. 保存最优参数表 #####
print("\n##### 保存最优参数表 #####")
params_df = pd.DataFrame({
    '模型': ['LightGBM'],
    '最优参数': [str(lgb_best_params)]
})
params_df.to_csv('结果表/LightGBM最优超参数.csv', index=False, encoding='utf-8-sig')
print("LightGBM最优参数表已保存")

##### 8. 保存模型评估结果表 #####
print("\n##### 保存模型评估结果表 #####")
results_df = pd.DataFrame({
    '模型': ['LightGBM'],
    'MAE': [lgb_mae],
    'R²': [lgb_r2],
    'RMSE': [lgb_rmse],
    '尾部MAE': [lgb_tail_mae]
})
results_df.to_csv('结果表/LightGBM评估结果.csv', index=False, encoding='utf-8-sig')
print("LightGBM评估结果表已保存")
print(results_df)

##### 9. 绘制真实值与预测值对比图 #####
print("\n##### 绘制真实值与预测值对比图 #####")
fig = plt.figure(figsize=(10, 8))
sns.set_context("talk", font_scale=1.2)
plt.scatter(y_test, lgb_pred, alpha=0.6, color='coral', edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('LightGBM - 真实值与预测值对比', fontsize=24, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('图片/LightGBM真实值与预测值对比.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

##### 10. 提取特征重要性并绘图 #####
print("\n##### 提取特征重要性并绘图 #####")
lgb_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': lgb_best_model.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

fig = plt.figure(figsize=(12, 8))
sns.set_context("talk", font_scale=1.2)
bars = plt.barh(lgb_importance['特征'], lgb_importance['重要性'], color=sns.color_palette("Blues_r", n_colors=10))
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.0f}', 
             ha='left', va='center', fontsize=12)
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('LightGBM - Top10特征重要性', fontsize=24, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('图片/LightGBMTop10特征重要性.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

##### 11. 保存模型 #####
print("\n##### 保存模型 #####")
import pickle
with open('结果表/LightGBM模型.pkl', 'wb') as f:
    pickle.dump(lgb_best_model, f)
print("LightGBM模型已保存")

print("\n##### LightGBM模型训练完成 #####")
