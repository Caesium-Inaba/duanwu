"""
测试同学提供的 DNN 模型：加载模型、在完整数据上预测、对比评估
"""
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── 路径 ──
MODEL_PATH = 'model/dnn_model.pth'
SCALER_PATH = 'model/scalers.pkl'
DATA_PATH = '原题/Cement_ESP_Data.csv'
OUT_DIR = 'img/model_test'
import os; os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载模型 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型架构：11 → 128 → 256 → 128 → 64 → 1 (4层隐层，ReLU)
model = torch.nn.Sequential(
    torch.nn.Linear(11, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# 同学的模型用了 BatchNorm 或 Dropout 层，导致 Linear 索引为 0,3,6,9,12
# 重新映射到纯 Sequential(Linear,ReLU)×4 的索引 0,2,4,6,8
old_keys = list(state_dict.keys())
for k in old_keys:
    new_k = k.removeprefix('network.')
    # 映射: '0'→'0', '3'→'2', '6'→'4', '9'→'6', '12'→'8'
    idx = int(new_k.split('.')[0])
    new_idx = idx // 3 * 2
    mapped_k = str(new_idx) + '.' + '.'.join(new_k.split('.')[1:])
    state_dict[mapped_k] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
print(f'模型加载成功，设备: {device}')
print(f'模型结构:\n{model}\n')

# ── 加载 scaler ──
with open(SCALER_PATH, 'rb') as f:
    scalers = pickle.load(f)
print(f'Scaler 加载成功，keys: {list(scalers.keys())}')
X_scaler = scalers['scaler_X']
y_scaler = scalers['scaler_y']

# ── 加载数据 ──
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
print(f'数据加载成功: {len(df)} 行')

# 尝试匹配同学使用的特征列
# DNN 通常用: Temp_C, C_in_gNm3, Q_Nm3h, U1~U4, T1~T4
feature_cols = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h',
                'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV',
                'T1_s', 'T2_s', 'T3_s', 'T4_s']

# 如果 scaler 训练时用了更多/更少特征，按 scaler 的 n_features 调整
n_features = X_scaler.n_features_in_ if hasattr(X_scaler, 'n_features_in_') else X_scaler.scale_.shape[0]
# scaler 没有存特征名，使用 11 特征列（与模型输入匹配）
feature_cols = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h',
                'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV',
                'T1_s', 'T2_s', 'T3_s', 'T4_s']
print(f'Scaler 特征数={n_features}，列: {feature_cols}')

X_raw = df[feature_cols].values
# 模型输出为 eta（除尘效率，0~1 归一化），用真实 eta 对比
eta_true = (df['C_in_gNm3'] - df['C_out_mgNm3'] / 1000) / df['C_in_gNm3']
y_raw = eta_true.values.reshape(-1, 1)

# ── 标准化 + 预测 ──
X_norm = X_scaler.transform(X_raw)
X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_norm = model(X_tensor).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))

# ── 评估 ──
y_true = y_raw.astype(np.float64)
print(f'pred NaN 数: {np.isnan(y_pred).sum()}, true NaN 数: {np.isnan(y_true).sum()}')
print(f'pred range: [{np.nanmin(y_pred):.4f}, {np.nanmax(y_pred):.4f}]')
print(f'true range: [{np.nanmin(y_true):.4f}, {np.nanmax(y_true):.4f}]')

mse = np.nanmean((y_pred - y_true) ** 2)
mae = np.nanmean(np.abs(y_pred - y_true))
ss_res = np.nansum((y_true - y_pred) ** 2)
ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f'\n=== 全量数据评估 ===')
print(f'MSE:  {mse:.6f}')
print(f'MAE:  {mae:.6f}')
print(f'R2:   {r2:.6f}')
print(f'预测范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]')
print(f'真实范围: [{y_true.min():.4f}, {y_true.max():.4f}]')

# ── 绘图1: 预测 vs 真实散点 ──
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(y_true, y_pred, s=2, alpha=0.5)
ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=1)
ax1.set_xlabel('True eta')
ax1.set_ylabel('Predicted eta')
ax1.set_title(f'DNN Prediction (R2={r2:.4f})')
ax1.grid(True)
fig1.savefig(f'{OUT_DIR}/pred_vs_true.png', dpi=150)
plt.close(fig1)

# ── 绘图2: 时间序列对比 ──
fig2, ax2 = plt.subplots(figsize=(12, 4))
t = df['timestamp']
ax2.plot(t, y_true, '.', ms=1, alpha=0.5, label='True')
ax2.plot(t, y_pred, '.', ms=1, alpha=0.5, label='Predicted')
ax2.set_xlabel('Time')
ax2.set_ylabel('eta')
ax2.set_title('DNN Prediction vs True (Time Series)')
ax2.legend(loc='best')
ax2.grid(True)
fig2.autofmt_xdate()
fig2.savefig(f'{OUT_DIR}/time_series.png', dpi=150)
plt.close(fig2)

# ── 绘图3: 残差分析 ──
residuals = y_true.flatten() - y_pred.flatten()
fig3, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(residuals, bins=50, edgecolor='none', alpha=0.7)
axes[0, 0].set_xlabel('Residual')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Residual Distribution')
axes[0, 0].grid(True)

axes[0, 1].scatter(y_pred, residuals, s=2, alpha=0.5)
axes[0, 1].axhline(0, color='r', linestyle='--', lw=1)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residual vs Predicted')
axes[0, 1].grid(True)

axes[1, 0].plot(t, residuals, '.', ms=1, alpha=0.5)
axes[1, 0].axhline(0, color='r', linestyle='--', lw=1)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Residual vs Time')
axes[1, 0].grid(True)
fig3.autofmt_xdate()

# 残差正态 QQ（手动画）
from scipy import stats as sp_stats
res_sorted = np.sort(residuals)
theoretical = sp_stats.norm.ppf((np.arange(len(res_sorted)) + 0.5) / len(res_sorted),
                                 loc=np.mean(residuals), scale=np.std(residuals))
axes[1, 1].scatter(theoretical, res_sorted, s=2, alpha=0.5)
axes[1, 1].plot(theoretical, theoretical, 'r--', lw=1)
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')
axes[1, 1].set_title('Q-Q Plot')
axes[1, 1].grid(True)

fig3.tight_layout()
fig3.savefig(f'{OUT_DIR}/residual_analysis.png', dpi=150)
plt.close(fig3)

print(f'\n图片已保存到 {OUT_DIR}/')
