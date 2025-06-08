# improve_residual.py  ────────────────────────────────────────
# 0) 환경 준비 (TkAgg는 macOS 터미널·PyCharm에서 창이 안 뜰 때만 필요)
# -------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.nonparametric.smoothers_lowess import lowess   # 없으면: pip install statsmodels

CSV_PATH  = '/Users/limtae-kyu/Advenced_python_programming/ev_charging_patterns_cleaned.csv'
MODEL_PTH = '/Users/limtae-kyu/Advenced_python_programming/ev_range_model.pth'

# 1) 데이터 로드 & 전처리 --------------------------------------
df = pd.read_csv(CSV_PATH)
df['estimated_range_km'] = df['energy_kwh'] * 5  # 타깃 정의

cat_cols = ['vehicle_model', 'user_type']
num_cols = [c for c in df.columns if c not in cat_cols + ['estimated_range_km']]

for col in cat_cols:                               # 라벨 인코딩
    df[col] = LabelEncoder().fit_transform(df[col])

df[num_cols] = StandardScaler().fit_transform(df[num_cols])  # 스케일링

X_cat = df[cat_cols].values.astype(np.int64)
X_num = df[num_cols].values.astype(np.float32)
y_all = df['estimated_range_km'].values.astype(np.float32)

_, X_cat_val, _, X_num_val, _, y_val = train_test_split(
    X_cat, X_num, y_all, test_size=0.2, random_state=42
)

# 2) 모델 정의 (저장 당시와 동일한 변수명 사용) ---------------
class EVRangeModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_num_features):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(d + 1, e) for d, e in zip(cat_dims, embed_dims)]
        )
        self.batch_norm_num = nn.BatchNorm1d(num_num_features)   # ★ 이름 맞춤

        self.fc1 = nn.Linear(sum(embed_dims) + num_num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.2)

    def forward(self, x_cat, x_num):
        x_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], 1)
        x_num = self.batch_norm_num(x_num)
        x = torch.cat([x_cat, x_num], 1)
        x = self.dp1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dp2(torch.relu(self.bn2(self.fc2(x))))
        return self.out(x).squeeze(1)

cat_dims   = [df[c].nunique() for c in cat_cols]
embed_dims = [min(50, (d + 1)//2) for d in cat_dims]
model      = EVRangeModel(cat_dims, embed_dims, len(num_cols))

state_dict = torch.load(MODEL_PTH, map_location='cpu', weights_only=True)  # 경고 제거
model.load_state_dict(state_dict)
model.eval()

# 3) 검증셋 예측 ----------------------------------------------
with torch.no_grad():
    preds = model(torch.tensor(X_cat_val),
                  torch.tensor(X_num_val)).cpu().numpy()

residuals = y_val - preds
rmse      = np.sqrt(mean_squared_error(np.zeros_like(residuals), residuals))

# 4) 가독성 높은 잔차 플롯 -------------------------------------
def plot_residuals_simple(y_pred, residuals,
                          rmse_band=True, trend=True,
                          save_png=None):
    """Residuals vs Predictions  (Scatter + ±RMSE 밴드 + LOWESS)"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 산점도 (투명도·작은 검정 테두리)
    ax.scatter(y_pred, residuals,
               s=45, alpha=0.65,
               facecolors='tab:blue', edgecolors='k', linewidths=0.3)

    # ±RMSE 회색 밴드
    if rmse_band:
        ax.axhspan(-rmse, rmse, color='grey', alpha=0.15,
                   label=f'±1 RMSE ({rmse:.1f} km)')

    # 0 기준선
    ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # LOWESS 추세선
    if trend:
        low = lowess(residuals, y_pred, frac=0.2, return_sorted=True)
        ax.plot(low[:, 0], low[:, 1],
                color='black', linewidth=1.5, label='LOWESS trend')

    # 라벨·스타일
    ax.set_xlabel('Predicted Range (km)')
    ax.set_ylabel('Residual (km)')
    ax.set_title('Residuals vs Predictions')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper left')
    plt.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=150)
        print(f"Saved plot to: {save_png}")

    plt.show(block=True)

# 5) 실행 -------------------------------------------------------
if __name__ == '__main__':
    plot_residuals_simple(preds, residuals, save_png=None)