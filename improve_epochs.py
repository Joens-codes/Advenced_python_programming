# improve_epochs.py ────────────────────────────────────────────
# 0. 환경 준비 -------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

CSV_PATH  = '/Users/limtae-kyu/Advenced_python_programming/ev_charging_patterns_cleaned.csv'
MODEL_PTH = '/Users/limtae-kyu/Advenced_python_programming/ev_range_model.pth'

# 1. 데이터 로드 & 전처리 --------------------------------------
df = pd.read_csv(CSV_PATH)
df['estimated_range_km'] = df['energy_kwh'] * 5

cat_cols = ['vehicle_model', 'user_type']
num_cols = [c for c in df.columns if c not in cat_cols + ['estimated_range_km']]

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X_cat = df[cat_cols].values.astype(np.int64)
X_num = df[num_cols].values.astype(np.float32)
y_all = df['estimated_range_km'].values.astype(np.float32)

X_cat_tr, X_cat_val, X_num_tr, X_num_val, y_tr, y_val = train_test_split(
    X_cat, X_num, y_all, test_size=0.2, random_state=42
)

class EVDataset(Dataset):
    def __init__(self, xc, xn, y):
        self.xc = torch.tensor(xc, dtype=torch.long)
        self.xn = torch.tensor(xn, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.xc[i], self.xn[i], self.y[i]

train_loader = DataLoader(EVDataset(X_cat_tr, X_num_tr, y_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(EVDataset(X_cat_val, X_num_val, y_val),
                          batch_size=32)

# 2. 모델 아키텍처 (저장 당시와 동일) --------------------------
class EVRangeModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_num_features):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(d + 1, e) for d, e in zip(cat_dims, embed_dims)]
        )
        self.batch_norm_num = nn.BatchNorm1d(num_num_features)
        self.fc1 = nn.Linear(sum(embed_dims) + num_num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.dp1 = nn.Dropout(0.3); self.dp2 = nn.Dropout(0.2)
    def forward(self, x_cat, x_num):
        x_cat = torch.cat([emb(x_cat[:, i])
                           for i, emb in enumerate(self.embeddings)], 1)
        x_num = self.batch_norm_num(x_num)
        x = torch.cat([x_cat, x_num], 1)
        x = self.dp1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dp2(torch.relu(self.bn2(self.fc2(x))))
        return self.out(x).squeeze(1)

cat_dims   = [df[c].nunique() for c in cat_cols]
embed_dims = [min(50, (d + 1)//2) for d in cat_dims]
model      = EVRangeModel(cat_dims, embed_dims, len(num_cols))

# 3. 학습 세팅 --------------------------------------------------
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
epochs     = 90
train_loss_hist, val_loss_hist = [], []

for epoch in range(epochs):
    # --- Train ---
    model.train(); running = 0
    for xc, xn, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xc, xn), y)
        loss.backward(); optimizer.step()
        running += loss.item() * len(y)
    train_loss = running / len(train_loader.dataset)
    # --- Val ---
    model.eval(); running = 0
    with torch.no_grad():
        for xc, xn, y in val_loader:
            running += criterion(model(xc, xn), y).item() * len(y)
    val_loss = running / len(val_loader.dataset)
    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    print(f'Epoch {epoch+1:>3}/{epochs} | '
          f'Train: {train_loss:,.0f} | Val: {val_loss:,.0f}')

# 4. 최종 가중치 저장(선택) ------------------------------------
torch.save(model.state_dict(), MODEL_PTH)
print('Model weights updated →', MODEL_PTH)

# 5. 손실 곡선 시각화 -----------------------------------------
def plot_loss_curve(tr, vl, smooth_window=5,
                    log_y=False, save_png=None):
    ep = np.arange(1, len(tr)+1)
    tr, vl = np.array(tr), np.array(vl)

    # 이동 평균 스무딩
    if len(tr) >= smooth_window*2:
        tr_s = uniform_filter1d(tr, size=smooth_window, mode='nearest')
        vl_s = uniform_filter1d(vl, size=smooth_window, mode='nearest')
    else:
        tr_s, vl_s = tr, vl

    best_ep = int(vl.argmin()+1); best_val = vl.min()

    plt.figure(figsize=(9,5))
    plt.plot(ep, tr_s, label='Train (smooth)', lw=2, color='tab:blue')
    plt.plot(ep, vl_s, label='Val (smooth)',   lw=2, color='tab:orange')
    plt.plot(ep, tr, alpha=.3, ls='--', color='tab:blue')
    plt.plot(ep, vl, alpha=.3, ls='--', color='tab:orange')
    plt.scatter(best_ep, best_val, color='red', zorder=5)
    plt.text(best_ep, best_val,
             f'  Best Val\n  Epoch {best_ep}',
             va='bottom', ha='left', color='red', fontsize=9)

    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss per Epoch')
    if log_y: plt.yscale('log'); plt.ylabel('MSE Loss (log)')
    plt.grid(True, ls='--', lw=.5, alpha=.6)
    plt.legend(); plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=150)
        print('Saved plot →', save_png)
    plt.show(block=True)

# 호출
plot_loss_curve(train_loss_hist, val_loss_hist,
                smooth_window=5, log_y=False,
                save_png=None)