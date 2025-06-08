# improve_ev_range.py
# -------------------------------------------------------------
# 0) 환경 준비
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

CSV_PATH = '/Users/limtae-kyu/Advenced_python_programming/ev_charging_patterns_cleaned.csv'
MODEL_PTH = '/Users/limtae-kyu/Advenced_python_programming/ev_range_model.pth'

# ─────────────────────────────────────────────────────────────
# 1) 데이터 로드 & 동일 전처리
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['estimated_range_km'] = df['energy_kwh'] * 5

cat_cols = ['vehicle_model', 'user_type']
num_cols = [c for c in df.columns if c not in cat_cols + ['estimated_range_km']]

# 라벨 인코딩
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 스케일링
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 검증 셋 분리
X_cat = df[cat_cols].values.astype(np.int64)
X_num = df[num_cols].values.astype(np.float32)
y_true = df['estimated_range_km'].values.astype(np.float32)

_, X_cat_val, _, X_num_val, _, y_val = train_test_split(
    X_cat, X_num, y_true, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────
# 2) 모델 아키텍처 정의 (학습 때와 동일)
# ─────────────────────────────────────────────────────────────
class EVRangeModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_num_features):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d + 1, e) for d, e in zip(cat_dims, embed_dims)
        ])
        self.batch_norm_num = nn.BatchNorm1d(num_num_features)
        self.fc1 = nn.Linear(sum(embed_dims) + num_num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.2)

    def forward(self, x_cat, x_num):
        x_cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x_num = self.batch_norm_num(x_num)
        x = torch.cat([x_cat, x_num], dim=1)
        x = self.dp1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dp2(torch.relu(self.bn2(self.fc2(x))))
        return self.out(x).squeeze(1)

cat_dims = [df[c].nunique() for c in cat_cols]
embed_dims = [min(50, (d + 1)//2) for d in cat_dims]
model = EVRangeModel(cat_dims, embed_dims, len(num_cols))
model.load_state_dict(torch.load(MODEL_PTH, map_location='cpu'))
model.eval()

# ─────────────────────────────────────────────────────────────
# 3) 검증-셋 예측
# ─────────────────────────────────────────────────────────────
with torch.no_grad():
    preds = model(
        torch.tensor(X_cat_val), torch.tensor(X_num_val)
    ).numpy()

# ─────────────────────────────────────────────────────────────
# 4) 산점도 + 지표 시각화 (가독성 강화 버전)
#     · 핵심 데이터(1~99 퍼센타일)만 클리핑 해 축을 압축 → 패턴 강조
#     · 산점도 + Hexbin(로그 스케일) 이중 표현 → 밀집도 한눈에
#     · 성능 지표(R²·MAE·RMSE·N) 박스 삽입
# ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('TkAgg')          # macOS 터미널 / PyCharm 창 안 뜰 때만 유지

def plot_actual_vs_pred(y_true, y_pred, clip_pct=1, gridsize=45,
                        cmap='Blues', save_png=None):
    """
    발표용 'Actual vs Predicted' 시각화
    - clip_pct : 하위·상위 퍼센타일 (%) 잘라내 축 범위 압축(왜곡)
    - gridsize : Hexbin 해상도
    - cmap     : Hexbin 컬러맵
    - save_png : 파일명 지정 시 자동 저장
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # ① 축 범위 클리핑 ------------------------------------------------
    lo = np.percentile(np.concatenate([y_true, y_pred]), clip_pct)
    hi = np.percentile(np.concatenate([y_true, y_pred]), 100 - clip_pct)
    lims = [lo, hi]

    # ② 성능 지표 -----------------------------------------------------
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n    = len(y_true)

    # ③ 그래프 --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3-1) 산점도 (테두리 있는 투명 마커)
    ax.scatter(y_true, y_pred, s=25, alpha=0.5,
               facecolors='none', edgecolors='grey', linewidths=0.5)

    # 3-2) Hexbin (로그스케일 밀도)
    hb = ax.hexbin(
        y_true, y_pred,
        gridsize=gridsize, cmap=cmap,
        bins='log', extent=lims*2, alpha=0.9
    )
    cb = fig.colorbar(
        hb, ax=ax, pad=0.01,
        format=FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    cb.set_label("Counts (log)")

    # 3-3) 45° 기준선
    ax.plot(lims, lims, 'r--', lw=1, label='Ideal (y = x)')

    # ④ 지표 박스 -----------------------------------------------------
    stats = (
        f"N    = {n:,}\n"
        f"R²   = {r2:.3f}\n"
        f"MAE  = {mae:.1f} km\n"
        f"RMSE = {rmse:.1f} km"
    )
    ax.text(
        0.02, 0.98, stats,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35',
                  facecolor='white', alpha=0.85)
    )

    # ⑤ 레이블·서식 ---------------------------------------------------
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual Range (km)')
    ax.set_ylabel('Predicted Range (km)')
    ax.set_title('Actual vs Predicted EV Range (Density view)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper left')
    plt.tight_layout()

    # ⑥ 파일 저장(선택) ---------------------------------------------
    if save_png:
        fig.savefig(save_png, dpi=150)
        print(f"Saved plot to: {save_png}")

    plt.show(block=True)        # 창 닫을 때까지 스크립트 대기

plot_actual_vs_pred(y_val, preds)
