import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # 데이터 분할을 위해 추가
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
df = pd.read_csv("ev_charging_patterns_cleaned.csv")
# df["estimated_range_km"] = df["energy_kwh"] * 5  # ## 삭제: 불필요한 타겟 생성 라인

# --- 변수 정의 ---
# ## 변경: 타겟 변수를 'soc_delta'로 명확히 지정
TARGET_VARIABLE = 'soc_delta'
categorical_cols = ["vehicle_model", "user_type"]
# ## 변경: 입력 피처에서 타겟 변수인 'soc_delta' 제외
numerical_cols = [col for col in df.columns if col not in categorical_cols + [TARGET_VARIABLE]]

# 라벨 인코딩
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 수치형 스케일링
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- 데이터 분리 (X, y) ---
X_cat = df[categorical_cols].values.astype(np.int64)
X_num = df[numerical_cols].values.astype(np.float32)
y = df[TARGET_VARIABLE].values.astype(np.float32)  # ## 변경: 타겟 변수를 y로 지정

# --- 훈련/검증 데이터셋 분리 ---
# ## 추가: scikit-learn의 train_test_split을 사용하여 명시적으로 분리
X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42
)


# 2. PyTorch 데이터셋 클래스
class EVDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.from_numpy(X_cat)
        self.X_num = torch.from_numpy(X_num)
        self.y = torch.from_numpy(y).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


train_dataset = EVDataset(X_cat_train, X_num_train, y_train)
val_dataset = EVDataset(X_cat_val, X_num_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 3. 모델 클래스 정의
# (기존 모델 구조와 동일하므로 변경 없음)
class EVModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_numerical_features):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim, embed_dim in zip(cat_dims, embed_dims)
        ])
        self.bn_num = nn.BatchNorm1d(num_numerical_features)

        embed_sum = sum(e.embedding_dim for e in self.embeddings)
        total_features = embed_sum + num_numerical_features

        self.fc1 = nn.Linear(total_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(64, 1)

    def forward(self, x_cat, x_num):
        x_embed = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, 1)
        x_num = self.bn_num(x_num)

        x = torch.cat([x_embed, x_num], 1)

        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        return self.out(x)


cat_dims = [len(le.classes_) for le in label_encoders.values()]
embed_dims = [(dim, min(50, (dim + 1) // 2)) for dim in cat_dims]
model = EVModel(cat_dims, [e[1] for e in embed_dims], len(numerical_cols))

# 4. 훈련 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# (이후 훈련 루프, EarlyStopping 클래스, 평가 로직 등은 기존 코드와 동일하여 생략)
# ... (기존 코드의 훈련 및 평가 부분) ...

# ## 모델 저장 부분 변경 제안
# torch.save(model.state_dict(), "soc_delta_model.pth")
# print("Model saved as soc_delta_model.pth")