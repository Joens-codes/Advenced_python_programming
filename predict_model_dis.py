import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


# 1. 데이터 및 모델 정의와 전처리

# 학습에 사용된 전체 데이터 로드
df = pd.read_csv("ev_charging_patterns_cleaned.csv")
df["estimated_range_km"] = df["energy_kwh"] * 5

categorical_cols = ["vehicle_model", "user_type"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["estimated_range_km"]]

# 라벨 인코더 저장
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 수치형 스케일러 저장
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 카테고리 및 임베딩 차원
cat_dims = [df[col].nunique() for col in categorical_cols]
embed_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]


# 2. 모델 클래스 정의


class EVRangeModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_num_features):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim + 1, embed_dim)
            for cat_dim, embed_dim in zip(cat_dims, embed_dims)
        ])
        embed_out_dim = sum(embed_dims)
        self.batch_norm_num = nn.BatchNorm1d(num_num_features)

        self.fc1 = nn.Linear(embed_out_dim + num_num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(64, 1)

    def forward(self, x_cat, x_num):
        x_cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_embeds = torch.cat(x_cat_embeds, dim=1)
        x_num = self.batch_norm_num(x_num)
        x = torch.cat([x_cat_embeds, x_num], dim=1)
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        return self.out(x).squeeze(1)


# 3. 모델 불러오기


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EVRangeModel(cat_dims, embed_dims, len(numerical_cols))
model.load_state_dict(torch.load("ev_range_model.pth", map_location=device))
model.to(device)
model.eval()


# 4. 예측할 5명 사용자 데이터


user_data = pd.DataFrame([
    [29.37, 86.11, 60.71, "BMW i3", 27.95, "Commuter", 0.59, 36.38, 102.66, 1.30, 56.74],
    [10.11, 84.66, 12.33, "Hyundai Kona", 14.31, "Casual Driver", 3.13, 30.67, 3.93, 2.14, 74.55],
    [6.85, 69.91, 19.13, "Chevy Bolt", 21.00, "Commuter", 2.45, 27.51, 7.79, 1.31, 63.06],
    [54.26, 63.74, 19.63, "Hyundai Kona", -7.83, "Long-Distance Traveler", 2.02, 10.22, 9.72, -1.30, 9.48],
    [75.22, 71.98, 43.18, "Nissan Leaf", -5.27, "Long-Distance Traveler", 1.17, 14.33, 36.98, -2.72, -3.24]
], columns=[
    "soc_start", "soc_end", "energy_kwh", "vehicle_model", "temperature", "user_type",
    "duration_hr", "charging_rate", "energy_per_hour", "rate_efficiency", "soc_delta"
])

# 라벨 인코딩 적용
for col in categorical_cols:
    user_data[col] = label_encoders[col].transform(user_data[col])

# 스케일링 적용
user_data[numerical_cols] = scaler.transform(user_data[numerical_cols])

# 텐서 변환
X_cat_user = torch.tensor(user_data[categorical_cols].values, dtype=torch.long).to(device)
X_num_user = torch.tensor(user_data[numerical_cols].values, dtype=torch.float32).to(device)


# 5. 예측 및 시각화


with torch.no_grad():
    predictions = model(X_cat_user, X_num_user).cpu().numpy()

labels = [f"User {i+1}" for i in range(len(predictions))]

# 히스토그램 (각 사용자에 맞춘 라벨링)
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, predictions, color="skyblue", edgecolor="black")
plt.title("Predicted Driving Range per User")
plt.xlabel("User")
plt.ylabel("Estimated Range (km)")
plt.ylim(0, max(predictions) * 1.2)
for bar, value in zip(bars, predictions):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{value:.1f}", ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
