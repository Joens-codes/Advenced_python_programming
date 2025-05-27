import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리
df = pd.read_csv("ev_charging_patterns_cleaned.csv")
df["estimated_range_km"] = df["energy_kwh"] * 5  # 타깃

categorical_cols = ["vehicle_model", "user_type"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["estimated_range_km"]]

# 라벨 인코딩
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 수치형 스케일링
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 데이터 분리
from sklearn.model_selection import train_test_split
X_cat = df[categorical_cols].values.astype(np.int64)
X_num = df[numerical_cols].values.astype(np.float32)
y = df["estimated_range_km"].values.astype(np.float32)

X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42
)

# Dataset 클래스 정의
class EVDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

train_dataset = EVDataset(X_cat_train, X_num_train, y_train)
val_dataset = EVDataset(X_cat_val, X_num_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class EVRangeModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, num_num_features):
        super().__init__()
        # 임베딩 레이어 리스트
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
        out = self.out(x)
        return out.squeeze(1)

cat_dims = [df[col].nunique() for col in categorical_cols]
embed_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = EVRangeModel(cat_dims, embed_dims, len(numerical_cols))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실함수, optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x_cat, x_num, y in dataloader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_cat, x_num)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(dataloader.dataset)

# 검증 함수
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    actuals = []
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, y)
            running_loss += loss.item() * y.size(0)
            preds.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    return running_loss / len(dataloader.dataset), preds, actuals

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# 학습 루프
epochs = 100
early_stopping = EarlyStopping(patience=10, verbose=True)
train_losses, val_losses = [], []

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_actuals = eval_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# 베스트 모델 불러오기
model.load_state_dict(early_stopping.best_model_state)

# 모델 저장
torch.save(model.state_dict(), "ev_range_model.pth")
print("Model saved as ev_range_model.pth")

# 최종 평가 
val_preds = val_preds.flatten()
mse = mean_squared_error(val_actuals, val_preds)
mae = mean_absolute_error(val_actuals, val_preds)
rmse = np.sqrt(mse)
r2 = r2_score(val_actuals, val_preds)

print(f"\n Evaluation Metrics:")
print(f"  MSE  : {mse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# 시각화 

# 1. 실제 vs 예측 산점도
plt.figure(figsize=(8, 6))
sns.scatterplot(x=val_actuals, y=val_preds, alpha=0.7)
plt.plot([val_actuals.min(), val_actuals.max()], [val_actuals.min(), val_actuals.max()], 'r--')
plt.xlabel("Actual Range (km)")
plt.ylabel("Predicted Range (km)")
plt.title("Actual vs Predicted EV Range")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 에러 분포 히스토그램
errors = val_actuals - val_preds
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=True, color="purple")
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error (km)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. 잔차 플롯 (예측값에 따른 에러)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=val_preds, y=errors, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Range (km)")
plt.ylabel("Residual (Error)")
plt.title("Residuals vs Predictions")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 예측값과 실제값 분포 비교
plt.figure(figsize=(8, 6))
sns.kdeplot(val_actuals, label="Actual", fill=True, linewidth=2)
sns.kdeplot(val_preds, label="Predicted", fill=True, linewidth=2)
plt.legend()
plt.title("Actual vs Predicted Range Distribution")
plt.xlabel("Range (km)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 학습 곡선 (손실 변화)
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
