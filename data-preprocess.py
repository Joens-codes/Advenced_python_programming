import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


def load_data(file_path): # EV 충전 패턴 데이터 전처리
    
    """
    지정한 컬럼(또는 전체)에서 결측치가 있는 행을 제거하는 함수

    Parameters:
    - file_path : str
        EV 충전 패턴 데이터가 저장된 CSV 파일 경로
    Returns:
    - EV_df : pandas.DataFrame
        전처리된 EV 충전 패턴 데이터프레임
    """
    
    df = pd.read_csv(file_path)
    
    columns_to_keep = [
    'State of Charge (Start %)',   # 충전 시작 시 배터리 잔량 (%)
    'State of Charge (End %)',     # 충전 후 배터리 잔량 (%) 
    'Energy Consumed (kWh)',       # 사용된 에너지 (킬로와트시)
    'Vehicle Model',               # 차량 모델명
    'Temperature (°C)',            # 온도 (섭씨)
    'User Type',                   # 사용자 유형
    'Charging Duration (hours)',   # 충전 시간 (시간 단위)
    'Charging Rate (kW)'           # 충전 속도 (킬로와트)
    ]
    
    # 지정한 컬럼만 추출하여 새로운 데이터프레임 생성
    EV_df = df[columns_to_keep].copy()
    
    # 컬럼명 변경
    EV_df.columns = [
        'soc_start', 'soc_end', 'energy_kwh', 'vehicle_model',
        'temperature', 'user_type', 'duration_hr', 'charging_rate'
    ]

    return EV_df

def check_missing_values(df):
    """
    데이터프레임의 컬럼별 결측치 개수와 비율을 출력하는 함수

    Parameters:
    - df : pandas.DataFrame
        결측치를 확인할 데이터프레임

    Returns:
    - missing_df : pandas.DataFrame
        각 컬럼별 결측치 개수와 비율을 담은 데이터프레임
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_ratio = (missing_count / total_rows) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Ratio (%)': missing_ratio.round(2)
    })

    return missing_df

def plot_missing_values(df, save_path):
    """
    결측치를 시각화하여 PNG 파일로 저장

    Parameters:
    - df : pandas.DataFrame  
        결측치를 시각화할 데이터프레임
    - save_path : str
        결측치 히트맵을 저장할 파일 경로
    """

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("결측치 히트맵")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def get_preprocessor(num_cols, cat_cols):  
    """
    수치형/범주형 열에 대한 전처리 파이프라인 생성
    
    Parameters:
    - num_cols : list  
        수치형 컬럼 리스트
    - cat_cols : list
        범주형 컬럼 리스트
   
    Returns:
    - preprocessor : sklearn.compose.ColumnTransformer
        수치형/범주형 전처리 파이프라인
    """
    num_imputer = SimpleImputer(strategy='mean')  # 평균으로 결측치 대체
    cat_imputer = SimpleImputer(strategy='most_frequent')  # 가장 빈도가 높은 값으로 결측치 대체
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', num_imputer),
                ('scaler', scaler)
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', encoder)
            ]), cat_cols)
        ]
    )


def preprocess_data(df, preprocessor):
    """전처리 파이프라인을 데이터에 적용"""
    return preprocessor.fit_transform(df)

def detect_outliers(data, df, contamination=0.05):
    """
    Isolation Forest를 사용해 이상치 탐지 및 제거

    Parameters:
    - data : pandas.DataFrame  
        이상치를 탐지할 데이터프레임
    - df : pandas.DataFrame
        원본 데이터프레임 (이상치 제거 후 반환용)
    - contamination : float, default=0.05
        이상치 비율 (0~1 사이의 값) 
    
    Returns:
    - df_clean : pandas.DataFrame  
        이상치가 제거된 데이터프레임 복사본
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(data)
    
    return df[outlier_preds == 1].reset_index(drop=True)

def create_features(df):
    """
    추가적인 파생 변수 생성
    Parameters:
    - df : pandas.DataFrame  
        원본 데이터프레임   
    
    Returns:
    - df : pandas.DataFrame  
        파생 변수가 추가된 데이터프레임
    """
    df['energy_per_hour'] = df['energy_kwh'] / df['duration_hr'].replace(0, np.nan)
    df['rate_efficiency'] = df['charging_rate'] / df['temperature'].replace(0, np.nan)
    df['soc_delta'] = df['soc_end'] - df['soc_start'].replace(0, np.nan)
    
    return df

def remove_missing_values(df, columns=None):
    """
    지정한 컬럼(또는 전체)에서 결측치가 있는 행을 제거하는 함수

    Parameters:
    - df : pandas.DataFrame  
        결측치를 제거할 데이터프레임  
    - columns : list or None, default=None  
        결측치를 확인할 컬럼 리스트. None이면 전체 컬럼에서 확인함.

    Returns:
    - df_clean : pandas.DataFrame  
        결측치가 제거된 데이터프레임 복사본
    """
    if columns:
        df_clean = df.dropna(subset=columns).copy()
    else:
        df_clean = df.dropna().copy()
    
    return df_clean

def plot_correlations(df, cols, save_path):
    """
    수치형 변수들 간의 상관관계 시각화
    Parameters:
    - df : pandas.DataFrame  
        상관관계를 시각화할 데이터프레임    
    - cols : list
        상관관계를 시각화할 수치형 변수 리스트
    - save_path : str
        상관관계 매트릭스를 저장할 파일 경로
    """
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("변수 간 상관관계 매트릭스")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    data_path = "ev_charging_patterns.csv"
    missing_plot_path = "missing_values_heatmap.png"
    correlation_plot_path = "correlation_matrix.png"
    
    print("데이터 읽고 필요한 항목 선택.")
    df = load_data(data_path)
   
    print("결측치 시각화 중...")
    plot_missing_values(df, missing_plot_path)
    
    print("전처리 파이프라인 구성")
    num_cols = ['soc_start', 'energy_kwh', 'temperature', 'duration_hr', 'charging_rate']
    cat_cols = ['vehicle_model', 'user_type']
    preprocessor = get_preprocessor(num_cols, cat_cols)
    
    print("전처리 적용")
    processed_array = preprocess_data(df, preprocessor)

    print("이상치 제거")
    df_cleaned = detect_outliers(processed_array, df)

    print("파생 변수 생성")
    df_engineered = create_features(df_cleaned)

    print("상관관계 시각화")
    plot_correlations(df_engineered, num_cols + ['energy_per_hour', 'rate_efficiency'], correlation_plot_path)

    print("전처리 완료!")
    print(df_engineered.head())


if __name__ == "__main__":
    main()