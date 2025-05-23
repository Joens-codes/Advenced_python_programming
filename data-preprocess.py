import pandas as pd


def preprocess_ev_data(file_path): # EV 충전 패턴 데이터 전처리
    
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
        'SoC_start', 'SoC_end', 'energy_kWh', 'vehicle_model',
        'temperature', 'user_type', 'charging_dura', 'charging_rate'
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

if __name__ == "__main__":
    df = preprocess_ev_data('ev_charging_patterns.csv')
    print(df.head())
    print(check_missing_values(df))