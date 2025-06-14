import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# 1) 데이터 불러오기
# ======================================================================
df = pd.read_csv('ev_charging_patterns_cleaned.csv')   # 파일 경로를 맞춰 주세요

# ======================================================================
# 2) 파생 변수 생성: 충전 전후 SOC 차이
# ======================================================================
df['soc_delta'] = df['soc_end'] - df['soc_start']

# ======================================================================
# 3) 차종별 SOC 통계 요약
# ======================================================================
summary = (df
           .groupby('vehicle_model')
           .agg(avg_soc_start=('soc_start', 'mean'),
                avg_soc_end=('soc_end', 'mean'),
                avg_delta=('soc_delta', 'mean'),
                median_delta=('soc_delta', 'median'),
                sessions=('soc_delta', 'size'))
           .round(1)
           .sort_values(by='avg_delta', ascending=False))

# ---- (A) 콘솔에 바로 출력 ----
print('\n=== 차종별 SOC 요약 ===')
print(summary.to_string())

# ---- (B) CSV로 저장할 수도 있음 ----
summary.to_csv('soc_summary_by_vehicle.csv', encoding='utf-8-sig')
print('\n요약 결과를 "soc_summary_by_vehicle.csv"로 저장했습니다.')

# ======================================================================
# 4) SOC 산점도 시각화
# ======================================================================
plt.figure(figsize=(10, 6))

for model, group in df.groupby('vehicle_model'):
    plt.scatter(group['soc_start'],
                group['soc_end'],
                label=model,
                s=40,
                alpha=0.6,
                edgecolors='w',
                linewidths=0.5)

# 45도 기준선 추가
lims = (0, 100)
plt.plot(lims, lims, linestyle='--', linewidth=1, label='y = x')

# 축 범위 고정
plt.xlim(lims)
plt.ylim(lims)

# 레이블 및 제목
plt.xlabel('State of Charge at Start (%)', fontsize=12, labelpad=8)
plt.ylabel('State of Charge at End (%)', fontsize=12, labelpad=8)
plt.title('SOC Start vs SOC End by Vehicle Model', fontsize=14, pad=10)


# 그리드·범례
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(title='Vehicle Model', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()