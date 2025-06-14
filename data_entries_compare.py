import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
raw_data = pd.read_csv('ev_charging_patterns.csv')
clean_data = pd.read_csv('ev_charging_patterns_cleaned.csv')

# 데이터 개수 확인
num_entries_before = len(raw_data)
num_entries_after = len(clean_data)

# 시각화
plt.figure(figsize=(8, 6))
plt.bar(['Before Cleaning', 'After Cleaning'], [num_entries_before, num_entries_after], color=['skyblue', 'lightgreen'])
plt.ylabel('Number of Entries')
plt.title('Comparison of Data Entries Before and After Cleaning')
plt.show()
