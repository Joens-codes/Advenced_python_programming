import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 1. 데이터 불러오기 ─────────────────────────────────────────────────
raw_data   = pd.read_csv('/Users/limtae-kyu/Advenced_python_programming/ev_charging_patterns.csv')
clean_data = pd.read_csv('/Users/limtae-kyu/Advenced_python_programming/ev_charging_patterns_cleaned.csv')

# ── 2. 기준(threshold) 및 집계 ────────────────────────────────────────
threshold      = 1000                       # 1000 이하 영역 생략
before_count   = len(raw_data)              # 전처리 전 행 수
after_count    = len(clean_data)            # 전처리 후 행 수
values_plot    = [before_count - threshold,
                  after_count  - threshold] # 실제로 그릴 값(Δ)

# ── 3. 그래프 그리기 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(range(2), values_plot,
              color=['steelblue', 'seagreen'])

# X축
ax.set_xticks([0, 1])
ax.set_xticklabels(['Before Cleaning', 'After Cleaning'])

# Y축(실제 값으로 레이블 변환)
yticks = np.arange(0, max(values_plot) + 101, 100)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks + threshold)
ax.set_ylabel('Number of Entries')

# 제목
ax.set_title('Data Entries Before vs. After Cleaning\n(axis broken below 1000)')

# ── 4. 축 끊김 표시(지그재그) ──────────────────────────────────────────
d = .015   # 지그재그 길이
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)          # 왼쪽 하단
ax.plot((1-d, 1+d), (-d, +d), **kwargs)        # 오른쪽 하단

# ── 5. 주석(선택) ───────────────────────────────────────────────────
ax.text(0.5, -0.08, 'Values below 1000 omitted',
        ha='center', va='top', transform=ax.transAxes)

plt.tight_layout()
plt.show()