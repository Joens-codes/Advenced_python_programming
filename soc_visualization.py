import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

"""soc_visualization.py

Usage:
    python soc_visualization.py [path/to/ev_charging_patterns_cleaned.csv]

If no path is given, the script expects the cleaned CSV file to be in the same
folder as the script and named ``ev_charging_patterns_cleaned.csv``.
"""

# --------------------------------------------------
# 1. Load the cleaned dataset
# --------------------------------------------------
DEFAULT_CSV = Path(__file__).with_name("ev_charging_patterns_cleaned.csv")
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

try:
    cleaned_df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise SystemExit(f"❌  CSV 파일을 찾을 수 없습니다: {DATA_PATH}\n경로를 확인하거나 인자로 경로를 전달하세요.") from e

# --------------------------------------------------
# 2. Global plot style
# --------------------------------------------------
sns.set(style="whitegrid")

# --------------------------------------------------
# 3. Visualization functions
# --------------------------------------------------

def plot_soc_start_end(df: pd.DataFrame) -> None:
    """Scatter plot showing SOC start vs end grouped by vehicle model."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="soc_start",
        y="soc_end",
        hue="vehicle_model",
        palette="viridis",
        alpha=0.7,
    )
    plt.title("SOC Start vs SOC End by Vehicle Model")
    plt.xlabel("State of Charge at Start (%)")
    plt.ylabel("State of Charge at End (%)")
    plt.legend(title="Vehicle Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_soc_delta_hist(df: pd.DataFrame) -> None:
    """Histogram of SOC delta distribution with KDE overlay."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["soc_delta"], bins=30, kde=True, color="skyblue")
    plt.title("Distribution of SOC Change (Delta)")
    plt.xlabel("SOC Delta (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    plot_soc_start_end(cleaned_df)
    plot_soc_delta_hist(cleaned_df)
