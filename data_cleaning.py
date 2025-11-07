# File làm sạch dữ liệu HealthyAir Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=== LÀM SẠCH DỮ LIỆU ===")

df = pd.read_csv("HealthyAir_HCMC.csv")
print(f"Dữ liệu gốc: {df.shape[0]:,} dòng, {df.shape[1]} cột")

print("\n1. XỬ LÝ THỜI GIAN:")
df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
print(f"Số dòng có lỗi datetime: {df['datetime'].isna().sum()}")


df_valid_time = df[df['datetime'].notna()]
print(f"Thời gian từ: {df_valid_time['datetime'].min()}")
print(f"Thời gian đến: {df_valid_time['datetime'].max()}")


print("\n2. PHÂN TÍCH MISSING VALUES:")
missing_stats = df.isnull().sum()
missing_percent = (missing_stats / len(df)) * 100

for col in df.columns:
    if missing_stats[col] > 0:
        print(f"{col}: {missing_stats[col]:,} ({missing_percent[col]:.2f}%)")


print("\n3. PHÂN TÍCH OUTLIERS:")
numeric_cols = ['TSP', 'PM2.5', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']

outlier_stats = {}
for col in numeric_cols:
    if col in df.columns:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_stats[col] = {
            'count': len(outliers),
            'percent': (len(outliers) / len(data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'max_value': data.max()
        }
        
        print(f"{col}:")
        print(f"  Outliers: {len(outliers)} ({(len(outliers)/len(data)*100):.2f}%)")
        print(f"  Khoảng bình thường: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Giá trị max: {data.max():.2f}")


print("\n4. LÀM SẠCH DỮ LIỆU:")


df_clean = df.copy()

print("4.1. Xử lý Missing Values:")

df_clean = df_clean[df_clean['datetime'].notna()]
print(f"Sau khi loại bỏ datetime không hợp lệ: {len(df_clean):,} dòng")


for col in ['O3', 'SO2']:
    if missing_percent[col] > 20:
        print(f"  {col}: {missing_percent[col]:.2f}% missing - sẽ impute bằng median theo station")
        df_clean[col] = df_clean.groupby('Station_No')[col].transform(
            lambda x: x.fillna(x.median())
        )


for col in ['CO', 'NO2', 'TSP', 'Temperature', 'Humidity']:
    if col in df_clean.columns and df_clean[col].isna().sum() > 0:
        df_clean[col] = df_clean.groupby('Station_No')[col].transform(
            lambda x: x.fillna(x.median())
        )

print("Sau khi xử lý missing values:")
print(df_clean.isnull().sum())


print("\n4.2. Xử lý Outliers:")

def remove_outliers_iqr(data, column, multiplier=3.0):
    """Loại bỏ outliers sử dụng IQR method với multiplier cao hơn"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    before_count = len(data)
    data_filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    after_count = len(data_filtered)
    
    print(f"  {column}: Loại bỏ {before_count - after_count} outliers ({((before_count - after_count)/before_count*100):.2f}%)")
    return data_filtered

outlier_cols = ['CO', 'TSP', 'PM2.5', 'NO2'] 
for col in outlier_cols:
    if col in df_clean.columns:
        df_clean = remove_outliers_iqr(df_clean, col, multiplier=3.0)

print(f"\nDữ liệu sau khi làm sạch: {len(df_clean):,} dòng ({(len(df_clean)/len(df)*100):.2f}% dữ liệu gốc)")


print("\n5. KIỂM TRA DỮ LIỆU SAU LÀM SẠCH:")
print(f"Kích thước: {df_clean.shape}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
print(f"Khoảng thời gian: {df_clean['datetime'].min()} đến {df_clean['datetime'].max()}")

print("\nThống kê mô tả sau làm sạch:")
print(df_clean[numeric_cols].describe())

df_clean.to_csv("HealthyAir_HCMC_cleaned.csv", index=False)
print(f"\n✅ Đã lưu dữ liệu đã làm sạch vào: HealthyAir_HCMC_cleaned.csv")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df['PM2.5'].dropna(), bins=50, alpha=0.7, label='Trước làm sạch', color='red')
plt.hist(df_clean['PM2.5'], bins=50, alpha=0.7, label='Sau làm sạch', color='blue')
plt.title('Phân bố PM2.5')
plt.xlabel('PM2.5 (μg/m³)')
plt.ylabel('Tần suất')
plt.legend()


plt.subplot(2, 3, 2)
plt.hist(df['CO'].dropna(), bins=50, alpha=0.7, label='Trước làm sạch', color='red')
plt.hist(df_clean['CO'], bins=50, alpha=0.7, label='Sau làm sạch', color='blue')
plt.title('Phân bố CO')
plt.xlabel('CO (μg/m³)')
plt.ylabel('Tần suất')
plt.legend()


plt.subplot(2, 3, 3)
plt.hist(df['TSP'].dropna(), bins=50, alpha=0.7, label='Trước làm sạch', color='red')
plt.hist(df_clean['TSP'], bins=50, alpha=0.7, label='Sau làm sạch', color='blue')
plt.title('Phân bố TSP')
plt.xlabel('TSP (μg/m³)')
plt.ylabel('Tần suất')
plt.legend()


plt.subplot(2, 3, 4)
data_to_plot = [df_clean['PM2.5'], df_clean['CO']/100, df_clean['NO2']] 
plt.boxplot(data_to_plot, labels=['PM2.5', 'CO/100', 'NO2'])
plt.title('Boxplot các chất ô nhiễm (sau làm sạch)')
plt.ylabel('Nồng độ')


plt.subplot(2, 3, 5)
missing_before = df.isnull().sum()
missing_after = df_clean.isnull().sum()
cols_to_show = ['O3', 'CO', 'NO2', 'SO2', 'TSP']
x_pos = np.arange(len(cols_to_show))
plt.bar(x_pos - 0.2, [missing_before[col] for col in cols_to_show], 0.4, label='Trước', alpha=0.7, color='red')
plt.bar(x_pos + 0.2, [missing_after[col] for col in cols_to_show], 0.4, label='Sau', alpha=0.7, color='blue')
plt.xlabel('Cột dữ liệu')
plt.ylabel('Số missing values')
plt.title('So sánh Missing Values')
plt.xticks(x_pos, cols_to_show, rotation=45)
plt.legend()


plt.subplot(2, 3, 6)
sizes = [len(df), len(df_clean)]
labels = ['Dữ liệu gốc', 'Sau làm sạch']
colors = ['red', 'blue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Tỷ lệ dữ liệu còn lại')

plt.tight_layout()
plt.savefig('data_cleaning_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Task 1 hoàn thành: Dữ liệu đã được phân tích và làm sạch!")