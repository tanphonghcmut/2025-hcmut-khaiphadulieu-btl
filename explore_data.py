# Khám phá dữ liệu HealthyAir Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
print("=== KHÁM PHÁ DỮ LIỆU ===")
df = pd.read_csv("HealthyAir_HCMC.csv")

# Thông tin cơ bản
print("\n1. THÔNG TIN CƠ BẢN:")
print(f"Số dòng: {len(df):,}")
print(f"Số cột: {df.shape[1]}")
print(f"\nTên các cột:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n2. KIỂU DỮ LIỆU:")
print(df.dtypes)

print(f"\n3. 5 DÒNG ĐẦU:")
print(df.head())

print(f"\n4. THỐNG KÊ MÔ TẢ:")
print(df.describe())

print(f"\n5. MISSING VALUES:")
missing = df.isnull().sum()
print(missing[missing > 0])
if missing.sum() == 0:
    print("Không có missing values!")

print(f"\n6. GIÁ TRỊ UNIQUE:")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} giá trị unique")
    if col == 'Station_No':
        print(f"  Các station: {sorted(df[col].unique())}")

print(f"\n7. KHOẢNG THỜI GIAN:")
print(f"Từ: {df['date'].min()}")
print(f"Đến: {df['date'].max()}")

print(f"\n8. PHÂN BỐ THEO STATION:")
station_counts = df['Station_No'].value_counts().sort_index()
print(station_counts)

# Kiểm tra các giá trị bất thường
print(f"\n9. KIỂM TRA OUTLIERS (giá trị âm hoặc quá lớn):")
numeric_cols = ['TSP', 'PM2.5', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']
for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    negative_count = (df[col] < 0).sum()
    print(f"{col}: min={min_val:.2f}, max={max_val:.2f}, negative={negative_count}")