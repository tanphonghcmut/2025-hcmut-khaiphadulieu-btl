# EDA Analysis và tạo Target Variable cho HealthyAir TP.HCM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Thiết lập matplotlib cho tiếng Việt
plt.rcParams['font.size'] = 12
plt.style.use('default')

print("=== EDA ANALYSIS - DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025 ===")

# 1. Đọc và xử lý dữ liệu cơ bản
df = pd.read_csv("HealthyAir_HCMC.csv")
print(f"📊 Dữ liệu gốc: {df.shape[0]:,} dòng, {df.shape[1]} cột")

# Chuyển đổi cột datetime
df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

# 2. TẠO BIẾN TARGET CHO CLASSIFICATION (Task 2)
print("\n🎯 TASK 2: CREATING AQI TARGET VARIABLE")

# Tạo AQI Level dựa trên PM2.5 (chuẩn WHO và EPA)
def create_aqi_level(pm25):
    if pm25 <= 12:
        return 'Good'
    elif pm25 <= 35.4:
        return 'Moderate'
    elif pm25 <= 55.4:
        return 'Unhealthy'
    else:
        return 'Hazardous'

df['AQI_Level'] = df['PM2.5'].apply(create_aqi_level)

# Thống kê AQI Level
print("📈 Phân bố AQI Level:")
aqi_counts = df['AQI_Level'].value_counts()
for level, count in aqi_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {level}: {count:,} ({percentage:.1f}%)")

# 3. PHÂN TÍCH THEO STATION (Đại diện cho các khu vực khác nhau)
print(f"\n🏢 PHÂN TÍCH 6 STATIONS (Đại diện khu vực khác nhau)")
print("📍 Station mapping (giả định dựa trên thực tế TP.HCM):")
station_info = {
    1: "Quận 1 (Trung tâm thương mại)",
    2: "Quận 7 (Khu công nghiệp)", 
    3: "Thủ Đức (Giao thông đông đúc)",
    4: "Quận 3 (Khu dân cư)",
    5: "Bình Thạnh (Hỗn hợp)",
    6: "Quận 9 (Ngoại thành - ít ô nhiễm)"
}

for station, desc in station_info.items():
    print(f"  Station {station}: {desc}")

# 4. EDA VISUALIZATION (Task 3)
print(f"\n📊 TASK 3: EXPLORATORY DATA ANALYSIS")

# Tạo figure với multiple subplots
fig = plt.figure(figsize=(20, 15))

# Plot 1: PM2.5 Distribution by Station
plt.subplot(3, 3, 1)
df.boxplot(column='PM2.5', by='Station_No', ax=plt.gca())
plt.title('Phân bố PM2.5 theo Station', fontsize=12)
plt.xlabel('Số Station')
plt.ylabel('PM2.5 (μg/m³)')

# Plot 2: AQI Level Distribution
plt.subplot(3, 3, 2)
aqi_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
plt.title('Phân bố AQI Level')
plt.xlabel('Mức AQI')
plt.ylabel('Số lượng')
plt.xticks(rotation=45)

# Plot 3: Time Series PM2.5
plt.subplot(3, 3, 3)
monthly_pm25 = df.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
monthly_pm25['date'] = pd.to_datetime(monthly_pm25[['year', 'month']].assign(day=1))
plt.plot(monthly_pm25['date'], monthly_pm25['PM2.5'], marker='o')
plt.title('Xu hướng PM2.5 theo thời gian')
plt.xlabel('Thời gian')
plt.ylabel('PM2.5 (μg/m³)')
plt.xticks(rotation=45)

# Plot 4: Correlation Matrix
plt.subplot(3, 3, 4)
numeric_cols = ['PM2.5', 'TSP', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Ma trận tương quan')

# Plot 5: Temperature vs PM2.5
plt.subplot(3, 3, 5)
plt.scatter(df['Temperature'], df['PM2.5'], alpha=0.3, s=1)
plt.xlabel('Nhiệt độ (°C)')
plt.ylabel('PM2.5 (μg/m³)')
plt.title('Nhiệt độ vs PM2.5')

# Plot 6: Hourly Pattern
plt.subplot(3, 3, 6)
hourly_pm25 = df.groupby('hour')['PM2.5'].mean()
plt.plot(hourly_pm25.index, hourly_pm25.values, marker='o')
plt.title('PM2.5 trung bình theo giờ')
plt.xlabel('Giờ trong ngày')
plt.ylabel('PM2.5 (μg/m³)')
plt.grid(True, alpha=0.3)

# Plot 7: Monthly Pattern
plt.subplot(3, 3, 7)
monthly_pattern = df.groupby('month')['PM2.5'].mean()
plt.bar(monthly_pattern.index, monthly_pattern.values, 
        color=['blue' if x in [12,1,2] else 'red' if x in [3,4,5] else 
               'green' if x in [6,7,8] else 'orange' for x in monthly_pattern.index])
plt.title('PM2.5 trung bình theo tháng')
plt.xlabel('Tháng')
plt.ylabel('PM2.5 (μg/m³)')

# Plot 8: Station Comparison
plt.subplot(3, 3, 8)
station_pm25 = df.groupby('Station_No')['PM2.5'].mean()
bars = plt.bar(station_pm25.index, station_pm25.values, 
               color=['red' if x > 25 else 'yellow' if x > 15 else 'green' 
                      for x in station_pm25.values])
plt.title('PM2.5 trung bình theo Station')
plt.xlabel('Số Station')
plt.ylabel('PM2.5 (μg/m³)')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{station_pm25.values[i]:.1f}', ha='center')

# Plot 9: AQI Level by Station
plt.subplot(3, 3, 9)
aqi_station = pd.crosstab(df['Station_No'], df['AQI_Level'], normalize='index') * 100
aqi_station.plot(kind='bar', stacked=True, 
                color=['green', 'yellow', 'orange', 'red'])
plt.title('Phân bố AQI Level theo Station (%)')
plt.xlabel('Số Station')
plt.ylabel('Tỷ lệ (%)')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. THỐNG KÊ QUAN TRỌNG
print(f"\n📊 THỐNG KÊ CHỦ YẾU:")
print(f"🕐 Thời gian dữ liệu: {df['datetime'].min()} đến {df['datetime'].max()}")
print(f"📍 Số stations: {df['Station_No'].nunique()}")
print(f"⚠️ Station ô nhiễm nhất: Station {station_pm25.idxmax()} (PM2.5: {station_pm25.max():.1f})")
print(f"✅ Station sạch nhất: Station {station_pm25.idxmin()} (PM2.5: {station_pm25.min():.1f})")

# Thống kê missing values sau khi có datetime
print(f"\n🔍 MISSING VALUES STATUS:")
missing_stats = df.isnull().sum()
for col, missing in missing_stats.items():
    if missing > 0:
        percentage = (missing / len(df)) * 100
        print(f"  {col}: {missing:,} ({percentage:.1f}%)")

# Xuất dữ liệu đã xử lý
df_clean = df.dropna(subset=['PM2.5', 'TSP', 'Temperature', 'Humidity'])
df_clean.to_csv('cleaned_data.csv', index=False)
print(f"\n💾 Saved cleaned data: {df_clean.shape[0]:,} rows to 'cleaned_data.csv'")

print(f"\n✅ Task 2 & 3 COMPLETED!")
print(f"📊 Ready for Task 4: Machine Learning Models")