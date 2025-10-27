# DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM NĂM 2025 (CORRECTED VERSION)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Cài đặt thêm cho time series
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    print("✅ ARIMA library imported successfully")
except ImportError:
    print("⚠️ Installing statsmodels for ARIMA...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'statsmodels'])
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose

print("=== DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM NĂM 2025 (CORRECTED) ===")

# 1. Load và phân tích dữ liệu thực tế
df = pd.read_csv("HealthyAir_HCMC.csv")
df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

print(f"\n📅 THÔNG TIN DỮ LIỆU THỰC TẾ:")
print(f"   Từ: {df['datetime'].min().strftime('%d/%m/%Y %H:%M')}")
print(f"   Đến: {df['datetime'].max().strftime('%d/%m/%Y %H:%M')}")
print(f"   Tổng cộng: {len(df):,} bản ghi")

# Phân bố theo năm và tháng
year_dist = df['datetime'].dt.year.value_counts().sort_index()
print(f"\n📊 PHÂN BỐ THEO NĂM:")
for year, count in year_dist.items():
    months = df[df['datetime'].dt.year == year]['datetime'].dt.month.nunique()
    print(f"   {year}: {count:,} bản ghi ({months} tháng)")

# Kiểm tra tháng cuối cùng của 2022
last_2022_data = df[df['datetime'].dt.year == 2022]['datetime'].dt.month.max()
print(f"   📍 Dữ liệu 2022 chỉ đến tháng {last_2022_data}")

# 2. Chuẩn bị dữ liệu cho time series
print(f"\n⚙️ CHUẨN BỊ DỮ LIỆU CHO FORECASTING...")

# Tạo dữ liệu hàng ngày thay vì hàng giờ để dự báo dễ hơn
df_daily = df.groupby(df['datetime'].dt.date).agg({
    'PM2.5': 'mean',
    'TSP': 'mean', 
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Station_No': 'count'  # số measurements per day
}).reset_index()

df_daily.rename(columns={'datetime': 'date', 'Station_No': 'measurements_count'}, inplace=True)
df_daily['date'] = pd.to_datetime(df_daily['date'])

print(f"   📈 Dữ liệu daily: {len(df_daily)} ngày")
print(f"   📅 Từ {df_daily['date'].min().strftime('%d/%m/%Y')} đến {df_daily['date'].max().strftime('%d/%m/%Y')}")

# 3. Phân tích xu hướng hiện tại
print(f"\n📊 PHÂN TÍCH XU HƯỚNG HIỆN TẠI:")

# Monthly averages
monthly_avg = df.groupby([df['datetime'].dt.year, df['datetime'].dt.month])['PM2.5'].mean()
print("   PM2.5 trung bình theo tháng (μg/m³):")

for (year, month), pm25 in monthly_avg.items():
    month_name = ['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month]
    print(f"     {month_name} {year}: {pm25:.1f}")

# Tính trend tổng thể
recent_months = monthly_avg.tail(6)  # 6 tháng gần nhất
early_months = monthly_avg.head(6)   # 6 tháng đầu
trend = recent_months.mean() - early_months.mean()
print(f"\n   🔍 XU HƯỚNG TỔNG QUAN: {trend:+.2f} μg/m³ ({'tăng' if trend > 0 else 'giảm'})")

# 4. Dự báo đơn giản dựa trên xu hướng
print(f"\n🔮 DỰ BÁO CHO NĂM 2025:")

# Method 1: Linear trend projection
from sklearn.linear_model import LinearRegression

# Chuẩn bị data cho linear regression
df_daily['days_since_start'] = (df_daily['date'] - df_daily['date'].min()).dt.days
X = df_daily[['days_since_start']].values
y = df_daily['PM2.5'].values

# Loại bỏ NaN values
mask = ~np.isnan(y)
X_clean = X[mask]
y_clean = y[mask]

# Fit model
lr_model = LinearRegression()
lr_model.fit(X_clean, y_clean)

print(f"   📈 Linear Trend Model:")
print(f"      Slope: {lr_model.coef_[0]:.4f} μg/m³ per day")
print(f"      R² Score: {lr_model.score(X_clean, y_clean):.3f}")

# Predict for 2025
start_2025 = datetime(2025, 1, 1)
end_2025 = datetime(2025, 12, 31)
days_to_2025_start = (start_2025 - df_daily['date'].min()).days
days_to_2025_end = (end_2025 - df_daily['date'].min()).days

pm25_2025_start = lr_model.predict([[days_to_2025_start]])[0]
pm25_2025_end = lr_model.predict([[days_to_2025_end]])[0]

print(f"\n   🎯 DỰ BÁO PM2.5 NĂM 2025:")
print(f"      Đầu năm 2025: {pm25_2025_start:.1f} μg/m³")
print(f"      Cuối năm 2025: {pm25_2025_end:.1f} μg/m³")
print(f"      Trung bình 2025: {(pm25_2025_start + pm25_2025_end)/2:.1f} μg/m³")

# AQI Level prediction
def get_aqi_level(pm25):
    if pm25 <= 12: return "Tốt"
    elif pm25 <= 35.4: return "Trung bình" 
    elif pm25 <= 55.4: return "Kém"
    else: return "Nguy hại"

avg_pm25_2025 = (pm25_2025_start + pm25_2025_end) / 2
aqi_2025 = get_aqi_level(avg_pm25_2025)
print(f"      AQI Level dự báo 2025: {aqi_2025}")

# 5. Seasonal forecast (theo tháng)
print(f"\n📅 DỰ BÁO THEO MÙA 2025:")

# Tính seasonal pattern từ dữ liệu hiện tại
seasonal_pattern = df.groupby(df['datetime'].dt.month)['PM2.5'].mean()

# Apply trend + seasonal
monthly_forecast_2025 = {}
for month in range(1, 13):
    if month in seasonal_pattern.index:
        base_seasonal = seasonal_pattern[month]
        # Adjust với trend
        days_to_month = (datetime(2025, month, 15) - df_daily['date'].min()).days
        trend_adjustment = lr_model.predict([[days_to_month]])[0] - lr_model.predict([[df_daily['days_since_start'].mean()]])[0]
        forecast_pm25 = base_seasonal + trend_adjustment
        monthly_forecast_2025[month] = forecast_pm25
        
        month_name = ['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month]
        aqi_level = get_aqi_level(forecast_pm25)
        print(f"   {month_name} 2025: {forecast_pm25:.1f} μg/m³ ({aqi_level})")

# 6. Visualization
print(f"\n📊 TẠO BIỂU ĐỒ DỰ BÁO...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Historical trend với linear projection
ax1 = axes[0, 0]
ax1.scatter(df_daily['date'], df_daily['PM2.5'], alpha=0.3, s=10, label='Dữ liệu hàng ngày')
monthly_data = df.groupby(df['datetime'].dt.to_period('M'))['PM2.5'].mean()
monthly_dates = [pd.to_datetime(str(period)) for period in monthly_data.index]
ax1.plot(monthly_dates, monthly_data.values, 'r-', linewidth=2, label='Trung bình tháng')

# Add 2025 projection
future_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='MS')
future_pm25 = [monthly_forecast_2025.get(d.month, 20) for d in future_dates]
ax1.plot(future_dates, future_pm25, 'b--', linewidth=3, label='Dự báo 2025', marker='o')

ax1.set_title('Xu hướng PM2.5 và Dự báo 2025')
ax1.set_ylabel('PM2.5 (μg/m³)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Seasonal Pattern
ax2 = axes[0, 1]
months = list(range(1, 13))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
historical_seasonal = [seasonal_pattern.get(m, np.nan) for m in months]
forecast_seasonal = [monthly_forecast_2025.get(m, np.nan) for m in months]

ax2.plot(months, historical_seasonal, 'g-o', label='Mẫu lịch sử', linewidth=2)
ax2.plot(months, forecast_seasonal, 'b-s', label='Dự báo 2025', linewidth=2)
ax2.set_title('So sánh xu hướng theo mùa')
ax2.set_xlabel('Tháng')
ax2.set_ylabel('PM2.5 (μg/m³)')
ax2.set_xticks(months)
ax2.set_xticklabels(month_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: AQI Level Distribution 2025
ax3 = axes[1, 0]
aqi_levels = [get_aqi_level(pm25) for pm25 in monthly_forecast_2025.values()]
aqi_counts = pd.Series(aqi_levels).value_counts()
colors = {'Good': 'green', 'Moderate': 'yellow', 'Unhealthy': 'orange', 'Hazardous': 'red'}
bar_colors = [colors.get(level, 'gray') for level in aqi_counts.index]
ax3.bar(aqi_counts.index, aqi_counts.values, color=bar_colors)
ax3.set_title('Phân bố mức AQI - Dự báo 2025')
ax3.set_ylabel('Số tháng')

# Plot 4: Confidence intervals
ax4 = axes[1, 1]
# Tính standard deviation để tạo confidence interval
historical_std = df_daily['PM2.5'].std()
forecast_values = list(monthly_forecast_2025.values())
upper_bound = [f + historical_std for f in forecast_values]
lower_bound = [max(0, f - historical_std) for f in forecast_values]

ax4.plot(months, forecast_values, 'b-', linewidth=2, label='Dự báo')
ax4.fill_between(months, lower_bound, upper_bound, alpha=0.3, label='Khoảng tin cậy')
ax4.set_title('Dự báo 2025 với khoảng tin cậy')
ax4.set_xlabel('Tháng')
ax4.set_ylabel('PM2.5 (μg/m³)')
ax4.set_xticks(months)
ax4.set_xticklabels(month_names)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecast_2025_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Summary Report
print(f"\n📋 BÁO CÁO TÓM TẮT DỰ BÁO 2025:")
print(f"="*50)
print(f"🔍 Dựa trên dữ liệu: {df['datetime'].min().strftime('%m/%Y')} - {df['datetime'].max().strftime('%m/%Y')}")
print(f"📊 Tổng mẫu phân tích: {len(df):,} measurements")
print(f"📈 Xu hướng tổng thể: {lr_model.coef_[0]*365:.1f} μg/m³ per year")
print(f"🎯 PM2.5 trung bình 2025: {avg_pm25_2025:.1f} μg/m³")
print(f"⚠️ AQI Level 2025: {aqi_2025}")

worst_month = max(monthly_forecast_2025, key=monthly_forecast_2025.get)
best_month = min(monthly_forecast_2025, key=monthly_forecast_2025.get)
worst_month_name = ['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][worst_month]
best_month_name = ['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][best_month]

print(f"📉 Tháng tốt nhất: {best_month_name} ({monthly_forecast_2025[best_month]:.1f} μg/m³)")
print(f"📈 Tháng xấu nhất: {worst_month_name} ({monthly_forecast_2025[worst_month]:.1f} μg/m³)")

print(f"\n💡 KHUYẾN NGHỊ:")
if avg_pm25_2025 > 25:
    print(f"   ⚠️ Chất lượng không khí dự báo ở mức cần cảnh báo")
    print(f"   🏃 Nên hạn chế hoạt động ngoài trời vào {worst_month_name}")
    print(f"   😷 Khuyến khích đeo khẩu trang khi ra đường")
else:
    print(f"   ✅ Chất lượng không khí dự báo ở mức chấp nhận được")

print(f"\n✅ FORECAST 2025 COMPLETED!")
print(f"📁 Charts saved as 'forecast_2025_corrected.png'")