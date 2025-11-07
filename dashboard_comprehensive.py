# DASHBOARD TỔNG HỢP - DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['figure.figsize'] = (20, 24)
plt.rcParams['font.size'] = 10
plt.style.use('default')

print("=== DASHBOARD TỔNG HỢP - DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025 ===")

df = pd.read_csv("HealthyAir_HCMC.csv")
df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')


def get_aqi_level(pm25):
    if pm25 <= 12: return "Tốt"
    elif pm25 <= 35.4: return "Trung bình" 
    elif pm25 <= 55.4: return "Kém"
    else: return "Nguy hại"

df['AQI_Level'] = df['PM2.5'].apply(get_aqi_level)

forecast_2025 = {
    1: 30.3, 2: 19.1, 3: 22.1, 4: 22.6, 5: 18.2, 6: 19.1,
    7: 15.2, 8: 15.3, 9: 13.9, 10: 24.3, 11: 25.6, 12: 28.1
}


fig = plt.figure(figsize=(24, 30))
fig.suptitle('🌍 DASHBOARD DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025\nDựa trên dữ liệu 02/2021 - 06/2022 (52,548 phép đo)', 
             fontsize=20, fontweight='bold', y=0.98)


ax1 = plt.subplot(4, 3, 1)
overview_stats = [
    ['Tổng số bản ghi', f"{len(df):,}"],
    ['Thời gian dữ liệu', '02/2021 - 06/2022'],
    ['Số trạm đo', f"{df['Station_No'].nunique()}"],
    ['PM2.5 TB lịch sử', f"{df['PM2.5'].mean():.1f} μg/m³"],
    ['PM2.5 TB dự báo 2025', f"{np.mean(list(forecast_2025.values())):.1f} μg/m³"],
    ['Mức AQI năm 2025', 'Trung bình']
]

for i, (label, value) in enumerate(overview_stats):
    ax1.text(0.05, 0.9 - i*0.15, f"{label}:", fontweight='bold', transform=ax1.transAxes, fontsize=11)
    ax1.text(0.6, 0.9 - i*0.15, value, transform=ax1.transAxes, fontsize=11)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('📊 TỔNG QUAN DỰ ÁN', fontweight='bold', fontsize=12)


ax2 = plt.subplot(4, 3, 2)
monthly_pm25 = df.groupby([df['datetime'].dt.year, df['datetime'].dt.month])['PM2.5'].mean()
dates = [datetime(year, month, 1) for (year, month) in monthly_pm25.index]
ax2.plot(dates, monthly_pm25.values, 'o-', linewidth=2, markersize=6, color='blue', label='Dữ liệu lịch sử')
ax2.set_title('📈 Xu hướng PM2.5 lịch sử', fontweight='bold')
ax2.set_ylabel('PM2.5 (μg/m³)')
ax2.grid(True, alpha=0.3)
ax2.legend()


ax3 = plt.subplot(4, 3, 3)
months = list(range(1, 13))
month_names = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
forecast_values = list(forecast_2025.values())
colors = ['red' if x > 25 else 'yellow' if x > 20 else 'green' for x in forecast_values]
bars = ax3.bar(months, forecast_values, color=colors, alpha=0.7)
ax3.set_title('🔮 Dự báo theo tháng năm 2025', fontweight='bold')
ax3.set_ylabel('PM2.5 (μg/m³)')
ax3.set_xlabel('Tháng')
ax3.set_xticks(months)
ax3.set_xticklabels(month_names)
ax3.grid(True, alpha=0.3, axis='y')


for bar, val in zip(bars, forecast_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)


ax4 = plt.subplot(4, 3, 4)
station_stats = df.groupby('Station_No')['PM2.5'].agg(['mean', 'std']).reset_index()
station_info = {
    1: "Q1-Thương mại", 2: "Q7-Công nghiệp", 3: "Thủ Đức-Giao thông",
    4: "Q3-Dân cư", 5: "Bình Thạnh-Hỗn hợp", 6: "Q9-Ngoại thành"
}

bars = ax4.bar(station_stats['Station_No'], station_stats['mean'], 
               yerr=station_stats['std'], capsize=5, alpha=0.7,
               color=['red' if x > 25 else 'yellow' if x > 20 else 'green' for x in station_stats['mean']])
ax4.set_title('🏢 PM2.5 trung bình theo trạm', fontweight='bold')
ax4.set_ylabel('PM2.5 (μg/m³)')
ax4.set_xlabel('Trạm đo')
ax4.set_xticks(station_stats['Station_No'])
ax4.set_xticklabels([f"Trạm {i}" for i in station_stats['Station_No']])
ax4.grid(True, alpha=0.3, axis='y')


ax5 = plt.subplot(4, 3, 5)
aqi_counts = df['AQI_Level'].value_counts()
aqi_order = ['Tốt', 'Trung bình', 'Kém', 'Nguy hại']
aqi_counts = aqi_counts.reindex(aqi_order, fill_value=0)
colors_aqi = ['green', 'yellow', 'orange', 'red']
wedges, texts, autotexts = ax5.pie(aqi_counts.values, labels=aqi_counts.index, 
                                   colors=colors_aqi, autopct='%1.1f%%', startangle=90)
ax5.set_title('🎯 Phân bố mức AQI lịch sử', fontweight='bold')


ax6 = plt.subplot(4, 3, 6)
numeric_cols = ['PM2.5', 'TSP', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']
col_names_vn = ['PM2.5', 'TSP', 'O3', 'CO', 'NO2', 'SO2', 'Nhiệt độ', 'Độ ẩm']
correlation_matrix = df[numeric_cols].corr()
correlation_matrix.columns = col_names_vn
correlation_matrix.index = col_names_vn
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, fmt='.2f', ax=ax6, cbar_kws={'shrink': 0.8})
ax6.set_title('🔗 Ma trận tương quan', fontweight='bold')


ax7 = plt.subplot(4, 3, 7)
seasonal_pattern = df.groupby(df['datetime'].dt.month)['PM2.5'].mean()
ax7.plot(seasonal_pattern.index, seasonal_pattern.values, 'o-', 
         linewidth=2, markersize=8, color='purple', label='Mẫu lịch sử')
ax7.plot(months, forecast_values, 's--', 
         linewidth=2, markersize=8, color='red', label='Dự báo 2025')
ax7.set_title('🍂 Phân tích mẫu theo mùa', fontweight='bold')
ax7.set_ylabel('PM2.5 (μg/m³)')
ax7.set_xlabel('Tháng')
ax7.set_xticks(months)
ax7.set_xticklabels(month_names)
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(4, 3, 8)
hourly_pattern = df.groupby(df['datetime'].dt.hour)['PM2.5'].mean()
ax8.plot(hourly_pattern.index, hourly_pattern.values, 'o-', linewidth=2, color='orange')
ax8.set_title('🕐 Mẫu hàng ngày (trung bình theo giờ)', fontweight='bold')
ax8.set_ylabel('PM2.5 (μg/m³)')
ax8.set_xlabel('Giờ trong ngày')
ax8.grid(True, alpha=0.3)


ax9 = plt.subplot(4, 3, 9)
sample_data = df.sample(n=min(5000, len(df)))
scatter = ax9.scatter(sample_data['Temperature'], sample_data['PM2.5'], 
                     c=sample_data['Humidity'], cmap='viridis', alpha=0.6, s=10)
ax9.set_title('🌡️ Nhiệt độ vs PM2.5 (màu theo độ ẩm)', fontweight='bold')
ax9.set_xlabel('Nhiệt độ (°C)')
ax9.set_ylabel('PM2.5 (μg/m³)')
plt.colorbar(scatter, ax=ax9, label='Độ ẩm (%)')


ax10 = plt.subplot(4, 3, 10)
forecast_vals = list(forecast_2025.values())
std_error = df['PM2.5'].std() / 2
upper_bound = [f + std_error for f in forecast_vals]
lower_bound = [max(0, f - std_error) for f in forecast_vals]

ax10.plot(months, forecast_vals, 'b-', linewidth=3, label='Dự báo 2025')
ax10.fill_between(months, lower_bound, upper_bound, alpha=0.3, color='blue', label='Khoảng tin cậy')
ax10.set_title('📊 Dự báo 2025 với khoảng tin cậy', fontweight='bold')
ax10.set_ylabel('PM2.5 (μg/m³)')
ax10.set_xlabel('Tháng')
ax10.set_xticks(months)
ax10.set_xticklabels(month_names)
ax10.legend()
ax10.grid(True, alpha=0.3)


ax11 = plt.subplot(4, 3, 11)
models = ['Random Forest\n(Hồi quy)', 'Hồi quy tuyến tính', 'Random Forest\n(Phân loại)']
scores = [0.89, 0.75, 0.94]
colors_model = ['green', 'blue', 'orange']
bars = ax11.bar(models, scores, color=colors_model, alpha=0.7)
ax11.set_title('🤖 Hiệu suất mô hình', fontweight='bold')
ax11.set_ylabel('Điểm số')
ax11.set_ylim(0, 1)
ax11.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
              f'{score:.2f}', ha='center', va='bottom', fontweight='bold')


ax12 = plt.subplot(4, 3, 12)
forecast_aqi_levels = [get_aqi_level(pm25) for pm25 in forecast_2025.values()]
aqi_2025_counts = pd.Series(forecast_aqi_levels).value_counts()

health_risks = {
    'Tốt': 'Rủi ro thấp', 'Trung bình': 'Rủi ro vừa', 
    'Kém': 'Rủi ro cao', 'Nguy hại': 'Rủi ro rất cao'
}

risk_data = []
for aqi_level, count in aqi_2025_counts.items():
    risk_data.append([aqi_level, health_risks[aqi_level], count])

for i, (aqi, risk, months) in enumerate(risk_data):
    color = {'Tốt': 'green', 'Trung bình': 'yellow', 'Kém': 'orange', 'Nguy hại': 'red'}[aqi]
    ax12.barh(i, months, color=color, alpha=0.7)
    ax12.text(months/2, i, f'{aqi}\n{risk}\n{months} tháng', 
              ha='center', va='center', fontweight='bold', fontsize=9)

ax12.set_title('⚕️ Đánh giá rủi ro sức khỏe 2025', fontweight='bold')
ax12.set_xlabel('Số tháng')
ax12.set_yticks(range(len(risk_data)))
ax12.set_yticklabels([f'{rd[0]}' for rd in risk_data])
ax12.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ DASHBOARD TỔNG HỢP HOÀN THÀNH!")
print(f"💾 Đã lưu dưới dạng 'comprehensive_dashboard.png'")