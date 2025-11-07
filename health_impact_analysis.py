# PHÂN TÍCH TÁC ĐỘNG SỨC KHỎE - CHẤT LƯỢNG KHÔNG KHÍ TP.HCM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== PHÂN TÍCH TÁC ĐỘNG SỨC KHỎE VÀ ĐỀ XUẤT CHÍNH SÁCH ===")


df = pd.read_csv("HealthyAir_HCMC.csv")
df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')


def get_aqi_level(pm25):
    if pm25 <= 12: return "Tốt"
    elif pm25 <= 35.4: return "Trung bình"
    elif pm25 <= 55.4: return "Kém"
    else: return "Nguy hại"

def get_health_risk(pm25):
    if pm25 <= 12: return "Rủi ro thấp"
    elif pm25 <= 35.4: return "Rủi ro trung bình"
    elif pm25 <= 55.4: return "Rủi ro cao"
    else: return "Rủi ro rất cao"


df['AQI_Level'] = df['PM2.5'].apply(get_aqi_level)
df['Health_Risk'] = df['PM2.5'].apply(get_health_risk)


forecast_2025 = {
    1: 30.3, 2: 19.1, 3: 22.1, 4: 22.6, 5: 18.2, 6: 19.1,
    7: 15.2, 8: 15.3, 9: 13.9, 10: 24.3, 11: 25.6, 12: 28.1
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('PHÂN TÍCH TÁC ĐỘNG SỨC KHỎE - CHẤT LƯỢNG KHÔNG KHÍ TP.HCM', fontsize=16, fontweight='bold')


ax1 = axes[0, 0]
station_pm25 = df.groupby('Station_No')['PM2.5'].mean()
station_info = {
    1: "Q1-Thương mại", 2: "Q7-Công nghiệp", 3: "Thủ Đức-Giao thông",
    4: "Q3-Dân cư", 5: "Bình Thạnh-Hỗn hợp", 6: "Q9-Ngoại thành"
}
station_labels = [f"Trạm {i}\n{station_info[i].split('-')[1]}" for i in station_pm25.index]
colors = ['red' if x > 25 else 'orange' if x > 20 else 'yellow' if x > 15 else 'green' 
          for x in station_pm25.values]
bars = ax1.bar(range(len(station_pm25)), station_pm25.values, color=colors)
ax1.set_title('Rủi ro sức khỏe theo khu vực trạm đo')
ax1.set_ylabel('PM2.5 (μg/m³)')
ax1.set_xticks(range(len(station_pm25)))
ax1.set_xticklabels(station_labels, rotation=45, fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[0, 1]
vuln_groups = ['Trẻ em\n(0-14 tuổi)', 'Người cao tuổi\n(65+ tuổi)', 
               'Bệnh nhân\nhô hấp', 'Bệnh nhân\ntim mạch']
vuln_deaths = [2500, 1800, 3200, 2800] 
ax2.bar(range(len(vuln_groups)), vuln_deaths, color=['red', 'orange', 'darkred', 'purple'])
ax2.set_title('Số ca tử vong dư thừa theo nhóm dễ bị tổn thương')
ax2.set_ylabel('Ước tính số ca tử vong/năm')
ax2.set_xticks(range(len(vuln_groups)))
ax2.set_xticklabels(vuln_groups, rotation=0, fontsize=9)


ax3 = axes[0, 2]
monthly_pm25 = df.groupby(df['datetime'].dt.month)['PM2.5'].mean()
month_names = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
month_colors = ['red' if x > 25 else 'orange' if x > 20 else 'yellow' if x > 15 else 'green' 
                for x in monthly_pm25.values]
ax3.bar(monthly_pm25.index, monthly_pm25.values, color=month_colors)
ax3.set_title('Mẫu rủi ro sức khỏe theo mùa')
ax3.set_ylabel('PM2.5 (μg/m³)')
ax3.set_xlabel('Tháng')
ax3.set_xticks(monthly_pm25.index)
ax3.set_xticklabels([month_names[i-1] for i in monthly_pm25.index], rotation=0)
ax3.grid(True, alpha=0.3, axis='y')


ax4 = axes[1, 0]
econ_categories = ['Chi phí tử vong\ndư thừa', 'Chi phí\ny tế', 'Mất năng suất\nlao động']
econ_values = [60, 25, 15]  # Phần trăm
ax4.pie(econ_values, labels=econ_categories, autopct='%1.1f%%', startangle=90)
ax4.set_title('Phân bổ tác động kinh tế\n(% tổng chi phí/năm)')


ax5 = axes[1, 1]
forecast_months = list(range(1, 13))
forecast_pm25_values = list(forecast_2025.values())
forecast_colors = ['red' if x > 25 else 'orange' if x > 20 else 'yellow' if x > 15 else 'green' 
                  for x in forecast_pm25_values]
ax5.bar(forecast_months, forecast_pm25_values, color=forecast_colors)
ax5.set_title('Dự báo rủi ro sức khỏe năm 2025')
ax5.set_ylabel('PM2.5 (μg/m³)')
ax5.set_xlabel('Tháng năm 2025')
ax5.set_xticks(forecast_months)
ax5.set_xticklabels([month_names[i-1] for i in forecast_months], rotation=0)
ax5.grid(True, alpha=0.3, axis='y')

ax6 = axes[1, 2]
risk_levels = ['Tốt', 'Trung bình', 'Kém', 'Nguy hại']
historical_risk_counts = [df[df['AQI_Level'] == level].shape[0] for level in risk_levels]
forecast_risk_counts = [sum(1 for pm25 in forecast_2025.values() if get_aqi_level(pm25) == level) 
                       for level in risk_levels]

x = np.arange(len(risk_levels))
width = 0.35
ax6.bar(x - width/2, historical_risk_counts, width, label='Dữ liệu lịch sử', alpha=0.8)
ax6.bar(x + width/2, forecast_risk_counts, width, label='Dự báo 2025', alpha=0.8)
ax6.set_title('So sánh phân bố rủi ro sức khỏe')
ax6.set_ylabel('Số lượng')
ax6.set_xticks(x)
ax6.set_xticklabels(risk_levels, rotation=0)
ax6.legend()

plt.tight_layout()
plt.savefig('health_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ PHÂN TÍCH TÁC ĐỘNG SỨC KHỎE HOÀN THÀNH!")
print(f"💾 Biểu đồ đã lưu dưới dạng 'health_impact_analysis.png'")