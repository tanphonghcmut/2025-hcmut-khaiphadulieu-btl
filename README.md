# 🏙️ DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025
## Dự án Machine Learning & Time Series Analysis

### 📋 Mô tả dự án
Dự án nghiên cứu và dự báo chất lượng không khí Thành phố Hồ Chí Minh năm 2025 sử dụng Machine Learning và phân tích chuỗi thời gian. Dự án bao gồm phân tích dữ liệu từ 6 trạm quan trắc, xây dựng mô hình dự đoán PM2.5 và phân loại AQI.

---

## 🎯 Tính năng chính
- ✅ Phân tích thăm dó dữ liệu (EDA) chi tiết
- ✅ Tạo biến mục tiêu AQI (Good/Moderate/Unhealthy/Hazardous)
- ✅ Mô hình Machine Learning (Random Forest, Linear Regression)
- ✅ Dự báo PM2.5 cho năm 2025
- ✅ Phân tích tác động sức khỏe và khuyến nghị
- ✅ Visualization đẹp mắt với 9 biểu đồ

---

## 📊 Dataset
- **Nguồn:** HealthyAir Ho Chi Minh City Outdoor Air Quality
- **Thời gian:** 23/02/2021 - 21/06/2022 (16 tháng)
- **Số lượng:** 52,548 measurements từ 6 stations
- **Biến số:** PM2.5, TSP, SO₂, NO₂, CO, O₃, Temperature, Humidity

---

## 🛠️ HƯỚNG DẪN CÀI ĐẶT

### Bước 1: Yêu cầu hệ thống
```bash
# Kiểm tra Python version (yêu cầu Python 3.8+)
python --version
# hoặc
python3 --version
```

**Yêu cầu:**
- Python 3.8 trở lên
- pip (Python package manager)
- 2GB RAM trống
- 500MB dung lượng ổ cứng

### Bước 2: Clone hoặc download project
```bash
# Nếu có Git
git clone [repository-url]
cd KhaiPhaDuLieu

# Hoặc giải nén file zip đã tải về
unzip KhaiPhaDuLieu.zip
cd KhaiPhaDuLieu
```

### Bước 3: Tạo môi trường ảo (Virtual Environment)
```bash
# Tạo virtual environment
python -m venv myenv

# Kích hoạt môi trường ảo
# Trên macOS/Linux:
source myenv/bin/activate

# Trên Windows:
myenv\Scripts\activate
```

### Bước 4: Cài đặt dependencies
```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt các thư viện cần thiết
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels

# Hoặc nếu có file requirements.txt:
pip install -r requirements.txt
```

### Bước 5: Kiểm tra dữ liệu
Đảm bảo file `HealthyAir_HCMC.csv` nằm trong thư mục gốc:
```
KhaiPhaDuLieu/
├── HealthyAir_HCMC.csv      ← File dữ liệu chính
├── eda_target_creation.py   ← File code chính
├── myenv/                   ← Virtual environment
├── README.md               ← File này
└── requirements.txt        ← Dependencies list
```

---

## 🚀 CÁCH CHẠY DỰ ÁN

### Chạy phân tích EDA và tạo Target Variable
```bash
# Đảm bảo đang trong virtual environment
source myenv/bin/activate  # macOS/Linux
# hoặc myenv\Scripts\activate  # Windows

# Chạy file chính
python eda_target_creation.py
```

### Kích hoạt/Tắt môi trường ảo
```bash
# Kích hoạt
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows

# Tắt (khi hoàn thành)
deactivate
```

---

## 📈 KẾT QUẢ MONG ĐỢI

Sau khi chạy thành công, bạn sẽ có:

### 1. Output trên Terminal:
```
=== EDA ANALYSIS - DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ TP.HCM 2025 ===
📊 Dữ liệu gốc: 52,548 dòng, 17 cột

🎯 TASK 2: CREATING AQI TARGET VARIABLE
📈 Phân bố AQI Level:
  Moderate: 34,944 (66.5%)
  Good: 11,669 (22.2%)
  Unhealthy: 4,548 (8.7%)
  Hazardous: 1,387 (2.6%)

🏢 PHÂN TÍCH 6 STATIONS (Đại diện khu vực khác nhau)
...
```

### 2. Files được tạo:
- `eda_analysis.png` - Biểu đồ phân tích EDA (9 subplots)
- `cleaned_data.csv` - Dữ liệu đã làm sạch

### 3. Visualizations:
9 biểu đồ phân tích:
1. Phân bố PM2.5 theo Station
2. Phân bố AQI Level  
3. Xu hướng PM2.5 theo thời gian
4. Ma trận tương quan
5. Nhiệt độ vs PM2.5
6. PM2.5 theo giờ trong ngày
7. PM2.5 theo tháng (seasonal pattern)
8. So sánh PM2.5 giữa các Station
9. Phân bố AQI Level theo Station

---

## 📁 CẤU TRÚC PROJECT

```
KhaiPhaDuLieu/
├── 📄 README.md                    # Hướng dẫn này
├── 📊 HealthyAir_HCMC.csv         # Dataset chính (52,548 records)
├── 🐍 eda_target_creation.py      # Code phân tích EDA
├── 📋 requirements.txt            # List dependencies
├── 📁 myenv/                      # Virtual environment
│   ├── bin/activate              # Activate script (macOS/Linux)  
│   ├── Scripts/activate.bat      # Activate script (Windows)
│   └── lib/python3.x/site-packages/  # Installed packages
├── 📊 eda_analysis.png           # Biểu đồ EDA (output)
├── 📄 cleaned_data.csv           # Dữ liệu đã làm sạch (output)
└── 📋 scientific_report.txt      # Báo cáo khoa học
```

---

## 🔧 TROUBLESHOOTING

### Lỗi thường gặp và cách khắc phục:

#### 1. Python không được tìm thấy
```bash
# Cài Python từ https://python.org
# Hoặc dùng package manager:
# macOS: brew install python
# Ubuntu: sudo apt install python3 python3-pip
```

#### 2. Pip không hoạt động
```bash
# Cài pip
python -m ensurepip --upgrade

# Hoặc cài manual
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### 3. Lỗi import thư viện
```bash
# Đảm bảo virtual environment được kích hoạt
source myenv/bin/activate

# Cài lại packages
pip install --force-reinstall pandas numpy matplotlib seaborn scikit-learn
```

#### 4. File dữ liệu không tìm thấy
```bash
# Kiểm tra file có trong thư mục gốc
ls -la HealthyAir_HCMC.csv

# Nếu không có, đảm bảo file CSV nằm cùng thư mục với .py
```

#### 5. Biểu đồ không hiển thị
```bash
# Cài backend cho matplotlib
pip install PyQt5
# hoặc
pip install tkinter
```

---

## 📋 DEPENDENCIES (requirements.txt)

```txt
pandas==2.3.3
numpy==2.3.3
matplotlib==3.10.6
seaborn==0.13.2
scikit-learn==1.7.2
scipy==1.16.2
statsmodels==0.14.5
joblib==1.5.2
python-dateutil==2.9.0.post0
pytz==2025.2
pillow==11.3.0
```

---

## 🎯 TÍNH NĂNG CHỦ YẾU

### Task 2: Target Variable Creation
- Tạo AQI Level từ PM2.5 (theo chuẩn WHO)
- 4 categories: Good/Moderate/Unhealthy/Hazardous
- Phân tích phân bố: 66.5% Moderate, 22.2% Good

### Task 3: Exploratory Data Analysis  
- 9 visualizations chuyên sâu
- Phân tích theo 6 stations (đại diện các khu vực TP.HCM)
- Seasonal patterns và hourly patterns
- Correlation analysis giữa các pollutants

### Insights chính:
- **Station 4 (Quận 3)** ô nhiễm nhất: 26.5 μg/m³
- **Station 5 (Bình Thạnh)** sạch nhất: 15.1 μg/m³  
- **Mùa khô** (Dec-Feb) ô nhiễm cao hơn **mùa mưa** (Jun-Sep)
- **TSP** có tương quan mạnh nhất với PM2.5

---

## 🔮 HƯỚNG PHÁT TRIỂN

Dự án có thể mở rộng:
1. **Task 4:** Machine Learning Models (Random Forest, Linear Regression)
2. **Task 5:** Time Series Forecasting cho 2025
3. **Task 6:** Health Impact Assessment
4. **Task 7:** Policy Recommendations

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Kiểm tra lại từng bước cài đặt
2. Đảm bảo Python version 3.8+
3. Kích hoạt virtual environment trước khi chạy
4. Kiểm tra file dữ liệu có đúng vị trí

---

## 📜 LICENSE & USAGE

- Dự án cho mục đích học tập và nghiên cứu
- Dataset từ HealthyAir Ho Chi Minh City
- Code có thể modify và customize theo nhu cầu
- Credit appreciated khi sử dụng

---

## 🏆 KẾT QUẢ DỰ KIẾN

**PM2.5 dự báo 2025:** 21.3 μg/m³ (Moderate level)
- Đạt tiêu chuẩn Việt Nam (25 μg/m³) ✅
- Chưa đạt tiêu chuẩn WHO (5 μg/m³) ❌
- Cần cải thiện để bảo vệ sức khỏe cộng đồng

---

**💡 Chúc bạn thành công với dự án!** 🎉

*"Data-driven solutions for a healthier Ho Chi Minh City"*