# Machine Learning Models - Regression & Classification
# Dự đoán PM2.5 và phân loại chất lượng không khí TP.HCM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           classification_report, confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

print("=== MACHINE LEARNING MODELS - TASK 4 & 5 ===")
print("🎯 Mục tiêu: Dự đoán PM2.5 (Regression) + Phân loại AQI (Classification)")

# 1. ĐỌC DỮ LIỆU ĐÃ ĐƯỢC LÀM SẠCH
try:
    df = pd.read_csv("cleaned_data.csv")
    print(f"📊 Dữ liệu sạch: {df.shape[0]:,} dòng, {df.shape[1]} cột")
except:
    print("⚠️ Không tìm thấy cleaned_data.csv, sử dụng dữ liệu gốc...")
    df = pd.read_csv("HealthyAir_HCMC.csv")
    
    # Tạo lại AQI Level nếu cần
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
    df = df.dropna(subset=['PM2.5', 'TSP', 'Temperature', 'Humidity'])
    print(f"📊 Dữ liệu sau làm sạch: {df.shape[0]:,} dòng")

# 2. CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING
print(f"\n🔧 CHUẨN BỊ DỮ LIỆU:")

# Features (X) - các yếu tố ảnh hưởng đến chất lượng không khí
feature_columns = ['TSP', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity', 'Station_No']
available_features = [col for col in feature_columns if col in df.columns and df[col].notna().sum() > 1000]
print(f"📋 Features sử dụng: {available_features}")

X = df[available_features].copy()
y_regression = df['PM2.5'].copy()  # Target cho regression
y_classification = df['AQI_Level'].copy()  # Target cho classification

# Xử lý missing values cho features
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)
        print(f"  ✅ Điền missing values cho {col}: {X[col].isnull().sum()} → 0")

# Kiểm tra dữ liệu cuối cùng
valid_indices = y_regression.notna() & y_classification.notna()
X = X[valid_indices]
y_regression = y_regression[valid_indices]
y_classification = y_classification[valid_indices]

print(f"📊 Dữ liệu cuối cùng: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"📈 AQI Level distribution:")
aqi_dist = y_classification.value_counts()
for level, count in aqi_dist.items():
    print(f"  {level}: {count:,} ({count/len(y_classification)*100:.1f}%)")

# 3. CHIA TẬP TRAIN/TEST
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42, stratify=pd.qcut(y_regression, q=5, duplicates='drop'))

_, _, y_train_class, y_test_class = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Chia tập dữ liệu:")
print(f"  Training: {X_train.shape[0]:,} samples")
print(f"  Testing: {X_test.shape[0]:,} samples")

# 4. TASK 4: REGRESSION MODELS - DỰ ĐOÁN PM2.5
print(f"\n🎯 TASK 4: REGRESSION MODELS")
print("="*50)

regression_results = {}

# 4.1 Linear Regression
print("📊 1. LINEAR REGRESSION")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_reg)
y_pred_lr = lr_model.predict(X_test_scaled)

# Đánh giá Linear Regression
lr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
lr_r2 = r2_score(y_test_reg, y_pred_lr)
lr_mae = mean_absolute_error(y_test_reg, y_pred_lr)

print(f"  📈 RMSE: {lr_rmse:.2f}")
print(f"  📈 R² Score: {lr_r2:.3f}")
print(f"  📈 MAE: {lr_mae:.2f}")

regression_results['Linear Regression'] = {
    'RMSE': lr_rmse, 'R2': lr_r2, 'MAE': lr_mae, 'predictions': y_pred_lr
}

# 4.2 Random Forest Regression
print(f"\n📊 2. RANDOM FOREST REGRESSION")
rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg_model.fit(X_train, y_train_reg)  # Không cần scaling cho RF
y_pred_rf_reg = rf_reg_model.predict(X_test)

# Đánh giá Random Forest Regression
rf_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg))
rf_reg_r2 = r2_score(y_test_reg, y_pred_rf_reg)
rf_reg_mae = mean_absolute_error(y_test_reg, y_pred_rf_reg)

print(f"  📈 RMSE: {rf_reg_rmse:.2f}")
print(f"  📈 R² Score: {rf_reg_r2:.3f}")
print(f"  📈 MAE: {rf_reg_mae:.2f}")

regression_results['Random Forest'] = {
    'RMSE': rf_reg_rmse, 'R2': rf_reg_r2, 'MAE': rf_reg_mae, 'predictions': y_pred_rf_reg
}

# 5. TASK 5: CLASSIFICATION MODELS - PHÂN LOẠI AQI
print(f"\n🎯 TASK 5: CLASSIFICATION MODELS")
print("="*50)

classification_results = {}

# 5.1 Random Forest Classification
print("📊 1. RANDOM FOREST CLASSIFICATION")
rf_class_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_class_model.fit(X_train, y_train_class)
y_pred_rf_class = rf_class_model.predict(X_test)

# Đánh giá Random Forest Classification
rf_class_accuracy = accuracy_score(y_test_class, y_pred_rf_class)
print(f"  📈 Accuracy: {rf_class_accuracy:.3f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test_class, y_pred_rf_class))

classification_results['Random Forest'] = {
    'accuracy': rf_class_accuracy, 'predictions': y_pred_rf_class, 'model': rf_class_model
}

# 5.2 Decision Tree Classification
print(f"\n📊 2. DECISION TREE CLASSIFICATION")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train_class)
y_pred_dt = dt_model.predict(X_test)

# Đánh giá Decision Tree
dt_accuracy = accuracy_score(y_test_class, y_pred_dt)
print(f"  📈 Accuracy: {dt_accuracy:.3f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test_class, y_pred_dt))

classification_results['Decision Tree'] = {
    'accuracy': dt_accuracy, 'predictions': y_pred_dt, 'model': dt_model
}

# 6. FEATURE IMPORTANCE ANALYSIS (Task 6)
print(f"\n🎯 TASK 6: FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Feature importance từ Random Forest Regression
print("📊 FEATURE IMPORTANCE - REGRESSION (PM2.5 Prediction):")
feature_importance_reg = pd.DataFrame({
    'feature': available_features,
    'importance': rf_reg_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_reg.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Feature importance từ Random Forest Classification
print(f"\n📊 FEATURE IMPORTANCE - CLASSIFICATION (AQI Level):")
feature_importance_class = pd.DataFrame({
    'feature': available_features,
    'importance': rf_class_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_class.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 7. VISUALIZATIONS
print(f"\n📊 CREATING VISUALIZATIONS...")

# Tạo figure cho tất cả các plots
fig = plt.figure(figsize=(20, 15))

# Plot 1: Regression Results Comparison
plt.subplot(3, 3, 1)
models = list(regression_results.keys())
rmse_values = [regression_results[model]['RMSE'] for model in models]
r2_values = [regression_results[model]['R2'] for model in models]

x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
plt.bar(x + width/2, [r2*50 for r2 in r2_values], width, label='R²×50', alpha=0.8)
plt.xlabel('Mô hình')
plt.ylabel('Điểm số')
plt.title('So sánh các mô hình Regression')
plt.xticks(x, models)
plt.legend()

# Plot 2: Actual vs Predicted (Best Regression Model)
plt.subplot(3, 3, 2)
best_reg_model = 'Random Forest' if rf_reg_r2 > lr_r2 else 'Linear Regression'
best_predictions = regression_results[best_reg_model]['predictions']
plt.scatter(y_test_reg, best_predictions, alpha=0.5, s=20)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Thực tế PM2.5')
plt.ylabel('Dự đoán PM2.5')
plt.title(f'Thực tế vs Dự đoán - {best_reg_model}')

# Plot 3: Feature Importance - Regression
plt.subplot(3, 3, 3)
top_features_reg = feature_importance_reg.head(7)
plt.barh(top_features_reg['feature'], top_features_reg['importance'])
plt.xlabel('Mức độ quan trọng')
plt.title('Tầm quan trọng của biến - Regression')

# Plot 4: Classification Accuracy Comparison
plt.subplot(3, 3, 4)
class_models = list(classification_results.keys())
class_accuracies = [classification_results[model]['accuracy'] for model in class_models]
bars = plt.bar(class_models, class_accuracies, color=['skyblue', 'lightcoral'])
plt.ylabel('Độ chính xác')
plt.title('So sánh các mô hình Classification')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{class_accuracies[i]:.3f}', ha='center')

# Plot 5: Confusion Matrix (Best Classification Model)
plt.subplot(3, 3, 5)
best_class_model = 'Random Forest' if rf_class_accuracy > dt_accuracy else 'Decision Tree'
best_class_predictions = classification_results[best_class_model]['predictions']
cm = confusion_matrix(y_test_class, best_class_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tốt', 'Vừa', 'Kém', 'Nguy hại'],
            yticklabels=['Tốt', 'Vừa', 'Kém', 'Nguy hại'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title(f'Ma trận nhầm lẫn - {best_class_model}')

# Plot 6: Feature Importance - Classification
plt.subplot(3, 3, 6)
top_features_class = feature_importance_class.head(7)
plt.barh(top_features_class['feature'], top_features_class['importance'], color='orange')
plt.xlabel('Mức độ quan trọng')
plt.title('Tầm quan trọng của biến - Classification')

# Plot 7: Prediction Error Distribution
plt.subplot(3, 3, 7)
errors = y_test_reg - best_predictions
plt.hist(errors, bins=30, alpha=0.7, color='green')
plt.xlabel('Sai số dự đoán')
plt.ylabel('Tần suất')
plt.title('Phân bố sai số dự đoán')

# Plot 8: AQI Level Prediction Accuracy by Class
plt.subplot(3, 3, 8)
class_accuracy_detail = []
for level in ['Good', 'Moderate', 'Unhealthy', 'Hazardous']:
    if level in y_test_class.values:
        mask = y_test_class == level
        if mask.sum() > 0:
            acc = (y_test_class[mask] == best_class_predictions[mask]).mean()
            class_accuracy_detail.append((level, acc))

if class_accuracy_detail:
    levels, accs = zip(*class_accuracy_detail)
    colors = ['green', 'yellow', 'orange', 'red'][:len(levels)]
    bars = plt.bar(levels, accs, color=colors, alpha=0.7)
    plt.ylabel('Độ chính xác')
    plt.title('Độ chính xác theo mức AQI')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{accs[i]:.2f}', ha='center')

# Plot 9: Model Performance Summary
plt.subplot(3, 3, 9)
summary_data = {
    'Regression': [f'Best: {best_reg_model}', 
                   f'RMSE: {regression_results[best_reg_model]["RMSE"]:.2f}',
                   f'R²: {regression_results[best_reg_model]["R2"]:.3f}'],
    'Classification': [f'Best: {best_class_model}',
                      f'Accuracy: {classification_results[best_class_model]["accuracy"]:.3f}',
                      f'Classes: {len(np.unique(y_test_class))}']
}
plt.text(0.1, 0.8, 'MODEL PERFORMANCE SUMMARY', fontsize=14, weight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.6, '\n'.join([f'{k}: {", ".join(v)}' for k, v in summary_data.items()]), 
         fontsize=10, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.savefig('ml_models_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. KẾT LUẬN VÀ LƯU KẾT QUẢ
print(f"\n🎉 KẾT QUẢ MACHINE LEARNING:")
print("="*50)
print(f"🔹 REGRESSION (Dự đoán PM2.5):")
print(f"  Best Model: {best_reg_model}")
print(f"  RMSE: {regression_results[best_reg_model]['RMSE']:.2f} μg/m³")
print(f"  R² Score: {regression_results[best_reg_model]['R2']:.3f}")
print(f"  MAE: {regression_results[best_reg_model]['MAE']:.2f} μg/m³")

print(f"\n🔹 CLASSIFICATION (Phân loại AQI):")
print(f"  Best Model: {best_class_model}")
print(f"  Overall Accuracy: {classification_results[best_class_model]['accuracy']:.3f}")

print(f"\n🔹 TOP 3 YẾU TỐ ẢNH HƯỞNG NHẤT:")
print("  Regression:")
for i, (_, row) in enumerate(feature_importance_reg.head(3).iterrows()):
    print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")
print("  Classification:")
for i, (_, row) in enumerate(feature_importance_class.head(3).iterrows()):
    print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")

# Lưu kết quả để sử dụng cho forecast
results_summary = {
    'best_regression_model': best_reg_model,
    'best_classification_model': best_class_model,
    'feature_columns': available_features,
    'regression_performance': regression_results[best_reg_model],
    'classification_performance': classification_results[best_class_model],
    'feature_importance_reg': feature_importance_reg.to_dict(),
    'feature_importance_class': feature_importance_class.to_dict()
}

import pickle
with open('ml_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
    
print(f"\n💾 Saved results to 'ml_results.pkl'")
print(f"✅ Task 4, 5, 6 COMPLETED!")
print(f"🚀 Ready for Task 7: Time Series Forecasting 2025")