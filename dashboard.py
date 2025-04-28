import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load model ElasticNet
elastic_model = joblib.load('elasticnet_model.pkl')


# Load data dari CSV yang diupload
df = pd.read_csv('D:\#Perkuliahan\Datmin\fastapi\Wellbeing_and_lifestyle_data_Kaggle.csv')

# Periksa beberapa baris pertama untuk memverifikasi apakah data terbaca dengan benar
st.write(df.head())

# Scaling fitur untuk model ElasticNet
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['TODO_COMPLETED', 'SUFFICIENT_INCOME', 'DAILY_STRESS', 'FRUITS_VEGGIES', 'ACHIEVEMENT']])

# Prediksi dengan ElasticNet
predictions = elastic_model.predict(X_scaled)

# Menampilkan judul aplikasi
st.title("Dashboard Analisis Keseimbangan Kerja-Hidup")

# Pilar 1: Menampilkan Insight Prediksi
st.subheader("Prediksi Keseimbangan Kerja-Hidup")
st.write("Hasil prediksi menggunakan model ElasticNet untuk skor keseimbangan kerja-hidup adalah sebagai berikut:")

# Tampilkan tabel hasil prediksi
df['Prediksi Keseimbangan Kerja-Hidup'] = predictions
st.dataframe(df[['TODO_COMPLETED', 'SUFFICIENT_INCOME', 'DAILY_STRESS', 'FRUITS_VEGGIES', 'ACHIEVEMENT', 'Prediksi Keseimbangan Kerja-Hidup']])

# Pilar 2: Visualisasi Data untuk Analisis
st.subheader("Visualisasi Data")

# Visualisasi hubungan antara stres dan keseimbangan kerja-hidup
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['DAILY_STRESS'], y=df['WORK_LIFE_BALANCE_SCORE'], color='teal')
plt.title("Hubungan Antara Stres Harian dan Skor Keseimbangan Kerja-Hidup")
plt.xlabel("Stres Harian")
plt.ylabel("Skor Keseimbangan Kerja-Hidup")
st.pyplot()

# Visualisasi distribusi skor keseimbangan kerja-hidup
plt.figure(figsize=(8, 6))
sns.histplot(df['WORK_LIFE_BALANCE_SCORE'], bins=10, kde=True, color='blue')
plt.title("Distribusi Skor Keseimbangan Kerja-Hidup")
plt.xlabel("Skor Keseimbangan Kerja-Hidup")
plt.ylabel("Frekuensi")
st.pyplot()

# Pilar 3: Metrik Evaluasi Model
st.subheader("Evaluasi Model ElasticNet")
ols_r2 = r2_score(df['WORK_LIFE_BALANCE_SCORE'], predictions)
ols_mse = mean_squared_error(df['WORK_LIFE_BALANCE_SCORE'], predictions)
ols_mae = mean_absolute_error(df['WORK_LIFE_BALANCE_SCORE'], predictions)

# Menampilkan metrik
st.write(f"R² (Koefisien Determinasi): {ols_r2:.4f}")
st.write(f"Mean Squared Error (MSE): {ols_mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {ols_mae:.4f}")

# Pilar 4: Analisis Korelasi Fitur
st.subheader("Korelasi Antar Fitur")
correlation_matrix = df[['TODO_COMPLETED', 'SUFFICIENT_INCOME', 'DAILY_STRESS', 'FRUITS_VEGGIES', 'ACHIEVEMENT', 'WORK_LIFE_BALANCE_SCORE']].corr()

# Plot korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi Antar Fitur")
st.pyplot()

# Menambahkan komentar atau insight tambahan
st.write("Insight dari analisis menunjukkan korelasi kuat antara stres harian dan keseimbangan kerja-hidup.")
