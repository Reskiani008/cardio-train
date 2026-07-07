import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Risiko Kardiovaskular",
    page_icon="❤️",
    layout="centered"
)

# 2. Load Model dan Scaler
# Menggunakan st.cache_resource agar model tidak diload ulang setiap kali ada interaksi
@st.cache_resource
def load_components():
    with open('gradient_boosting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler (5).pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_components()
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py")
    st.stop()

# 3. Antarmuka Pengguna (UI)
st.title('❤️ Aplikasi Prediksi Penyakit Kardiovaskular')
st.write("Silakan masukkan data pemeriksaan kesehatan pasien di bawah ini:")

# Membuat form input
with st.form("data_pasien"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Catatan: Dataset Cardio sering menggunakan umur dalam 'hari' (days).
        # Kita minta input tahun, lalu kita kalikan 365 di belakang layar.
        age_years = st.number_input('Umur (Tahun)', min_value=20, max_value=120, value=50)
        gender = st.selectbox('Jenis Kelamin', [1, 2], format_func=lambda x: 'Wanita (1)' if x == 1 else 'Pria (2)')
        height = st.number_input('Tinggi Badan (cm)', min_value=100, max_value=250, value=165)
        weight = st.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=65.0)
        ap_hi = st.number_input('Tekanan Darah Sistolik (ap_hi)', min_value=50, max_value=250, value=120, help="Angka atas, normalnya sekitar 120")
        ap_lo = st.number_input('Tekanan Darah Diastolik (ap_lo)', min_value=30, max_value=200, value=80, help="Angka bawah, normalnya sekitar 80")

    with col2:
        cholesterol = st.selectbox('Tingkat Kolesterol', [1, 2, 3], format_func=lambda x: 'Normal (1)' if x == 1 else ('Di atas normal (2)' if x == 2 else 'Sangat tinggi (3)'))
        gluc = st.selectbox('Tingkat Glukosa', [1, 2, 3], format_func=lambda x: 'Normal (1)' if x == 1 else ('Di atas normal (2)' if x == 2 else 'Sangat tinggi (3)'))
        smoke = st.selectbox('Apakah Merokok?', [0, 1], format_func=lambda x: 'Tidak (0)' if x == 0 else 'Ya (1)')
        alco = st.selectbox('Konsumsi Alkohol?', [0, 1], format_func=lambda x: 'Tidak (0)' if x == 0 else 'Ya (1)')
        active = st.selectbox('Aktivitas Fisik?', [0, 1], format_func=lambda x: 'Tidak (0)' if x == 0 else 'Ya (1)')

    submit = st.form_submit_button("Lakukan Prediksi")

# 4. Logika Prediksi
if submit:
    # Mengonversi umur menjadi hari (karena model asli dari dataset Cardio biasanya dilatih dengan umur dalam hari)
    age_days = age_years * 365.25
    
    # Menyiapkan data untuk prediksi (urutan kolom harus sama dengan saat training/scaler)
    input_data = pd.DataFrame({
        'age': [age_days],
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })
    
    # Normalisasi (scaling) menggunakan scaler yang diupload
    scaled_input = scaler.transform(input_data)
    
    # Memprediksi hasil
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    st.divider()
    
    # Menampilkan Hasil
    if prediction == 1:
        st.error(f"⚠️ **Hasil Prediksi:** Pasien terdeteksi BERISIKO TINGGI terkena penyakit kardiovaskular.")
        st.write(f"Tingkat Keyakinan Model: **{probabilities[1] * 100:.2f}%**")
    else:
        st.success(f"✅ **Hasil Prediksi:** Pasien diprediksi BERISIKO RENDAH terkena penyakit kardiovaskular.")
        st.write(f"Tingkat Keyakinan Model: **{probabilities[0] * 100:.2f}%**")