import streamlit as st
import joblib
import numpy as np

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
fitur  = joblib.load('fitur.pkl')

st.title('🔮 Prediksi Customer Churn')
st.write('Masukkan data pelanggan untuk memprediksi kemungkinan churn.')

complains    = st.selectbox('Pernah komplain?', [0, 1], format_func=lambda x: 'Ya' if x else 'Tidak')
status       = st.selectbox('Status akun', [1, 2], format_func=lambda x: 'Aktif' if x == 1 else 'Non-Aktif')
freq_of_use  = st.slider('Frekuensi penggunaan (per bulan)', 0, 200, 50)

if st.button('Prediksi'):
    data    = np.array([[complains, status, freq_of_use]])
    data_sc = scaler.transform(data)
    pred    = model.predict(data_sc)[0]
    prob    = model.predict_proba(data_sc)[0][1]

    if pred == 1:
        st.error(f'⚠️ Pelanggan diprediksi CHURN (probabilitas: {prob:.1%})')
    else:
        st.success(f'✅ Pelanggan diprediksi TIDAK churn (probabilitas churn: {prob:.1%})')