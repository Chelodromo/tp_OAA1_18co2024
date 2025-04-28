import streamlit as st
import pandas as pd
import requests

FASTAPI_URL = "http://fastapi_app:8000"  # URL interna para Docker, o http://localhost:8000 si corres local

st.title("🌎 Predicción de Polvo Atmosférico")

# --- Selección de modo
modo = st.radio("Selecciona el modo:", ["Manual", "Batch (CSV)"])

if modo == "Manual":
    st.subheader("🚀 Ingresá los datos manualmente")

    TempOut = st.number_input("Temperatura Exterior (TempOut)", value=20.0)
    DewPt_ = st.number_input("Punto de Rocío (DewPt.)", value=5.0)
    WSpeed = st.number_input("Velocidad de Viento Media (WSpeed)", value=10.0)
    WHSpeed = st.number_input("Ráfaga de Viento (WHSpeed)", value=20.0)
    Bar = st.number_input("Presión Barométrica (Bar)", value=1013.0)
    Rain = st.number_input("Lluvia (Rain)", value=0.0)
    ET = st.number_input("Evapotranspiración (ET)", value=0.0)
    WDir_deg = st.number_input("Dirección del Viento (WDir_deg)", value=180.0)
    Date_num = st.number_input("Fecha (Date_num, timestamp)", value=1743465600.0)

    if st.button("🔮 Predecir"):
        payload = {
            "TempOut": TempOut,
            "DewPt_": DewPt_,
            "WSpeed": WSpeed,
            "WHSpeed": WHSpeed,
            "Bar": Bar,
            "Rain": Rain,
            "ET": ET,
            "WDir_deg": WDir_deg,
            "Date_num": Date_num
        }
        response = requests.post(f"{FASTAPI_URL}/predict", json=payload)

        if response.status_code == 200:
            st.success(f"Predicción: {response.json()}")
        else:
            st.error(f"Error en predicción: {response.text}")

else:
    st.subheader("📁 Subí un archivo CSV")

    uploaded_file = st.file_uploader("Elegí el archivo CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa del archivo:")
        st.dataframe(df)

        if st.button("🔮 Predecir en batch"):
            payload = df.to_dict(orient="records")
            response = requests.post(f"{FASTAPI_URL}/predict_batch", json=payload)

            if response.status_code == 200:
                preds = response.json()
                st.write("📈 Predicciones:")
                st.json(preds)
            else:
                st.error(f"Error en predicción batch: {response.text}")
