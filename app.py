"""
app.py
------
Dashboard interactivo con Streamlit para predecir riesgo crediticio.

Uso:
    streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from model import CreditNet, Preprocessor
from data.generate_data import generate_credit_data

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Riesgo Crediticio",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df   = generate_credit_data(n=1000)
    prep = Preprocessor()
    _, _, _, input_dim = prep.fit_transform(df)
    model = CreditNet(input_dim)
    model.load_state_dict(torch.load('credit_model.pt', weights_only=True))
    model.eval()
    return model, prep

model, prep = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 Predictor de Riesgo Crediticio")
st.markdown("Ingresa los datos del cliente para evaluar si se debe aprobar o rechazar el crédito.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Datos personales")
    age      = st.slider("Edad", 18, 75, 35)
    housing  = st.selectbox("Tipo de vivienda", ['own', 'rent', 'free'])
    employment = st.selectbox("Antigüedad laboral",
                              ['unemployed', '<1yr', '1-4yr', '4-7yr', '>7yr'])

with col2:
    st.subheader("Datos financieros")
    amount   = st.number_input("Monto del crédito ($)", 500, 15000, 3000, step=500)
    duration = st.selectbox("Plazo (meses)", [6, 12, 18, 24, 36, 48, 60])
    savings  = st.selectbox("Ahorros", ['none', 'little', 'moderate', 'quite_rich', 'rich'])

with col3:
    st.subheader("Historial")
    credit_history   = st.selectbox("Historial crediticio",
                                    ['existing_paid', 'paid', 'delayed', 'critical', 'no_credits'])
    purpose          = st.selectbox("Propósito del crédito",
                                    ['car', 'furniture', 'education', 'business', 'repairs', 'other'])
    existing_credits = st.selectbox("Créditos existentes", [1, 2, 3, 4])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔍 Evaluar cliente", type="primary", use_container_width=True):
    cliente = {
        'age': age, 'duration': duration, 'amount': amount,
        'savings': savings, 'employment': employment,
        'credit_history': credit_history, 'purpose': purpose,
        'housing': housing, 'existing_credits': existing_credits
    }

    with torch.no_grad():
        tensor = prep.transform_single(cliente)
        prob   = model(tensor).item()

    decision = "✅ APROBAR" if prob > 0.5 else "❌ RECHAZAR"
    color    = "green" if prob > 0.5 else "red"

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Decisión", decision)
    with r2:
        st.metric("Probabilidad de pago", f"{prob*100:.1f}%")
    with r3:
        st.metric("Nivel de riesgo",
                  "Bajo" if prob > 0.7 else "Medio" if prob > 0.5 else "Alto")

    # Gauge visual
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh(0, prob,        color='#16a34a', height=0.5)
    ax.barh(0, 1 - prob, left=prob, color='#dc2626', height=0.5)
    ax.axvline(0.5, color='white', linewidth=2, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Probabilidad de pago del cliente')
    st.pyplot(fig)

st.divider()
st.caption("Proyecto académico · Red Neuronal con PyTorch · Riesgo Crediticio")