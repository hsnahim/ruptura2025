import streamlit as st
import pandas as pd
import os
import sys

# Adiciona o back-end ao sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "back_end"))

try:
    import main
except ImportError:
    st.warning("Não foi possível importar main.py do back-end.")

try:
    import predictor_regreção
except ImportError:
    st.warning("Não foi possível importar predictor_regreção.py do back-end.")

try:
    import predictor_calssificação
except ImportError:
    st.warning("Não foi possível importar predictor_calssificação.py do back-end.")

st.set_page_config(page_title="Atena IA", page_icon="🧠", layout="wide")
st.title("🧠 Atena IA — Análise de Dados e Previsões")

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Pré-visualização dos Dados")
    st.dataframe(df.head())

    # --- Rodar main.py automaticamente ---
    st.info("Executando pré-processamento...")
    if hasattr(main, "run_pipeline"):
        processed_data = main.run_pipeline(df)
    else:
        processed_data = df
    st.success("Pré-processamento concluído.")

    # --- Escolha tipo de análise ---
    tipo_analise = st.radio("Tipo de Análise:", ["Regressão", "Classificação"], horizontal=True)

    # --- Tabela de parâmetros editável ---
    st.subheader("🧩 Parâmetros para análise (1 linha)")
    input_data = pd.DataFrame([{col: "" for col in df.columns}])
    edited_data = st.data_editor(input_data, num_rows="fixed", use_container_width=True, key="param_editor")
    st.write("Parâmetros atuais:")
    st.dataframe(edited_data)

    # --- Botão Analisar / Prever ---
    if st.button("🚀 Analisar / Prever"):
        st.info(f"Rodando predição de {tipo_analise}...")
        if tipo_analise == "Regressão":
            if hasattr(predictor_regreção, "predict"):
                resultado = predictor_regreção.predict(edited_data)
                st.success("Regressão concluída!")
                st.dataframe(resultado)
            else:
                st.warning("Função 'predict' não encontrada no predictor de regressão.")
        else:
            if hasattr(predictor_calssificação, "predict"):
                resultado = predictor_calssificação.predict(edited_data)
                st.success("Classificação concluída!")
                st.dataframe(resultado)
            else:
                st.warning("Função 'predict' não encontrada no predictor de classificação.")
else:
    st.info("Faça upload de um arquivo CSV para iniciar a análise.")
