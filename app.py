import streamlit as st
import pandas as pd
import os
import sys

# Adiciona o diretório do projeto ao sys.path para imports locais
sys.path.append(os.path.dirname(__file__))

# Observação: fazemos importações tardias (dentro do bloco do botão) para
# evitar erros/avisos no carregamento inicial do Streamlit quando dependências
# estiverem faltando ou modelos não existirem.
main = None
predictor_regrecao = None
predictor_calssificacao = None

st.set_page_config(page_title="Atena IA", page_icon="🧠", layout="wide")
st.title("🧠 Atena IA — Análise de Dados e Previsões")

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Pré-visualização dos Dados")
    st.dataframe(df.head())

    # --- Rodar main.py automaticamente ---
    st.info("Executando Treinamento de IA...")
    if hasattr(main, "run_pipeline"):
        processed_data = main.run_pipeline(df)
    else:
        processed_data = df
    st.success("Treinamento de IA concluído.")

    # --- Escolha tipo de análise ---
    tipo_analise = st.radio("Tipo de Análise:", [
                            "Regressão", "Classificação"], horizontal=True)

    # --- Tabela de parâmetros editável ---
    st.subheader("🧩 Parâmetros para análise (1 linha)")
    input_data = pd.DataFrame([{col: "" for col in df.columns}])
    edited_data = st.data_editor(
        input_data, num_rows="fixed", use_container_width=True, key="param_editor")
    st.write("Parâmetros atuais:")
    st.dataframe(edited_data)

    # --- Botão Analisar / Prever ---
    if st.button("🚀 Analisar / Prever"):
        st.info(f"Rodando predição de {tipo_analise}...")

        # Importações tardias com tratamento de erro claro
        if main is None:
            try:
                import main as _main
                main = _main
            except Exception:
                st.warning(
                    "Não foi possível importar 'main'. Pré-processamento será pulado.")

        if tipo_analise == "Regressão":
            try:
                import predictor_regrecao as predictor_regrecao
            except Exception as e:
                st.warning(
                    f"Não foi possível importar o módulo de regressão: {type(e).__name__} {e}")
                predictor_regrecao = None

            if predictor_regrecao and hasattr(predictor_regrecao, "predict"):
                try:
                    resultado = predictor_regrecao.predict(edited_data)
                    st.success("Regressão concluída!")
                    st.dataframe(resultado)
                except FileNotFoundError as e:
                    st.error(f"Modelos não encontrados: {e}")
                except Exception as e:
                    st.error(
                        f"Erro ao executar predição de regressão: {type(e).__name__} {e}")
            else:
                st.warning(
                    "Função 'predict' não encontrada no predictor de regressão.")
        else:
            try:
                import predictor_calssificacao as predictor_calssificacao
            except Exception as e:
                st.warning(
                    f"Não foi possível importar o módulo de classificação: {type(e).__name__} {e}")
                predictor_calssificacao = None

            if predictor_calssificacao and hasattr(predictor_calssificacao, "predict"):
                try:
                    resultado = predictor_calssificacao.predict(edited_data)
                    st.success("Classificação concluída!")
                    # Exibir apenas a coluna de classe prevista e as probabilidades formatadas (_pct)
                    pct_cols = [
                        c for c in resultado.columns if c.endswith('_pct')]
                    display_cols = []
                    if 'predicao_classe' in resultado.columns:
                        display_cols.append('predicao_classe')
                    display_cols += pct_cols

                    # Exibir o DataFrame completo (valores brutos) diretamente
                    st.dataframe(resultado)
                except Exception as e:
                    st.error(
                        f"Erro ao executar predição de classificação: {type(e).__name__} {e}")
            else:
                st.warning(
                    "Função 'predict' não encontrada no predictor de classificação.")
else:
    st.info("Faça upload de um arquivo CSV para iniciar a análise.")
