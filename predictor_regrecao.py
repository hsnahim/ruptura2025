# predictor_regressao.py
import os
import pandas as pd
import joblib
import config


def _carregar_modelos():
    """Carrega os dois modelos de regressão salvos e retorna (tree_1, tree_2)."""
    path_tree_1 = os.path.join(config.MODEL_SAVE_PATH, 'decision_tree_1.pkl')
    path_tree_2 = os.path.join(config.MODEL_SAVE_PATH, 'decision_tree_2.pkl')

    if not os.path.exists(path_tree_1) or not os.path.exists(path_tree_2):
        raise FileNotFoundError(
            "Um ou ambos os arquivos de modelo de regressão não foram encontrados.")

    tree_1 = joblib.load(path_tree_1)
    tree_2 = joblib.load(path_tree_2)

    return tree_1, tree_2


def predict(entrada_df: pd.DataFrame) -> pd.DataFrame:
    """
    Função esperada pelo front-end (Streamlit):
    - entrada_df: DataFrame com as colunas de entrada (1 linha é o caso típico)
    - retorna um DataFrame com colunas: 'predicao_árvore_1', 'predicao_árvore_2', 'predicao_regressao'

    Lança exceções claras em caso de erro (FileNotFoundError, TypeError, RuntimeError).
    """
    if not isinstance(entrada_df, pd.DataFrame):
        raise TypeError("entrada_df deve ser um pandas.DataFrame")

    # Carregar modelos
    try:
        tree_1, tree_2 = _carregar_modelos()
    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar os modelos: {e}")

    # Prever
    try:
        pred1 = tree_1.predict(entrada_df)
        pred2 = tree_2.predict(entrada_df)
    except Exception as e:
        raise RuntimeError(f"Erro durante a predição: {e}")

    pred_final = (pred1 + pred2) / 2

    # Incluir coluna com nome do target (ex.: 'Vs (m/h)') contendo o valor previsto
    result_df = pd.DataFrame({
        'predicao_árvore_1': pd.Series(pred1).ravel(),
        'predicao_árvore_2': pd.Series(pred2).ravel(),
        'predicao_regressao': pd.Series(pred_final).ravel()
    })

    target_name = getattr(config, 'TARGET_REGRESSION', 'predicao')
    # adicionar coluna com o mesmo valor da predição final e com o nome do target
    result_df[target_name] = pd.Series(pred_final).ravel()

    return result_df


def testar_regressao():
    """Teste local que usa a função predict com um exemplo fixo."""
    print("--- Testando Modelo de Regressão Treinado ---")

    situacao_para_prever = pd.DataFrame({
        'pH': [6.0],
        '% Sólidos': [35.0],
        'Dosagem (g/t)': [20.0],
        'Turbidez (FTU)': [150.0]
    })

    try:
        resultado = predict(situacao_para_prever)
        print("\n--- Resultado da Previsão ---")
        print(resultado)
    except Exception as e:
        print(f"Erro no teste de regressão: {e}")


if __name__ == "__main__":
    testar_regressao()
