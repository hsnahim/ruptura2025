# predictor.py (VERSÃO CORRIGIDA E ATUALIZADA)
import pandas as pd
import joblib
import config
import os


def _carregar_pipeline():
    pipeline_path = os.path.join(
        config.MODEL_SAVE_PATH, 'classification_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Pipeline de classificação não encontrado em '{pipeline_path}'")
    return joblib.load(pipeline_path)


def predict(entrada_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame (1 linha) e retorna um DataFrame com a classe prevista
    e probabilidades por classe (se disponíveis no pipeline).
    """
    if not isinstance(entrada_df, pd.DataFrame):
        raise TypeError("entrada_df deve ser um pandas.DataFrame")

    pipeline = _carregar_pipeline()

    # mapeamento solicitado para rótulos de risco
    label_map = {
        'Falha': 'Risco Alto',
        'Talvez': 'Risco Moderado',
        'Sucesso': 'Risco Baixo'
    }

    # Remover colunas alvo se presentes (ex.: 'Resultado' ou 'Vs (m/h)')
    X = entrada_df.copy()
    for alvo in (config.TARGET_CLASSIFICATION, config.TARGET_REGRESSION):
        if alvo in X.columns:
            X = X.drop(columns=[alvo])

    # Tentar descobrir nomes de features que o pipeline espera
    expected = None
    try:
        expected = list(getattr(pipeline, 'feature_names_in_', None))
    except Exception:
        expected = None

    if expected is None:
        try:
            model = getattr(pipeline, 'named_steps', {}).get('model')
            expected = list(getattr(model, 'feature_names_in_', [])) or None
        except Exception:
            expected = None

    if expected is None:
        try:
            pre = getattr(pipeline, 'named_steps', {}).get('preprocessor')
            if pre is not None and hasattr(pre, 'get_feature_names_out'):
                try:
                    expected = list(pre.get_feature_names_out())
                except TypeError:
                    # Alguns transformers exigem input_features
                    expected = list(pre.get_feature_names_out(X.columns))
        except Exception:
            expected = None

    # Se sabemos as features esperadas, alinhar e remover colunas extras
    if expected is not None:
        # identificar colunas ausentes e extras
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]

        if missing:
            raise RuntimeError(
                f"Colunas esperadas ausentes para o modelo: {missing}")

        if extra:
            # remover colunas extras que o modelo não espera (ex.: Resultado, Vs (m/h))
            X = X[[c for c in expected if c in X.columns]]

    try:
        pred = pipeline.predict(X)
    except Exception as e:
        raise RuntimeError(f"Erro ao executar predict: {e}")

    # tentar obter probabilidades usando X (mesmas features que foram usadas em predict)
    probs = None
    classes = None
    try:
        probs = pipeline.predict_proba(X)
        classes = getattr(pipeline, 'classes_', None)
        if classes is None and hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            classes = pipeline.named_steps['model'].classes_
    except Exception:
        probs = None

    # Aplicar mapeamento na predição principal
    mapped_preds = [label_map.get(v, v) for v in pd.Series(pred).ravel()]
    out = pd.DataFrame({'predicao_classe': mapped_preds})

    # Adicionar apenas colunas percentuais (sem colunas numéricas brutas)
    if probs is not None:
        if classes is not None:
            for idx, cls in enumerate(classes):
                mapped = label_map.get(cls, cls)
                pct = (probs[:, idx] * 100).round(2)
                out[f'prob_{mapped}_pct'] = pct.astype(str) + '%'
        else:
            for idx in range(probs.shape[1]):
                pct = (probs[:, idx] * 100).round(2)
                out[f'prob_{idx}_pct'] = pct.astype(str) + '%'

    return out


def testar_situacao():
    print("--- Testando Modelo de Classificação Treinado ---")
    try:
        situacao_para_prever = pd.DataFrame({
            'pH': [6.0],
            '% Sólidos': [35.0],
            'Dosagem (g/t)': [20.0],
            'Turbidez (FTU)': [150.0]
        })
        print(predict(situacao_para_prever))
    except Exception as e:
        print(f"Erro no teste de classificação: {e}")


if __name__ == "__main__":
    testar_situacao()
