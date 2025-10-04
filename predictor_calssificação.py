# predictor.py (VERSÃO CORRIGIDA E ATUALIZADA)
import pandas as pd
import joblib
import config

def testar_situacao():
    """
    Carrega o pipeline de CLASSIFICAÇÃO treinado e prevê o resultado
    para uma nova situação.
    """
    print("--- Testando Modelo de Classificação Treinado ---")
    
    # 1. Carregar o pipeline de CLASSIFICAÇÃO salvo pelo main.py
    try:
        # O nome do arquivo salvo na opção 1 do main.py é 'classification_pipeline.pkl'
        pipeline_path = config.MODEL_SAVE_PATH + 'classification_pipeline.pkl'
        classification_pipeline = joblib.load(pipeline_path)
        print(f"Pipeline de classificação carregado com sucesso de '{pipeline_path}'")
    except FileNotFoundError:
        print(f"ERRO: Pipeline não encontrado em '{pipeline_path}'.")
        print("Por favor, execute 'python main.py' com a opção 1 para treinar e salvar o modelo primeiro.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o modelo: {e}")
        print("O arquivo pode estar corrompido ou ser de um modelo de regressão.")
        return

    # 2. Definir a situação para prever (com as colunas do novo dataset)
    # As colunas devem ser as mesmas usadas no TREINO do modelo de classificação.
    situacao_para_prever = pd.DataFrame({
        'pH': [6.0],
        '% Sólidos': [35.0],
        'Dosagem (g/t)': [20.0],
        'Turbidez (FTU)': [150.0]
    })

    print("\nSituação a ser prevista:")
    print(situacao_para_prever)

    # 3. Fazer a predição
    try:
        predicao = classification_pipeline.predict(situacao_para_prever)
        probabilidades = classification_pipeline.predict_proba(situacao_para_prever)

        classes_aprendidas = classification_pipeline.named_steps['model'].classes_

        print("\n--- Resultado da Previsão ---")
        print(f"Resultado Previsto: {predicao[0]}")
        
        print("\nProbabilidades calculadas para cada classe:")
        for i, class_name in enumerate(classes_aprendidas):
            print(f"  - {class_name}: {probabilidades[0][i]:.2%}")

    except AttributeError:
        print("\nERRO CRÍTICO: O modelo carregado não é um modelo de classificação.")
        print("Ele não possui o método 'predict_proba'.")
        print("Certifique-se de que o arquivo 'classification_pipeline.pkl' contém o modelo correto.")
    except ValueError as e:
        print(f"\nERRO DE DADOS: As colunas na 'situação para prever' não batem com as que o modelo espera.")
        print(f"Detalhe do erro: {e}")


if __name__ == "__main__":
    testar_situacao()