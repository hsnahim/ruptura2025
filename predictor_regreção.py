# predictor_regressao.py
import pandas as pd
import joblib
import config

def testar_regressao():
    """
    Carrega o ensemble de árvores de decisão treinado e prevê o resultado
    para uma nova situação.
    """
    print("--- Testando Modelo de Regressão Treinado ---")
    
    # 1. Carregar os dois pipelines de regressão salvos
    try:
        path_tree_1 = config.MODEL_SAVE_PATH + 'decision_tree_1.pkl'
        path_tree_2 = config.MODEL_SAVE_PATH + 'decision_tree_2.pkl'
        
        tree_1 = joblib.load(path_tree_1)
        tree_2 = joblib.load(path_tree_2)
        
        print(f"Modelos de regressão carregados com sucesso de '{config.MODEL_SAVE_PATH}'")
    except FileNotFoundError:
        print(f"ERRO: Modelos de regressão não encontrados.")
        print("Por favor, execute 'python main.py' com a opção 2 para treinar e salvar os modelos primeiro.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os modelos: {e}")
        return

    # 2. Definir a situação para prever (com as colunas do dataset gerado)
    # As colunas de entrada para a regressão são as mesmas da classificação
    situacao_para_prever = pd.DataFrame({
        'pH': [6.0],
        '% Sólidos': [35.0],
        'Dosagem (g/t)': [20.0],
        'Turbidez (FTU)': [150.0]
    })

    print("\nSituação a ser prevista:")
    print(situacao_para_prever)

    # 3. Fazer a predição com cada árvore
    try:
        predicao_1 = tree_1.predict(situacao_para_prever)
        predicao_2 = tree_2.predict(situacao_para_prever)
        
        # 4. Calcular a média para a predição final do ensemble
        predicao_final = (predicao_1 + predicao_2) / 2

        print("\n--- Resultado da Previsão ---")
        print(f"Previsão da Árvore 1: {predicao_1[0]:.2f} (m/h)")
        print(f"Previsão da Árvore 2: {predicao_2[0]:.2f} (m/h)")
        print(f"Previsão Final (Média do Ensemble): {predicao_final[0]:.2f} (m/h)")

    except Exception as e:
        print(f"\nERRO DURANTE A PREDIÇÃO: {e}")
        print("Verifique se as colunas em 'situacao_para_prever' estão corretas.")

if __name__ == "__main__":
    testar_regressao()
