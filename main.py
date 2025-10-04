# main.py (VERSÃO FINAL CORRIGIDA)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, r2_score

import config
from src.data_loader import load_data
from src.model_trainer import train_naive_bayes_classifier, train_transparent_tree_ensemble

def main():
    print("Bem-vindo ao sistema de treinamento e VALIDAÇÃO de modelos transparentes!")
    choice = input("Qual módulo você deseja treinar e validar?\n1. Classificação (Naive Bayes)\n2. Regressão (Ensemble de Árvores)\nEscolha (1 ou 2): ")

    df = load_data(config.DATA_PATH)
    if df is None:
        return

    if choice == '1':
        target = config.TARGET_CLASSIFICATION
        X = df.drop(columns=[target, config.TARGET_REGRESSION])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=config.RANDOM_STATE)

        # AGORA RECEBEMOS APENAS O PIPELINE, QUE CONTÉM TUDO
        classification_pipeline = train_naive_bayes_classifier(X_train, y_train)
        
        print("\n--- Validando o Modelo de Classificação ---")
        # USAMOS O PIPELINE DIRETAMENTE PARA PREDIZER. ELE FAZ O ENCODING E IMPUTAÇÃO INTERNAMENTE.
        y_pred = classification_pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=y.unique()) # Adicionado 'labels' para garantir consistência
        report = classification_report(y_test, y_pred, zero_division=0) # Adicionado 'zero_division' para evitar warnings

        print(f"Acurácia no conjunto de teste: {acc:.2f}")
        print("Matriz de Confusão:")
        print(cm)
        print("Relatório de Classificação:")
        print(report)

        # SALVAMOS O PIPELINE INTEIRO
        joblib.dump(classification_pipeline, config.MODEL_SAVE_PATH + 'classification_pipeline.pkl')
        print(f"\nPipeline de classificação salvo em '{config.MODEL_SAVE_PATH}'")

    elif choice == '2':
        # Esta parte não precisa de alterações
        target = config.TARGET_REGRESSION
        X = df.drop(columns=[target, config.TARGET_CLASSIFICATION])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)

        tree_1, tree_2 = train_transparent_tree_ensemble(X_train, y_train)

        print("\n--- Validando o Modelo de Regressão ---")
        pred_1 = tree_1.predict(X_test)
        pred_2 = tree_2.predict(X_test)
        final_pred = (pred_1 + pred_2) / 2

        mae = mean_absolute_error(y_test, final_pred)
        r2 = r2_score(y_test, final_pred)

        print(f"Erro Médio Absoluto (MAE) no conjunto de teste: {mae:.2f}")
        print(f"R² (Coeficiente de Determinação) no conjunto de teste: {r2:.2f}")

        joblib.dump(tree_1, config.MODEL_SAVE_PATH + 'decision_tree_1.pkl')
        joblib.dump(tree_2, config.MODEL_SAVE_PATH + 'decision_tree_2.pkl')
        print(f"\nModelos de regressão salvos em '{config.MODEL_SAVE_PATH}'")

    else:
        print("Opção inválida.")

if __name__ == "__main__":
    main()