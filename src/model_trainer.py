# src/model_trainer.py

# Adicione estas importações no início do arquivo
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Mantenha as outras importações e a função train_transparent_tree_ensemble
from src.preprocessor import create_preprocessor
from sklearn.tree import DecisionTreeRegressor
import config


def train_naive_bayes_classifier(X, y):
    """Módulo 1: Treina um modelo Naive Bayes para classificação usando um pipeline robusto."""
    print("\n--- Treinando Módulo 1: Naive Bayes (Classificação) ---")
    
    # Criamos um pipeline que executa 3 passos em sequência:
    # 1. Encoder: Converte categorias para números, gerando NaN para desconhecidos.
    # 2. Imputer: Substitui os NaN pela categoria mais frequente ("moda") que viu no treino.
    # 3. Model: O modelo final que recebe os dados já limpos.
    
    classification_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('model', CategoricalNB())
    ])
    
    # Treinamos o pipeline inteiro com os dados de treino
    classification_pipeline.fit(X, y)
    
    print("Pipeline de Classificação Naive Bayes treinado com sucesso.")
    # Agora retornamos o pipeline inteiro. Ele lida com o encoding e a imputação internamente.
    return classification_pipeline

# A função train_transparent_tree_ensemble permanece a mesma
def train_transparent_tree_ensemble(X, y):
    """Módulo 2: Treina um ensemble transparente de Árvores de Decisão para regressão."""
    print("\n--- Treinando Módulo 2: Ensemble de Árvores (Regressão) ---")
    
    preprocessor = create_preprocessor(X)

    pipeline_tree_1 = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=config.TREE_MAX_DEPTH_1, random_state=config.RANDOM_STATE))
    ])

    pipeline_tree_2 = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=config.TREE_MAX_DEPTH_2, random_state=config.RANDOM_STATE))
    ])

    pipeline_tree_1.fit(X, y)
    pipeline_tree_2.fit(X, y)

    print("Ensemble de Árvores de Decisão treinado com sucesso.")
    return pipeline_tree_1, pipeline_tree_2