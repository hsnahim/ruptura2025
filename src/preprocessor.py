# src/preprocessor.py
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(dataframe):
    """
    Cria um ColumnTransformer para pré-processar dados numéricos e categóricos.
    Detecta automaticamente os tipos de coluna a partir do dataframe.
    """
    numerical_features = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_features = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Colunas numéricas identificadas: {numerical_features}")
    print(f"Colunas categóricas identificadas: {categorical_features}")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor