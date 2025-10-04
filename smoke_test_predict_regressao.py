import pandas as pd
import sys
import os

# Garantir que o diretório do projeto esteja no path
sys.path.append(os.path.dirname(__file__))

try:
    import predictor_regrecao as pr
    df = pd.DataFrame({
        'pH': [6.0],
        '% Sólidos': [35.0],
        'Dosagem (g/t)': [20.0],
        'Turbidez (FTU)': [150.0]
    })
    print('Chamando pr.predict...')
    res = pr.predict(df)
    print('\nResultado da predição:')
    print(res)
except Exception as e:
    print('ERRO:', type(e).__name__, e)
