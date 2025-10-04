import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    import predictor_calssificacao as pc
    df = pd.DataFrame({
        'pH': [6.0],
        '% Sólidos': [35.0],
        'Dosagem (g/t)': [20.0],
        'Turbidez (FTU)': [150.0]
    })
    print('Chamando pc.predict...')
    res = pc.predict(df)
    print('\nResultado da predição:')
    print(res)
except Exception as e:
    print('ERRO:', type(e).__name__, e)
