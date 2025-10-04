import importlib
import traceback
import sys
import os
sys.path.append(os.path.dirname(__file__))
try:
    importlib.import_module('predictor_regrecao')
    print('import predictor_regrecao OK')
except Exception as e:
    print('IMPORT ERROR:', type(e).__name__, e)
    traceback.print_exc()
