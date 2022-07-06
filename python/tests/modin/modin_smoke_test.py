import os
os.environ['MODIN_STORAGE_FORMAT'] = 'omnisci'
import modin.pandas as pd
a = pd.DataFrame([[1, 11]], columns=['a', 'b'])
a['a'] = a['a'] + a['b']
print(a['a'])