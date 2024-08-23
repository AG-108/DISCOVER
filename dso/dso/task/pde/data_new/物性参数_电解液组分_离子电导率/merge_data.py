import numpy as np
import pandas as pd

X_raw = []
y_raw = []

temps = [20, 30]

for i in range(2, 6):
    for (j, t) in enumerate(temps):
        path = f'./EC_EMC_{i}_{10-i}_weight+LiPF6_{t}°C.csv'
        data = pd.read_csv(path)
        X = data['LiPF6浓度 mol/kg'].values
        col_one = np.ones((X.shape[0],1))
        X = np.hstack((col_one*i, col_one*t, X.reshape(-1,1)))
        X_raw.append(X)
        y = data['离子电导率 mS/cm'].values
        y_raw.append(y)

X = np.vstack(X_raw)
y = np.hstack(y_raw)

pd.DataFrame(X).to_csv('X.csv', index=False, header=False)
pd.DataFrame(y).to_csv('y.csv', index=False, header=False)