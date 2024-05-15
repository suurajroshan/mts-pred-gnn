import pandas as pd
import numpy as np

# Import the CSV file
df = pd.read_csv('histcorr06.csv', header=None)
print(df.shape)

dfn = df- np.identity(df.shape[0])
dfn = dfn.astype(int)

# Save the new matrix as a CSV file
dfn.to_csv('histcorr06_new.csv', index=False, header=False, sep=',')

dfnew = pd.read_csv('histcorr06_new.csv', header=None)
print(dfnew.shape)