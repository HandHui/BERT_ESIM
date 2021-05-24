import pandas as pd


### 拼接A/B两个csv
dataa = pd.read_csv('Aresult.csv')
datab = pd.read_csv('Bresult.csv')
c = pd.concat([dataa,datab],axis=0)
c.to_csv('result.csv',index=False)