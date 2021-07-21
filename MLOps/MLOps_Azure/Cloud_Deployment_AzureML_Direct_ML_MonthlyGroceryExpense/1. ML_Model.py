import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("Location/Expenses_Prediction.csv")

scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)
X = df.iloc[:,:2]
Y= df.iloc[:,-1]

lr_model = LinearRegression()
lr_model.fit(X, Y)

Y_predict = lr_model.predict(X)
r2 = r2_score(Y, Y_predict)
print('R2 score is {}'.format(r2))

joblib.dump(lr_model, 'monthlyprediction.joblib')