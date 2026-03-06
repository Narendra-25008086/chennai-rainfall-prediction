import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("../data/chennai_rainfall_data.csv")

data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

X = data[['year','month','day']]
y = data['Rainfall_mm_day']

model = RandomForestRegressor(n_estimators=100)

model.fit(X,y)

joblib.dump(model,"../model.pkl")

print("Model trained successfully")