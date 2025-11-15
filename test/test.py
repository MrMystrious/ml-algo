import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mlAlgo.linear_regression import LinearRegression  

data = pd.read_csv(r"data\house_price_regression_dataset.csv")

train = data.iloc[:-100]
test = data.iloc[-100:]

x_train = train[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
                 "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]].values
y_train = train[["House_Price"]].values

x_test = test[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
               "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]].values
y_test = test[["House_Price"]].values

scaler_X = StandardScaler()
x_train_scaled = scaler_X.fit_transform(x_train)
x_test_scaled = scaler_X.transform(x_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

li = LinearRegression(x_train_scaled, y_train_scaled, lr=0.01)
li.train(iter=100, batch=50)  

y_pred_scaled = li.predict(x_test_scaled).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled)  

ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - ss_res / ss_tot
print(f"Model accuracy ( score): {r2:.4f}")

comparison = np.hstack([y_pred, y_test])
print(comparison[:10]) 
