import pandas as pd
import torch
import numpy as np
from typing import List, Annotated

class LinearRegression:
    def __init__(self, features: np.ndarray, y_labels: np.ndarray, lr: float):
        self.__xraw = torch.tensor(features, dtype=torch.float32)
        self.__y = torch.tensor(y_labels, dtype=torch.float32).reshape(-1, 1)
        self.__lr = lr

        self.inp_fea_size = self.__xraw.size(1)  # (num_samples, num_features)
        self.no_inps = self.__xraw.size(0)
        self.__X = torch.cat([torch.ones(self.no_inps, 1), self.__xraw], dim=1)  # add bias
        self.__theta = torch.rand(self.inp_fea_size + 1, 1, dtype=torch.float32)

    def batch_inp(self, batch: int):
        if self.no_inps % batch > 0:
            print(f"batch should be divisible {self.no_inps} {batch}")
        st = 0
        while st + batch <= self.no_inps:
            yield self.__X[st:st + batch], self.__y[st:st + batch]
            st += batch
        if st < self.no_inps:
            yield self.__X[st:], self.__y[st:]

    def error(self, batch: int, y: Annotated[torch.Tensor, "outputs"], y_: Annotated[torch.Tensor, "outputs"]):
        m = y.size(0)
        e = 0.5 / m * ((y_ - y).T @ (y_ - y))
        return e

    def update_theta(self, batch: int, x: Annotated[torch.Tensor, "outputs"], y: Annotated[torch.Tensor, "outputs"], y_: Annotated[torch.Tensor, "outputs"]):
        m = y.size(0)
        dj = (self.__lr / m) * (x.T @ (y_ - y))
        self.__theta -= dj

    def train(self, iter: int, batch: int):
        for it in range(iter):
            j = 0
            for x_batch, y_batch in self.batch_inp(batch):
                y_ = x_batch @ self.__theta
                j += self.error(batch, y_batch, y_)
                self.update_theta(batch, x_batch, y_batch, y_)
            print(f"iteration {it+1} -> error {j.item():.6f}")

    def predict(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        X_bias = torch.cat([torch.ones(x_tensor.size(0), 1), x_tensor], dim=1)
        return X_bias @ self.__theta

    def dis(self):
        print(self.__theta)

    def score(self, x: np.ndarray, y_true: np.ndarray):

        y_pred = self.predict(x).detach().numpy()
        y_true = np.array(y_true).reshape(-1, 1)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2 * 100

""" if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv(r"ml-algo\data\house_price_regression_dataset.csv")
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    x_train = train[["Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built",
                    "Lot_Size","Garage_Size","Neighborhood_Quality"]].values
    y_train = train[["House_Price"]].values

    x_test = test[["Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built",
                "Lot_Size","Garage_Size","Neighborhood_Quality"]].values
    y_test = test[["House_Price"]].values

    scaler_X = StandardScaler()
    x_train_scaled = scaler_X.fit_transform(x_train)
    x_test_scaled = scaler_X.transform(x_test)

    li = LinearRegression(x_train_scaled, y_train, lr=0.01)
    li.train(20, 50)

    accuracy = li.score(x_test_scaled, y_test)
    print(f"Model accuracy : {accuracy:.2f}%") """
