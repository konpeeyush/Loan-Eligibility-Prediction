import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def marks_prediction(marks):
    X = pd.read_csv("score.csv",usecols=["Hours"])
    Y = pd.read_csv("score.csv",usecols=["Scores"])

    X=X.values
    Y=Y.values

    model=LinearRegression()
    model.fit(X,Y)

    X_test=np.array(marks)
    X_test=X_test.reshape(1,-1)

    return model.predict(X_test)
