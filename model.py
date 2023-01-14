import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Read the data
data = pd.read_csv('train.csv')

# Drop the columns that are not needed
data.Gender = data.Gender.fillna('Male')
data.Married = data.Married.fillna('No')
data.Dependents = data.Dependents.fillna('0')
data.Self_Employed = data.Self_Employed.fillna('No')
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
data.Credit_History = data.Credit_History.fillna(1.0)

# Locating Values
X = data.iloc[:, 1: 12].values
y = data.iloc[:, 12].values

# Transforming the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Encoding the categorical data


labelEncoder_X = LabelEncoder()  # DUMP THIS

# copy this
for i in range(0, 5):
    X[:, i] = labelEncoder_X.fit_transform(X[:, i])
X[:, 10] = labelEncoder_X.fit_transform(X[:, 10])

# don't copy this
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# copy and dump this
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting the model
classifier = XGBClassifier()
classifier.fit(X, y)


# Saving the model
pickle.dump(classifier, open('classifier.pkl', 'wb'))
pickle.dump(labelEncoder_X, open('labelEncoder_X.pkl', 'wb'))
pickle.dump(sc, open('sc.pkl', 'wb'))
