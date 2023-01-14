from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

classifier = pickle.load(open('classifier.pkl', 'rb'))
labelEncoder_X = pickle.load(open('labelEncoder_X.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    creditHistory = request.form['creditHistory']
    selfEmployed = request.form['selfEmployed']
    education = request.form['education']
    married = request.form['married']
    coApplicantIncome = request.form['coApplicantIncome']
    propertyArea = request.form['propertyArea']
    dependents = request.form['dependents']
    loanAmountTerm = request.form['loanAmountTerm']
    loanAmount = request.form['loanAmount']
    gender = request.form['gender']
    applicantIncome = request.form['applicantIncome']

    dataSet = np.array(
        [gender, married, dependents, education, selfEmployed, applicantIncome, coApplicantIncome, loanAmount,
         loanAmountTerm, creditHistory, propertyArea])

    dataSet = dataSet.reshape(1, -1)
    labelencoder_X = LabelEncoder()
    for i in range(0, 5):
        dataSet[:, i] = labelencoder_X.fit_transform(dataSet[:, i])
    dataSet[:, 10] = labelencoder_X.fit_transform(dataSet[:, 10])

    dataSet = sc.fit_transform(dataSet)

    pred = classifier.predict(dataSet)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
