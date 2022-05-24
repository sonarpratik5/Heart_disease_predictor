import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


@st.cache
def get_data(filename):
    heart_data = pd.read_csv(filename)
    return heart_data


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
prediction = st.container()
results = st.container()

with header:
    st.title('Welcome to our AI Expert system')
    st.text("In this project we will train and test the dataset to determine whether a person has Heart Disease based "
            "of Medical Data")
    st.markdown('##')
    st.markdown('##')

with dataset:
    st.header("Heart Disease Dataset")
    st.text("I Downloaded this dataset from Kaggle")

    heart_data = get_data('.\data\data.csv')
    st.write(heart_data.head(7))
    st.markdown('##')
    st.markdown('##')
    st.subheader("Graphical Representation of trestbps column")
    trestbps_visual = pd.DataFrame(heart_data['trestbps'].value_counts().head(20))
    st.bar_chart(trestbps_visual)
    st.markdown('##')
    st.markdown('##')

with features:
    st.header("features of the dataset")
    st.write(heart_data.columns)
    st.markdown('##')
    st.markdown('##')

with modelTraining:
    st.header("Logistic Regression")
    st.text("We used Logistic Regression to train and test the model")

    sel_col, disp_col = st.columns(2)
    sel_col.markdown(
        'Logistic regression is a process of modeling the probability of a discrete outcome given an input '
        'variable. The most common logistic regression models a binary outcome; something that can take two '
        'values such as true/false, yes/no, and so on. Multinomial logistic regression can model scenarios '
        'where there are more than two possible discrete outcomes. Logistic regression is a useful analysis '
        'method for classification problems, where you are trying to determine if a new sample fits best '
        'into a category.')

    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train.values, Y_train.values)
    X_train_prediction = model.predict(X_train.values)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train.values)
    disp_col.write('Accuracy of training data: ')
    disp_col.write(training_data_accuracy)

    X_test_prediction = model.predict(X_test.values)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test.values)
    disp_col.write('Accuracy of testing data: ')
    disp_col.write(test_data_accuracy)
    st.markdown('##')
    st.markdown('##')

with prediction:
    st.header("Heart Disease prediction System")
    st.text("We used Logistic Regression to train and test the model")

    sel_col, disp_col = st.columns(2)

    form = st.form(key='my_form')
    age = int(form.number_input('Age', step=1))
    sex1 = form.selectbox('Enter your Gender', options=['Male', 'female'])
    if sex1 == 'Male':
        sex = 1
    else:
        sex = 0

    cp = form.number_input('Chest Pain Type', step=1)

    trestbps = form.slider('resting blood pressure', min_value=80, max_value=200, value=120,
                           step=5)
    chol = int(form.number_input('serum cholestoral in mg/dl'))
    fbs = form.number_input('fasting blood sugar > 120 mg/dl', min_value=0, max_value=1)

    restecg = int(form.selectbox('resting electrocardiographic results (values 0,1,2)', options=[0, 1, 2]))
    thalach = int(form.slider('maximum heart rate achieved', min_value=50, max_value=250, value=120))
    exang = form.number_input('exercise induced angina', min_value=0, max_value=1)

    oldpeak = int(form.number_input('ST depression induced by exercise relative to rest', min_value=0, max_value=5))
    slope = int(form.selectbox('the slope of the peak exercise ST segment', options=[0, 1, 2]))
    ca = int(form.number_input('number of major vessels (0-3) colored by flourosopy', step=1, max_value=5, min_value=0))

    thal = int(
        form.number_input('number of major vessels (0-3) colored by flourosopy', step=1, max_value=3, min_value=0))
    submit_button = form.form_submit_button(label='Submit')

with results:
    st.header("Result Analysis")
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_np = np.asarray(input_data)
    input_data_reshaped = input_data_np.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    st.write('The result of the prediction is')

    if prediction[0] == 0:
        st.subheader('the person is free of heart disease')

    else:
        st.subheader('the person has heart disease')
