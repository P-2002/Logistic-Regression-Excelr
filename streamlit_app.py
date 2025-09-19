import streamlit as st
import pandas as pd
import pickle

#Load Model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Titanic Survival Prediction App')

Pclass = st.selectbox('Passenger Class(1=1st, 2=2nd, 3=3rd)', [1,2,3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 25)
SibSp = st.number_input('Number of Siblings/Spouses Abroad', 0,10,0)
Parch = st.number_input('Number of Parents/Children Abroad', 0,10,0)
Fare = st.number_input('Passenger Fare', min_value=0.0, max_value=600.0, value=50.0)
Embarked = st.selectbox('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)',['C','Q','S'])

#Encode Inputs
Sex = 0 if Sex =='male' else 1
Embarked_dict = {'C': 0, 'Q': 1, 'S': 2}[Embarked]

features = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [Embarked]
})

#Prediction Button
if st.button('Predict'):
    prediction = model.predict(features)
    result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
    st.subheader(f'Prediction: {result}')