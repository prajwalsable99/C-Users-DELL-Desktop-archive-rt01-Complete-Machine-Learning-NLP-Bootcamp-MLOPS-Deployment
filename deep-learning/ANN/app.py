import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf

import joblib

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler



from tensorflow.keras.models import load_model

le=joblib.load('models/genderEncoder.pkl')

ohe=joblib.load('models/oheEncoder.pkl')

scalar=joblib.load('models/scalar.pkl')

model=load_model('models/model.h5')

#app
st.title("Customer Churn Prediction")


CreditScore= st.number_input('credit score')
Age=st.slider('Age',18,90)
Tenure=st.slider('Tenure',0,10)
Balance= st.number_input('Balance')
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('Has credit card', [0,1])
IsActiveMember=st.selectbox('IsActiveMember', [0,1])
EstimatedSalary=st.number_input('EstimatedSalary')

sex=st.selectbox('sex',le.classes_)
geography= st.selectbox('Geography',ohe.categories_[0])



sex_en= le.transform([sex])

geography_en=ohe.transform([[geography]])[0]


inps= [CreditScore, Age, Tenure, Balance, NumOfProducts,HasCrCard,IsActiveMember, EstimatedSalary ,sex_en,geography_en ] 

x=np.hstack(inps).reshape(1,-1)

x_sc=scalar.transform(x)
print(x)


st.markdown("prediction here")
st.divider()
pred=model.predict(x)[0][0]
if pred>0.5 :
    st.write("churn will exit")
else:
    st.write("churn will not exit")