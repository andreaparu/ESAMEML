import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
import joblib
warnings.filterwarnings('ignore')


def main():
    st.text("Esame Machine learning  - regressione case")
    newmodel = joblib.load('regressionecase.pkl')

    x1 = st.slider('CRIM 1', min_value=0, max_value=10, value=5)
    x2 = st.slider('ZN 2', min_value=0, max_value=10, value=5)
    x3 = st.slider('INDUS 3', min_value=0, max_value=10, value=5)
    x4 = st.slider('CHAS 4', min_value=0, max_value=10, value=5)
    x5 = st.slider('NOX 5', min_value=0, max_value=10, value=5)
    x6 = st.slider('RM 6', min_value=0, max_value=10, value=5)
    x7 = st.slider('AGE 7', min_value=0, max_value=10, value=5)
    x8 = st.slider('DIS 8', min_value=0, max_value=10, value=5)
    x9 = st.slider('RAD 9', min_value=0, max_value=10, value=5)
    x10 = st.slider('TAX 10', min_value=0, max_value=10, value=5)
    x11 = st.slider('PTRATIO 11', min_value=0, max_value=10, value=5)
    x12 = st.slider('B 12', 1, 500, 1)
    x13 = st.slider('ISTAT 13', min_value=0, max_value=10, value=5)
    prediction = newmodel.predict([[x1,x2,x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]])
    

    st.write('IL TUO PRICE è')
    st.write(round(prediction[0], 2))


if __name__ == "__main__":
    main()