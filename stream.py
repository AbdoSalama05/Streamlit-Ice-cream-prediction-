import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"IceCreamSales-temperatures.csv")

X = df[["Temperature"]]
y = df["Ice Cream Profits"]

model = LinearRegression()
model.fit(X, y)

st.title("Ice Cream Profit Prediction")

temp = st.number_input("Temperature")

if st.button("Predict"):
    profit = model.predict([[temp]])[0]
    st.success(f"Predicted Profit = {profit:.2f}")
