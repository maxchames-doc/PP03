import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('./model/guarda_modelo.pkl', 'rb') as file:
        data=pickle.load(file)
    return data


data = load_model()

modelo = data['model']

def show_predict_page():
    st.title("Predicción de Tipos de Flores (IRIS DATASET")

    st.write("""### Ingrese los parametros""")
    sepal_l = st.number_input("Largo del sépalo")
    sepal_w = st.number_input("Ancho del sépalo")
    petal_l = st.number_input("Largo del pétalo")
    petal_w = st.number_input("Ancho del pétalo")

    ok = st.button("Calcular el tipo de flor")
    if ok:
        X = np.array([[sepal_l,sepal_w,petal_l,petal_w]])
        y = modelo.predict(X)
        tipo = {0: 'setosa', 1: 'versicolor', 2:'virginica'}

        st.subheader(f"La flor es del tipo {tipo[y[0]]}")

show_predict_page()
