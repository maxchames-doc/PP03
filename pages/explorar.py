import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset('iris')
registros = len(iris)
atributos = iris.shape
tipos = iris.dtypes.astype(str)
estadistico = iris.describe()
categorias = iris.species.unique()
g = sns.pairplot(iris, hue='species')

def show_explore_page():
    st.title("Explorador de la fuente de  datos")
    st.subheader("Vista general del dataset")
    col1, col2 = st.columns(2)
    col1.metric("Registros", registros)
    col2.metric("Atributos", atributos[1]-1)

    st.subheader("Estructura de datos del dataset")
    st.write(tipos)

    st.subheader("Valores de la variable categórica")
    st.dataframe(categorias)

    st.subheader("Resumen estadístico")
    st.dataframe(estadistico)

    st.subheader("Vista detallada de los registros")
    st.dataframe(iris)

    st.write("""#### Relaciones entre atributos""")

    st.pyplot(g)


show_explore_page()
