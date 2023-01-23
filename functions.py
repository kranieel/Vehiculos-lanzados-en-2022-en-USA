import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

def scattergraph(df,col_x,col_y,company):
    '''
    Ésta función recibe 4 parámetros. 
    Un DF, la columna que será el valor X, la columna que será el valor Y, y el nombre de la compañía.
    Con éstas 4 variables la función devolverá un scatterplot.
    '''
    #El tamaño de el gráfico
    plt.figure(figsize=(9, 6))
    scatter = sns.scatterplot(x=df[col_x], y=df[col_y], s=100)

    #Le ponemos titulos
    plt.title(f"Precio según los caballos de coches {company} en 2022", size=18)
    plt.xlabel("Caballos", size=14)
    plt.ylabel("Precio", size=14)

    plt.show()

def scattergraphregresion(df,col_x,col_y,company):
    '''
    Ésta función recibe 4 parámetros. 
    Un DF, la columna que será el valor X, la columna que será el valor Y, y el nombre de la compañía.
    Con éstas 4 variables la función devolverá un scatterplot con un modelo de regresión lineal.
    La linea roja representará la predicción y las verdes la desviación estandar hacía arriba y abajo.
    '''
    #Definimos la X e Y.
    x=df[col_x]
    y=df[col_y]
    #Hacemos el modelo de Regresión.
    N = len(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    cov_xy = (1 / (N-1)) * np.sum((x - mean_x) * (y - mean_y))
    beta_1_hat = cov_xy / var_x
    beta_0_hat = mean_y - beta_1_hat * mean_x
    y_pred = beta_0_hat + beta_1_hat * x

    sigma_square_hat = (1 / N) * np.sum((y - y_pred)**2)
    sigma_hat = np.sqrt(sigma_square_hat)

    #Visualizamos el gráfico con la regresión lineal
    plt.figure(figsize=(9, 6))
    scatter = sns.scatterplot(x=x, y=y, s=100)

    #Linea de regresión
    x_line = np.linspace(min(x)-30, max(x)+30, 100) #Valor minimo - valor máximo - 100 intervalos
    y_line = beta_0_hat + beta_1_hat * x_line #Calculamos los valores de Y

    #label es las etiquetas - leyenda en el grafico
    sns.lineplot(x=x_line, y=y_line, color='red', label=r"$y_{pred} = \beta_0 + \beta_1 x$") #Linea de Regresión roja
    sns.lineplot(x=x_line, y=y_line + sigma_hat, linestyle='--', color='green', label=r"$y_{pred} \pm \sigma$") # "+ sigma_hat" una desviación estandar por encima.
    sns.lineplot(x=x_line, y=y_line - sigma_hat, linestyle='--', color='green') # "- sigma_hat" una desviación estandar por debajo.

    #Los titulos del gráfico.
    plt.title(f"Precio según los caballos de coches {company} en 2022", size=18)
    plt.xlabel("Caballos", size=14)
    plt.ylabel("Precio", size=14)

    plt.show()