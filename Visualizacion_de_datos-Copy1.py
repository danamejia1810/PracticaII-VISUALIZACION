#!/usr/bin/env python
# coding: utf-8

# <div style="width: 100%; clear: both;">
# <div style="float: left; width: 50%;">
# <img src="http://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg", align="left">
# </div>
# <div style="float: right; width: 50%;">
# <p style="margin: 0; padding-top: 22px; text-align:right;">M2.859 - Visualización de datos</p>
# <p style="margin: 0; text-align:right;">2022-2 · Máster universitario en Ciencia de datos (Data science)</p>
# <p style="margin: 0; text-align:right; padding-button: 100px;">Estudios de Informática, Multimedia y Telecomunicación</p>
# </div>
# </div>
# <div style="width:100%;">&nbsp;</div>

# <div class="alert alert-block alert-info">
# <strong>Nombre y apellidos:</strong> Dayana Katherine Mejia Quintero
#     
# </div>

# 

# # A9: Creación de la visualización y entrega del proyecto (Práctica II): 
# 
# Está actividad está dividida en dos partes:
# 
#  - **[Código](#ej1)**:proceso de recolección y limpieza de los datos.
#  - **[Link de visualización](#ej2)**: Se muestra la página web de Tableau Public para la presentación de la visualización.

# <u>Descripción y enunciado</u>: 
# 
# En esta segunda parte de la práctica, el estudiante tendrá que desarrollar una visualización de datos que demuestre su conocimiento del campo, así como el uso de diferentes herramientas y técnicas, basadas en el conjunto de datos seleccionado y validado en la primera parte de la práctica.
# 
# <u>Esta práctica tiene como finalidad los siguientes objetivos</u>:
# 
# - Utilizar herramientas diversas y avanzadas para la creación de visualizaciones.
# - Crear un proyecto de visualización de datos con una estructura, diseño y contenido de calidad profesional.
# - Practicar diversos tipos de enfoques respecto a las preguntas clave a responder.
# - Comprender los elementos de interactividad que aportan valor a una visualización.
# 
# 

# ### Brecha de Género en el sector tecnológico: ¿Como crecen las mujeres?
# 
# 

# El dataset escogido para la realización de la práctica de la asignatura es el
# 2022 Kaggle Machine Learning & Data Science Survey. El motivo por el que se escoge
# este dataset es porque nos gustaría conocer la composición de las mujeres en el sector
# tecnológico que hacen parte de la plataforma de Kaggle y su contraste con la contraparte
# masculina para analizar la brecha de género. Actualmente es necesario abordar este
# tópico para poder identificar las variables que puedan impactar en el bajo porcentaje de
# la fuerza de trabajo de las mujeres y la disparidad que pueda existir en el salario para
# hacer medidas que permitan crear políticas para incrementar las oportunidades de
# mujeres en el campo.
# 
# El conjunto de datos es extraído de una entrevista realizada a personas que
# trabajan en el campo de Machine Learning desde 09/16/2022 a 10/16/2022 por la
# plataforma Kaggle para sus usuarios registrados. Estos datos son los más recientes de
# dicha entrevista y tiene un formato de estructurado abierto de tercer nivel que es el de
# CSV para poder ser descargado. También hace parte dos archivos en PDF que explican
# las preguntas realizadas en un formato de primer nivel lo que menciona los nombres de
# las variables. En su conjunto, el enfoque como tal de los datos es de conocer las personas
# que trabajan en los campos tecnológicos y diversas características añadidas a ella como las herramientas que manejan, de que país son, si trabajan remoto o presencial entre otras, lo que permite extraer diversas conclusiones a partir de su estudio. En nuestro caso, se hará un estudio con enfoque femenino para poder observar su representación
# en la fuerza de trabajo y los estudiantes que conforman el campo. 
# 
# La página web para descargar los archivos se encuentran en: https://www.kaggle.com/competitions/kaggle-survey-2022
# 
# ** Según Kaggle, se han realizado 43 preguntas pero las respuestas a preguntas de selección multiple fueron divididas en varias columnas con una columnapor respuesta dada.

# Iniciamos la Práctica con la carga de las siguientes librerías:

# In[133]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn import svm
import re

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Extracción, exploración y limpieza de datos:
# Vamos a leer el archivo de la encuesta del 2022:

# In[134]:


#Definimos la clase:
class globals:
    pass

def invert(d:dict):
    new = dict()
    
    for key, val in d.items():
        if val not in new:
            new[val] = []
        new[val].append(key)
        
    return new  
# definimos para los valores que traen valores vacios y no vacios
def response_extract(data, columns, anti_columns):
    ''' for values at a given index 
        if any of data[columns] is not na and any of data[anti_columns] is na :returns True
        if any of data[columns] is not na but all of data[anti_columns] is not na :returns False
        if all of data[columns] is na: returns na
    '''
        
    data['response'] = data[columns].notna().any(axis=1).replace({False: pd.NA})
    data['response'].loc[data[anti_columns].notna().all(axis=1)] = False 
    
    return data['response']


# In[135]:


# Subimos los archivos
data_2022 = pd.read_csv("kaggle_survey_2022_responses.csv", low_memory=False)
data_2022.head()


# In[136]:


# Eliminamos la primera fila de los datasets para mejor comprensión

data_2022 = data_2022.drop(0).assign(year='2022')


# In[137]:


data_2022= data_2022.replace({'Q8':{'Bachelorâ€™s degree' : 'Bachelor´s degree', 'Masterâ€™s degree' : 'Master´s degree' }})

#Dado de que tiene diferentes nombres los valores vamos a cambiarlos
data_2022=data_2022.replace({'Q3':{'Prefer to self-describe' : 'A different identity',  'Nonbinary': 'A different identity' }})
#rellenamos el null con 'Prefer not to say'
data_2022.head()


# In[138]:


print("El número de observaciones y variables del dataset es:", data_2022.shape)


# Veamos los valores nulos:

# In[139]:


data_2022.isnull().sum()


# In[140]:


#rellenamos el null con 'Prefer not to say'
data_2022['Q3'] =  data_comb['Q2'].fillna('Prefer not to say')
data_2022['Q5'] =  data_comb['Q5'].fillna('Dont say')
data_2022['Q4'] =  data_comb['Q4'].fillna('Dont say')
data_2022['Q2'] =  data_comb['Q3'].fillna('Dont say')


# Como son valores que se han segmentado de varias preguntas se dejará así. Ahora se usara el dataset combinado extraido de:https://www.kaggle.com/datasets/andradaolteanu/kaggle-data-science-survey-20172021 del 2017 al 2021.

# In[141]:


# subimos el archivo combinado
data_comb  = pd.read_csv("kaggle_survey_2017_2021.csv", low_memory=False)
data_comb.head()


# In[142]:


# Limpiamos de nuevo la columna
data_comb = data_comb.drop(0)
data_comb.head()
data_comb = data_comb.rename(columns={'-': 'Year'})
data_comb.head()


# In[143]:


# reemplazamos los valores por comprensión
data_comb["Q4"].unique()


# In[144]:


data_comb=data_comb.replace({'Q4':{'Bachelorâ€™s degree' : "Bachelor's degree", 'Masterâ€™s degree' : "Master's degree",
                                  'Some college/university study without earning a bachelorâ€™s degree': "Some college/university study",
                                  "Some college/university study without earning a bachelor's degree": "Some college/university study",
                                  'I did not complete any formal education past high school':'No formal education past high school'}})


# In[145]:


data_comb["Q4"].unique()


# In[146]:


#Vemos como tenemos genero:
data_comb["Q2"].unique()


# In[147]:


#Dado de que tiene diferentes nombres los valores vamos a cambiarlos
data_comb=data_comb.replace({'Q2':{'Male' : 'Man', 'Female' : 'Woman', 'Non-binary, genderqueer, or gender non-conforming': 'A different identity','Prefer to self-describe' : 'A different identity',  'Nonbinary': 'A different identity' }})

#rellenamos el null con 'Prefer not to say'
data_comb['Q2'] =  data_comb['Q2'].fillna('Prefer not to say')
data_comb['Q5'] =  data_comb['Q5'].fillna('Dont say')
data_comb['Q4'] =  data_comb['Q4'].fillna('Dont say')
data_comb['Q3'] =  data_comb['Q3'].fillna('Dont say')
data_comb.head()


# In[148]:


data_comb["Q2"].unique()
data_comb.head()


# In[150]:


#Descargamos los archivos:

data_comb.to_csv(r'C:\Users\dayan\Documents\data_combined.csv', index=False, header=True)
data_2022.to_csv(r'C:\Users\dayan\Documents\data_2022.csv', index=False, header=True)


# Nuestra visualización se hara en:

# In[ ]:




