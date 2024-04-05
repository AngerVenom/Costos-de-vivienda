#Desarrollado por Maximo  Gonzalez 2023

# %% [markdown]
# Data
# 
# - `price`: precio
# - `area`: área
# - `bedrooms`: habitaciones
# - `bathrooms`: baños
# - `stories`: pisos
# - `mainroad`: carretera principal
# - `guestroom`: cuarto de huéspedes
# - `basement`: sótano
# - `hotwaterheating`: calefacción de agua caliente
# - `airconditioning`: aire acondicionado
# - `parking`: estacionamiento
# - `prefarea`: área preferida
# - `furnishingstatus`: estado del amueblado

# %%
#LIBRERIAS
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#Modelo de regresion lineal
from sklearn.linear_model import LinearRegression
#Para rendimiento
from sklearn.feature_selection import RFE
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
#Carga de datos
df = pd.read_csv('housing.csv', delimiter=',')
df.dataframeName = 'housing.csv'
numRows, numCols = df.shape
print(f'Hay {numRows} filas y {numCols} columnas.')

# %%
#Muestra los primeros 5 datos
df.head(5)

# %%
categorical_list = [x for x in df.columns if df[x].dtype =='object']

# %%
#LISTA PARA VARIABLES NUMERICAS
numerical = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# %%
# Defininendo la funcion de dummies
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True).astype(int)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Aplicando la funciona a los valores categoritos

df = dummies('mainroad',df)
df = dummies('guestroom',df)
df = dummies('hotwaterheating',df)
df = dummies('basement',df)
df = dummies('airconditioning',df)
df = dummies('prefarea',df)
df = dummies('furnishingstatus',df)
# %%
numpy.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100) #0.7 es el 70% para entrenamiento

# %%
scaler = MinMaxScaler()
df_train[numerical] = scaler.fit_transform(df_train[numerical])

# %%
df_train.head()

# %%
y_train = df_train.pop('price')
X_train = df_train

# %%
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

# %% [markdown]
# CREACION DEL MODELO

# %%
X_train.columns[rfe.support_]

# %%
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()

# %%
def build_model(X,y):
    X = sm.add_constant(X) 
    lm = sm.OLS(y,X).fit() 
    print(lm.summary()) # Resumen del modelo
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)

# %% [markdown]
# #Modelo 1

# %%
X_train_new = build_model(X_train_rfe,y_train)

# %%
X_train_new = X_train_new.drop(["bedrooms"], axis = 1)

# %% [markdown]
# #MODELO 2

# %%
X_train_new = build_model(X_train_new,y_train)

# %%
#Calculando la varianza
checkVIF(X_train_new)

# %%
X_train_new = X_train_new.drop(["yes"], axis = 1)

# %% [markdown]
# #MODELO 3

# %%
X_train_new = build_model(X_train_new,y_train)

# %%
#Calculando la varianza
checkVIF(X_train_new)

# %%
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)

# %%
# Escalando los valores 
df_test[numerical] = scaler.fit_transform(df_test[numerical])

# %%
#Dividir la variable x de la variable y del conjunto de prueba
y_test = df_test.pop('price')
X_test = df_test

# %%
# seleccionando las características elegidas del conjunto de entrenamiento para el conjunto de prueba
X_train_new = X_train_new.drop('const',axis=1)
# Creando el dataframe X_test_new eliminando variables de X_test
X_test_new = X_test[X_train_new.columns]

X_test_new = sm.add_constant(X_test_new)

# %%
#PREDICCIONES
# Seleccionar 5 observaciones aleatorias del conjunto de prueba
random_indices = numpy.random.choice(X_test_new.index, size=3, replace=False)

# Extraer las características y etiquetas correspondientes
X_random = X_test_new.loc[random_indices]
y_true = y_test.loc[random_indices]

# Realizar predicciones
y_pred = lm.predict(X_random)

# Crear un DataFrame para mostrar los resultados
results = pd.DataFrame({
    'Precio predicho': y_pred,
    'Precio verdadero': y_true
})

print(results)


