import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

dataset= pd.read_csv("testing.csv")
dataset['fecha']= dataset['fecha'].apply(pd.to_datetime)
dataset = (
    dataset.set_index('fecha').between_time("06:00", "8:00").reset_index()
)
dropped_columns= ['id','tipo_elem','ocupacion','carga','vmed','error','periodo_integracion']
for column in dropped_columns:
    dataset=dataset.drop(columns=[column])
hora_decimal = dataset['fecha'].dt.hour + (dataset['fecha'].dt.minute / 60.0)
dataset['fecha'] = np.clip((hora_decimal - 6) / 2, 0, 1)
intensidad= dataset['intensidad']
max_intensidad: int= -1
min_intensidad: int= 9999
for i in intensidad:
    if max_intensidad< i:
        max_intensidad= i
    elif min_intensidad> i:
        min_intensidad= i
dataset['intensidad'] = np.clip((intensidad-min_intensidad)/max_intensidad, 0, 1)
print(dataset.head())
x= dataset[['fecha']]
y= dataset['intensidad']
x_train, x_test, y_train, y_test= train_test_split(x, y,test_size=0.2, random_state=12)
lr=LinearRegression().fit(x_train,y_train)
y_pred=lr.predict(x_test)
sqr_error= mean_squared_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)
print("El error cuadratico medio es: ", sqr_error)
print("El r cuadrado es: ", r2)