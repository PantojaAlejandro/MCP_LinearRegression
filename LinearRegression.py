import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import re
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def main():
    #Read CSV File
    dataset= pd.read_csv("02-2019.csv", sep=';')
    #Preprocessing, deleting unnecessary columns
    dataset = dataset[dataset['error'] == 'N']
    dataset = dataset[dataset['tipo_elem'] == 'M30']
    dropped_columns= ['id','tipo_elem','ocupacion','carga','vmed','error','periodo_integracion']
    dataset = dataset.drop(columns=dropped_columns)
    #Normalize the rest of the dataset, we only remain with dates between 6 and 8 am
    dataset['fecha'] = dataset['fecha'].apply(normalizeTime)
    dataset = dataset[dataset['fecha'] != -1]
    intensidad = dataset['intensidad']
    max_intensidad = intensidad.max()
    min_intensidad = intensidad.min()
    dataset['intensidad'] = np.clip((intensidad - min_intensidad) / max_intensidad, 0, 1)
    #just print to check everything is fine
    print(dataset.head())
    #training
    x= dataset[['fecha']]
    y= dataset['intensidad']
    x_train, x_test, y_train, y_test= train_test_split(x, y,test_size=0.2, random_state=12)
    lr=LinearRegression().fit(x_train,y_train)
    #testing
    y_pred=lr.predict(x_test)
    sqr_error= mean_squared_error(y_test, y_pred)
    r2= r2_score(y_test, y_pred)
    print("El error cuadratico medio es: ", sqr_error)
    print("El r2 es: ", r2)
    trainTestRidge(x_train, y_train, x_test, y_test)
    trainTestLasso(x_train, y_train, x_test, y_test)
    trainTestElasticNet(x_train, y_train, x_test, y_test)
    trainTestPoly(x_train, y_train, x_test, y_test)

def normalizeTime (str: str):
    """
    This method normalizes the date
    :param str: the string of the date
    :return: the normalized date (float)
    """
    array = str.split(":")
    fecha_hora = array[0].split(" ")
    hora = int(fecha_hora[1])
    minuto = float(array[1])
    if hora < 6 or hora > 8 or (hora == 8 and minuto > 0):
        return -1
    else:
        return np.clip(((hora + (minuto / 60)) - 6) / 2, 0, 1)
def getCSVs():
    """
    Method that gets all CSVs
    :return:
    """
    with open('208627-0-transporte-ptomedida-historico.rdf', 'r') as file_urls:
        lines = file_urls.readlines()
        urls = [re.search('https://datos(.+?).zip', l).group(0) for l in lines if ".zip" in l]
        urls_wget = ["wget -nc " + x for x in urls]
        with open("download_script.sh", "w") as outputfile:
            outputfile.write("\n".join(urls_wget))

def trainTestRidge(x_train, y_train, x_test, y_test):
    best_alpha: float
    best_r2: float=0
    for alpha in np.linspace(0.01, 1, num=100):
        ridge = Ridge(alpha=alpha).fit(x_train, y_train)
        r2= r2_score(y_test,ridge.predict(x_test))
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
            error_cuadratico= mean_squared_error(y_test,ridge.predict(x_test))
    print("Mejor alpha con ridge: ", best_alpha)
    print("Mejor r2 con ridge: ", best_r2)
    print("Error cuadratico con ridge: ", error_cuadratico)

def trainTestLasso(x_train, y_train, x_test, y_test):
    best_alpha: float
    best_r2: float=0
    for alpha in np.linspace(0.001, 1, num=100):
        lasso = Lasso(alpha=alpha).fit(x_train, y_train)
        r2= r2_score(y_test,lasso.predict(x_test))
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
            error_cuadratico= mean_squared_error(y_test,lasso.predict(x_test))
    print("Mejor alpha con lasso: ", best_alpha)
    print("Mejor r2 con lasso: ", best_r2)
    print("Error cuadratico con lasso: ", error_cuadratico)

def trainTestElasticNet(x_train, y_train, x_test, y_test):
    best_alpha: float
    best_l1_ratio: float
    best_r2: float=0
    for alpha in np.linspace(0.01, 1, num=100):
        for l1_ratio in np.linspace(0.01, 1, num=50):
            elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(x_train, y_train)
            r2= r2_score(y_test,elastic_net.predict(x_test))
            if r2 > best_r2:
                best_alpha = alpha
                best_l1_ratio = l1_ratio
                best_r2 = r2
                error_cuadratico= mean_squared_error(y_test,elastic_net.predict(x_test))
    print("Mejor alpha con elastic net: ", best_alpha)
    print("Mejor l1_ratio con elastic net: ", best_l1_ratio)
    print("Mejor r2 con elastic net: ", best_r2)
    print("Error cuadratico con elastic net: ", error_cuadratico)

def trainTestPoly(x_train, y_train, x_test, y_test):
    best_degree: float
    best_r2: float=0
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    for i in range(2,5,1):
        poly = make_pipeline(PolynomialFeatures(degree=i), LinearRegression())
        poly.fit(x_train, y_train)
        predicciones = poly.predict(x_test)
        r2 = r2_score(y_test, predicciones)
        if r2 > best_r2:
            best_degree = i
            best_r2 = r2
            error_cuadratico= mean_squared_error(y_test,poly.predict(x_test))
    print("Mejor grado polinómico: ", best_degree)
    print("Mejor r2 con polinomio: ", best_r2)
    print("Error cuadratico medio con polinomio: ", error_cuadratico)






if __name__ == "__main__":
    main()