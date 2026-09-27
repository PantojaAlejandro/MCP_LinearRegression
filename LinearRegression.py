import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import re
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
    print("El r cuadrado es: ", r2)

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


if __name__ == "__main__":
    main()