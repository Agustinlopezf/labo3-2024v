import pandas as pd
import os
import numpy as np

#Para VM

carpeta_datasets= '~/buckets/b1/datasets'
carpeta_exp= '~/buckets/b1/exp'


#Para Máquina Local
'''
carpeta_datasets= 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Datos'
carpeta_exp= 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Resultados'
'''
nombre_archivo_resultados = 'resultados_LSTM_producto_cliente'

#Carga archivos
sell_in = pd.read_csv(os.path.join(carpeta_datasets, 'sell-in.txt'), delimiter = '\t')
tb_productos = pd.read_csv(os.path.join(carpeta_datasets, 'tb_productos_descripcion.txt'), delimiter = '\t')
tb_stocks = pd.read_csv(os.path.join(carpeta_datasets, 'tb_stocks.txt'), delimiter = '\t')
productos_predecir = pd.read_csv(os.path.join(carpeta_datasets, 'productos_a_predecir.txt'), delimiter = '\t') 



ventas_producto_mes = sell_in.groupby(['periodo', 'product_id', 'customer_id'])['tn'].sum()
ventas_producto_mes = ventas_producto_mes.reset_index()
ventas_producto_mes['Timestamp'] = pd.to_datetime(ventas_producto_mes['periodo'], format='%Y%m')
ventas_producto_mes.set_index('Timestamp', inplace=True)
ventas_producto_mes.drop(columns=['periodo'], inplace = True)


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
from sklearn.preprocessing import StandardScaler


def crear_dataset_supervisado(array, input_length, output_length):

    # Inicialización
    X, Y = [], []    # Listados que contendrán los datos de entrada y salida del modelo
    shape = array.shape
    if len(shape)==1: # Si tenemos sólo una serie (univariado)
        fils, cols = array.shape[0], 1
        array = array.reshape(fils,cols)
    else: # Multivariado
        fils, cols = array.shape

    # Generar los arreglos (utilizando ventanas deslizantes de longitud input_length)
    for i in range(fils-input_length-output_length + 1):
        X.append(array[i:i+input_length,0:cols])
        Y.append(array[i+input_length:i+input_length+output_length,-1].reshape(output_length,1))

    # Convertir listas a arreglos de NumPy
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def predecir(x, model, scaler):
    # Calcular predicción escalada en el rango de escalado
    y_pred_s = model.predict(x,verbose=0)
    # Llevar la predicción a la escala original
    y_pred = scaler.inverse_transform(y_pred_s)
    return y_pred.flatten()

ventana_input = 12
ventana_output = 2

lista_productos = []
lista_predicciones = []
lista_customers = []
scaler_list = []

i = 0
for producto in ventas_producto_mes['product_id'].unique():
    if producto in list(productos_predecir['product_id']):
        print(f'Entrenando producto nro {i}')
        i += 1
        for cliente in ventas_producto_mes['customer_id'].unique():
            lista_customers.append(cliente)
            ventas_mes_por_producto = ventas_producto_mes[(ventas_producto_mes['product_id'] == producto) & (ventas_producto_mes['customer_id'] == cliente)].copy()
            ventas_mes_por_producto.drop(columns=['product_id', 'customer_id'], inplace = True)

            #Escalar valor
            scaler = StandardScaler()
            ventas_mes_por_producto_escalado = scaler.fit_transform(ventas_mes_por_producto)
            scaler_list.append(scaler)
            if len(ventas_mes_por_producto) > ventana_input:
                try:
                    #Formatear valores para input LSTM
                    X, Y =crear_dataset_supervisado(ventas_mes_por_producto_escalado, ventana_input, ventana_output)
                    # Create and fit the LSTM network
                    model = Sequential()
                    model.add(LSTM(64, return_sequences=True, input_shape=(ventana_input, 1), recurrent_dropout=0.25))
                    model.add(LSTM(32, recurrent_dropout=0.25))
                    model.add(Dropout(0.5))
                    model.add(Dense(ventana_output))
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    model.fit(X, Y, epochs=100, batch_size=1, verbose=1)

                    #Predecir valores
                    prediccion_mes_2 = predecir(X[-1].reshape(1,-1), model, scaler)[1]

                    lista_productos.append(producto)
                    lista_predicciones.append(prediccion_mes_2)
                except:
                    lista_productos.append(producto)
                    lista_predicciones.append(ventas_mes_por_producto['tn'].mean())
            else:
                print('Valores insuficientes para usar ventana de 12 para LSTM, se predice por promedio')
                lista_productos.append(producto)
                lista_predicciones.append(ventas_mes_por_producto['tn'].mean())


resultados_kaggle_por_customer = pd.DataFrame({'product_id': lista_productos, 'customer_id': lista_customers, 'tn': lista_predicciones})
resultados_kaggle_por_customer['tn'] = resultados_kaggle_por_customer['tn'].apply(lambda x: max(0,x))
#Agrupar valores por product_id
resultados_kaggle = resultados_kaggle_por_customer.groupby(['product_id'])['tn'].sum()
resultados_kaggle = resultados_kaggle.reset_index()
resultados_kaggle.to_csv(os.path.join(carpeta_exp, nombre_archivo_resultados + '.csv'), index= False)