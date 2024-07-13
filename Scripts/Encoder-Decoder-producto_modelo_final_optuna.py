# %%
import pandas as pd
import os
import numpy as np
import json
import optuna
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
import json

entorno = 'VM'  # Elegir "VM" o "local" para correr en entorno local
nombre_experimento = 'Encoder-Decoder_modelo_final_optuna'
nombre_carpeta_optuna = 'Encoder-Decoder-producto_optuna'
ventana_input = 12
ventana_output = 2


# Configurar entorno
if entorno == 'VM':
    carpeta_datasets = os.path.expanduser('~/buckets/b1/datasets')
    carpeta_exp_base = os.path.expanduser('~/buckets/b1/exp')
elif entorno == 'local':
    carpeta_datasets = 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Datos'
    carpeta_exp_base = 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Resultados'
else:
    raise Exception("Entorno especificado incorrectamente")

carpeta_exp = os.path.join(carpeta_exp_base, nombre_experimento)
if not os.path.exists(carpeta_exp):
    os.makedirs(carpeta_exp)



dataset_completo = pd.read_csv(os.path.join(carpeta_datasets, 'df_producto_cliente_completo.csv'))


#Eliminar columnas no utilizadas
dataset_completo.head()

# %%
#Recuperar mejores hiperparametros de optimizacion
with open(os.path.join(os.path.join(carpeta_exp_base,nombre_carpeta_optuna), 'mejores_hiperparametros.json'), 'r') as f:
    best_params = json.load(f)
print(best_params)

units_lstm = [best_params['units_lstm_1'], best_params['units_lstm_2'], best_params['units_lstm_3']]
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
epochs = best_params['epochs']

# %%

ventas_producto_mes = dataset_completo.groupby(['Timestamp', 'product_id'])['tn'].sum()
ventas_producto_mes = ventas_producto_mes.reset_index()
ventas_producto_mes.set_index('Timestamp', inplace=True)
ventas_producto_mes

# %%
def crear_dataset_supervisado(array, input_length, output_length):
    # Inicialización
    X, Y = [], []    # Listados que contendrán los datos de entrada y salida del modelo
    shape = array.shape
    if len(shape) == 1:  # Si tenemos sólo una serie (univariado)
        array = array.reshape(-1, 1)
        cols = 1
    else:  # Multivariado
        fils, cols = array.shape

    # Generar los arreglos (utilizando ventanas deslizantes de longitud input_length)
    for i in range(fils - input_length - output_length + 1):
        X.append(array[i:i + input_length, :].reshape(input_length, cols))
        Y.append(array[i + input_length:i + input_length + output_length, -1].reshape(output_length, 1))

    # Convertir listas a arreglos de NumPy
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# %%
# Función para crear el modelo LSTM encoder-decoder
def crear_modelo_lstm_encoder_decoder(input_shape, units_lstm, dropout_rate, learning_rate):
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(units_lstm[0], return_sequences=True, recurrent_dropout=0.25)(encoder_inputs)
    encoder = LSTM(units_lstm[1], return_sequences=False, recurrent_dropout=0.25)(encoder)
    
    decoder_inputs = RepeatVector(ventana_output)(encoder)
    decoder = LSTM(units_lstm[2], return_sequences=True, recurrent_dropout=0.25)(decoder_inputs)
    decoder_outputs = TimeDistributed(Dense(1))(decoder)
    
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=[MeanSquaredError()])
    return model

def predecir(x, model, scaler):
    # Calcular predicción escalada en el rango de escalado
    y_pred_s = model.predict(x,verbose=0)
    # Llevar la predicción a la escala original
    y_pred = scaler.inverse_transform(y_pred_s.reshape(ventana_output,1))
    return y_pred.flatten()

# %%
lista_productos_LSTM = []
lista_predicciones_LSTM = []

for producto in ventas_producto_mes['product_id'].unique():
    ventas_mes_por_producto = ventas_producto_mes[ventas_producto_mes['product_id'] == producto].copy()
    ventas_mes_por_producto.drop(columns=['product_id'], inplace=True)
    
    cantidad_datos_no_nulos = len(ventas_mes_por_producto[ventas_mes_por_producto['tn'] != 0])
    

    if cantidad_datos_no_nulos >= (ventana_input + ventana_output):
        # Escalar valor
        scaler = RobustScaler()
        ventas_mes_por_producto['tn'] = scaler.fit_transform(ventas_mes_por_producto)
        
        # Formatear valores para input LSTM
        X, Y = crear_dataset_supervisado(ventas_mes_por_producto.values, ventana_input, ventana_output)

        # Crear y ajustar el modelo LSTM
        model = crear_modelo_lstm_encoder_decoder((ventana_input, X.shape[2]), units_lstm, dropout_rate, learning_rate)
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

        
        #Predecir valores
        input_prediccion = X[-1].reshape(1,X.shape[1],X.shape[2])
        prediccion_mes_2 = predecir(input_prediccion, model, scaler)[1]
        
        print(f'Producto {producto} - Prediccion: {prediccion_mes_2}')
        lista_productos_LSTM.append(producto)
        lista_predicciones_LSTM.append(prediccion_mes_2)
    else:
        print(f'Producto {producto}: Valores insuficientes para usar ventana de 12 para LSTM, se predice por promedio')
        lista_productos_LSTM.append(producto)
        ventas_2019 = ventas_mes_por_producto[(ventas_mes_por_producto.index >= '2019-01-01')]
        lista_predicciones_LSTM.append(ventas_2019['tn'].mean())

# %%
resultados_kaggle_LSTM = pd.DataFrame({'product_id': lista_productos_LSTM, 'tn': lista_predicciones_LSTM})
resultados_kaggle_LSTM['tn'] = resultados_kaggle_LSTM['tn'].apply(lambda x: max(0,x))
resultados_kaggle_LSTM.to_csv(os.path.join(carpeta_exp, 'predicciones_Encoder-Decoder_modelo_final_optuna.csv'), index= False)

# %%



