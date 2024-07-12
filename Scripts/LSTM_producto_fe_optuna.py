# %%
import pandas as pd
import os
import numpy as np
import optuna
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
import json

entorno = 'VM'  # Elegir "VM" o "local" para correr en entorno local
nombre_experimento = 'LSTM_producto_FE_optuna'
ventana_input = 12
ventana_output = 2
ventana_test = 3
lags = 6
cant_productos_considerar = 50
num_trials = 30



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

    
# Configurar logger de Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)
logger = optuna.logging.get_logger("optuna")
log_path = os.path.join(carpeta_exp, 'optuna_log.txt')
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)  

dataset_completo = pd.read_csv(os.path.join(carpeta_datasets, 'df_producto_cliente_completo.csv'))


dataset_completo.head()

# %%
ventas_producto_mes = dataset_completo.groupby(['Timestamp', 'product_id', 'cat1', 'cat2', 'cat3', 'brand', 'descripcion']).agg({
    'tn': 'sum',
    'edad_producto': 'first'
})
ventas_producto_mes = ventas_producto_mes.reset_index()

#Generar ventas por categoria y agregar al dataset
ventas_cat1 = ventas_producto_mes.groupby(['Timestamp', 'cat1'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat1'})
ventas_cat2 = ventas_producto_mes.groupby(['Timestamp', 'cat1', 'cat2'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat2'})
ventas_cat3 = ventas_producto_mes.groupby(['Timestamp', 'cat1', 'cat2', 'cat3'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat3'})
ventas_familia_productos = ventas_producto_mes.groupby(['Timestamp', 'cat1', 'cat2', 'cat3', 'brand', 'descripcion'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_familia_producto'})

ventas_producto_mes = ventas_producto_mes.merge(ventas_cat1, how='left', on=['Timestamp','cat1'])
ventas_producto_mes = ventas_producto_mes.merge(ventas_cat2, how='left', on=['Timestamp', 'cat1', 'cat2'])
ventas_producto_mes = ventas_producto_mes.merge(ventas_cat3, how='left', on=['Timestamp', 'cat1', 'cat2', 'cat3'])
ventas_producto_mes = ventas_producto_mes.merge(ventas_familia_productos, how='left', on=['Timestamp', 'cat1', 'cat2', 'cat3', 'brand', 'descripcion'])

ventas_producto_mes.drop(columns=['cat1', 'cat2', 'cat3', 'brand', 'descripcion'], inplace = True)
ventas_producto_mes.set_index('Timestamp', inplace=True)
ventas_producto_mes

# %%
ventas_dic_2019 = ventas_producto_mes[ventas_producto_mes.index == '2019-12-01']
ventas_dic_2019 = ventas_dic_2019.sort_values(by='tn', ascending = False)
lista_productos_mayores_ventas = list(ventas_dic_2019.iloc[:cant_productos_considerar]['product_id'].values)
lista_productos_mayores_ventas

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
# Función para crear el modelo LSTM
def crear_modelo_lstm(input_shape, units_lstm, dropout_rate, learning_rate):
    model = Sequential()
    model.add(LSTM(units_lstm[0], return_sequences=True, input_shape=input_shape, recurrent_dropout=0.25))
    model.add(LSTM(units_lstm[1], return_sequences=True, recurrent_dropout=0.25))
    model.add(LSTM(units_lstm[2], recurrent_dropout=0.25))
    model.add(Dropout(dropout_rate))
    model.add(Dense(ventana_output))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=[MeanSquaredError()])
    return model

# %%
# Función para la optimización con Optuna
def objective(trial):
    # Definir hiperparámetros a optimizar
    units_lstm = [
        trial.suggest_int('units_lstm_1', 32, 128),
        trial.suggest_int('units_lstm_2', 32, 128),
        trial.suggest_int('units_lstm_3', 32, 128)
    ]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 1, 32)
    epochs = trial.suggest_int('epochs', 50, 200)

    lista_productos_LSTM = []
    lista_predicciones_LSTM = []
    loss_list = []

    for producto in lista_productos_mayores_ventas:
        ventas_mes_por_producto = ventas_producto_mes[ventas_producto_mes['product_id'] == producto].copy()
        ventas_mes_por_producto.drop(columns=['product_id'], inplace=True)

        # Escalar valor
        scaler_tn = RobustScaler()
        scaler_edad = RobustScaler()
        scaler_ventas_cat1 = RobustScaler()
        scaler_ventas_cat2 = RobustScaler()
        scaler_ventas_cat3 = RobustScaler()
        ventas_familia_producto = RobustScaler()
        ventas_mes_por_producto['tn'] = scaler_tn.fit_transform(np.array(ventas_mes_por_producto['tn']).reshape(-1,1))
        ventas_mes_por_producto['edad_producto'] = scaler_edad.fit_transform(np.array(ventas_mes_por_producto['edad_producto']).reshape(-1,1))
        ventas_mes_por_producto['ventas_cat1'] = scaler_ventas_cat1.fit_transform(np.array(ventas_mes_por_producto['ventas_cat1']).reshape(-1,1))
        ventas_mes_por_producto['ventas_cat2'] = scaler_ventas_cat2.fit_transform(np.array(ventas_mes_por_producto['ventas_cat2']).reshape(-1,1))
        ventas_mes_por_producto['ventas_cat3'] = scaler_ventas_cat3.fit_transform(np.array(ventas_mes_por_producto['ventas_cat3']).reshape(-1,1))
        ventas_mes_por_producto['ventas_familia_producto'] = ventas_familia_producto.fit_transform(np.array(ventas_mes_por_producto['ventas_familia_producto']).reshape(-1,1))

        
        


        if len(ventas_mes_por_producto) >= (ventana_input + ventana_output):
            # Formatear valores para input LSTM
            X, Y = crear_dataset_supervisado(ventas_mes_por_producto.values, ventana_input, ventana_output)

            # Separar datos en entrenamiento y test
            train_size = len(X) - ventana_test
            X_train, X_test = X[:train_size], X[train_size:]
            Y_train, Y_test = Y[:train_size], Y[train_size:]

            # Crear y ajustar el modelo LSTM
            model = crear_modelo_lstm((ventana_input, X.shape[2]), units_lstm, dropout_rate, learning_rate)
            model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Evaluar el modelo
            loss = model.evaluate(X_test, Y_test, verbose=0)
            loss_list.append(loss)

    return np.mean(loss_list)

# %%
# Ejecución de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=num_trials)

# Guardar los mejores hiperparámetros
best_params = study.best_params
with open(os.path.join(carpeta_exp, 'mejores_hiperparametros.json'), 'w') as f:
    json.dump(best_params, f)

print("Mejores hiperparámetros encontrados: ", best_params)


