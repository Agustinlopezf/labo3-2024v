# %%
import pandas as pd
import os
import numpy as np
import optuna
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
import json

entorno = 'VM'  # Elegir "VM" o "local" para correr en entorno local
nombre_experimento = 'LSTM_GRU_producto_optuna'
ventana_input = 12
ventana_output = 2
ventana_test = 3
cant_productos_considerar = 40
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

ventas_producto_mes = dataset_completo.groupby(['Timestamp', 'product_id'])['tn'].sum()
ventas_producto_mes = ventas_producto_mes.reset_index()
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
# Función para crear el modelo 
def crear_modelo(input_shape, units_lstm, dense_units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Bidirectional(LSTM(units_lstm[0], return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units_lstm[1], return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units_lstm[2], return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units))
    model.add(Dense(ventana_output))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=[MeanSquaredError()])

    return model


# Function to add additional features to the data
def add_features(data):
    df = pd.DataFrame(data, columns=['tn'])
    df.index = pd.to_datetime(df.index)
    df['lag_1'] = df['tn'].shift(1)
    df['lag_2'] = df['tn'].shift(2)
    df['ma_3'] = df['tn'].rolling(window=3).mean()
    df['ma_6'] = df['tn'].rolling(window=6).mean()
    df['std_3'] = df['tn'].rolling(window=3).std()
    df['trend'] = range(len(df))
    df['month'] = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * df['month']/12)
    df['cos_month'] = np.cos(2 * np.pi * df['month']/12)
    df.dropna(inplace=True)
    return df.values

# %%
# Función para la optimización con Optuna
def objective(trial):
    # Definir hiperparámetros a optimizar
    units_lstm = [
        trial.suggest_int('units_lstm_1', 32, 128),
        trial.suggest_int('units_lstm_2', 32, 128),
        trial.suggest_int('units_lstm_3', 32, 128)
    ]
    
    dense_units = trial.suggest_int('dense_units', 8, 32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)


    lista_productos_LSTM = []
    lista_predicciones_LSTM = []
    loss_list = []

    for producto in lista_productos_mayores_ventas:
        print(producto)
        ventas_mes_por_producto = ventas_producto_mes[ventas_producto_mes['product_id'] == producto].copy()
        ventas_mes_por_producto.drop(columns=['product_id'], inplace=True)
        
        ventas_mes_por_producto_features = add_features(ventas_mes_por_producto)
        # Escalar valor
        scaler = RobustScaler()
        ventas_mes_por_producto_features = scaler.fit_transform(ventas_mes_por_producto_features)


        if len(ventas_mes_por_producto_features) >= (ventana_input + ventana_output):
            # Formatear valores para input LSTM
            X, Y = crear_dataset_supervisado(ventas_mes_por_producto_features, ventana_input, ventana_output)

            # Separar datos en entrenamiento y test
            train_size = len(X) - ventana_test
            X_train, X_test = X[:train_size], X[train_size:]
            Y_train, Y_test = Y[:train_size], Y[train_size:]
            
            early_stop = EarlyStopping(monitor='loss', patience=20)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=1e-6)

            # Crear y ajustar el modelo LSTM
            model = crear_modelo((ventana_input, X.shape[2]),units_lstm,dense_units, dropout_rate, learning_rate)
            model.fit(X_train, Y_train, epochs=150, batch_size=24, callbacks=[early_stop, reduce_lr], verbose=0)

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


