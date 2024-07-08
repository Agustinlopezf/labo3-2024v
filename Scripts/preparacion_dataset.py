# -*- coding: utf-8 -*-
"""preparacion_dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ve1zpbURlP5cDdEs3Wf1HFqiu1Nrz2yZ
"""

import pandas as pd
import os
import numpy as np
import joblib

entorno = 'VM'  # Elegir "VM" o "local" para correr en entorno local
nombre_experimento = 'LSTM_producto'
ventana_input = 12
ventana_output = 2
ventana_test = 3
lags = [1, 2, 3, 6, 12]


# Configurar entorno
if entorno == 'VM':
    carpeta_datasets = '~/buckets/b1/datasets'
    carpeta_exp_base = '~/buckets/b1/exp'
elif entorno == 'local':
    carpeta_datasets = 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Datos'
    carpeta_exp_base = 'C:\\Users\\alope\\Desktop\\Trámites\\Maestria Data Science - Universidad Austral\\Laboratorio de implementación 3\\Resultados'
else:
    raise Exception("Entorno especificado incorrectamente")

carpeta_exp = os.path.join(carpeta_exp_base, nombre_experimento)
if not os.path.exists(carpeta_exp):
    os.makedirs(carpeta_exp)

# Cargar datos
sell_in = pd.read_csv(os.path.join(carpeta_datasets, 'sell-in.txt'), delimiter='\t')
tb_productos = pd.read_csv(os.path.join(carpeta_datasets, 'tb_productos_descripcion.txt'), delimiter='\t')
tb_stocks = pd.read_csv(os.path.join(carpeta_datasets, 'tb_stocks.txt'), delimiter='\t')
productos_predecir = pd.read_csv(os.path.join(carpeta_datasets, 'productos_a_predecir.txt'), delimiter='\t')

#Combinar datasets
dataset_completo = sell_in.merge(tb_productos, how='left', on='product_id')
dataset_completo
# Extraer el mes
dataset_completo['mes'] = dataset_completo['periodo'].astype(str).str[4:6].astype(int)
# Calcular el trimestre
dataset_completo['quarter'] = dataset_completo['mes'].apply(lambda x: (x-1)//3 + 1)
dataset_completo['fin_quarter'] = dataset_completo['mes'].apply(lambda x: 1 if x in [3,6,9,12] else 0)

def diff_in_months(row):
    fecha_ingreso = row['fecha_ingreso_producto']
    fecha_actual = row['periodo']
    # Extraer año y mes de la primera fecha
    year_ingreso = fecha_ingreso // 100
    month_ingreso = fecha_ingreso % 100

    # Extraer año y mes de la segunda fecha
    year_actual = fecha_actual // 100
    month_actual = fecha_actual % 100

    # Calcular la diferencia en meses
    diff_months = (year_actual - year_ingreso) * 12 + (month_actual - month_ingreso)

    return diff_months



dataset_completo['fecha_ingreso_producto'] = dataset_completo.groupby('product_id')['periodo'].transform('min')
dataset_completo['edad_producto'] = dataset_completo.apply(diff_in_months, axis = 1)

dataset_completo.drop(columns=['fecha_ingreso_producto'], inplace = True)

ventas_cat1 = dataset_completo.groupby(['periodo', 'customer_id', 'cat1'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat1'})
ventas_cat2 = dataset_completo.groupby(['periodo', 'customer_id', 'cat1', 'cat2'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat2'})
ventas_cat3 = dataset_completo.groupby(['periodo', 'customer_id', 'cat1', 'cat2', 'cat3'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_cat3'})
ventas_familia_productos = dataset_completo.groupby(['periodo', 'customer_id', 'cat1', 'cat2', 'cat3', 'brand', 'descripcion'])['tn'].sum().reset_index().rename(columns={'tn': 'ventas_familia_producto'})

#Guardar las ventas agrupadas
joblib.dump(ventas_cat1, os.path.join(carpeta_datasets,'ventas_cat1.pkl'))
joblib.dump(ventas_cat2, os.path.join(carpeta_datasets,'ventas_cat2.pkl'))
joblib.dump(ventas_cat3, os.path.join(carpeta_datasets,'ventas_cat3.pkl'))
joblib.dump(ventas_familia_productos, os.path.join(carpeta_datasets,'ventas_familia_productos.pkl'))

df_fe_familia_productos = dataset_completo.merge(ventas_cat1, how='left', on=['periodo', 'customer_id', 'cat1'])
df_fe_familia_productos = df_fe_familia_productos.merge(ventas_cat2, how='left', on=['periodo', 'customer_id', 'cat1', 'cat2'])
df_fe_familia_productos = df_fe_familia_productos.merge(ventas_cat3, how='left', on=['periodo', 'customer_id', 'cat1', 'cat2', 'cat3'])
df_fe_familia_productos = df_fe_familia_productos.merge(ventas_familia_productos, how='left', on=['periodo', 'customer_id', 'cat1', 'cat2', 'cat3', 'brand', 'descripcion'])

df_fe_familia_productos

#Filtramos los productos de interes para reducir el tamaño del dataset
df_productos_filtrado = df_fe_familia_productos[df_fe_familia_productos['product_id'].isin(list(productos_predecir.values.reshape(-1)))]
df_productos_filtrado

df_productos_filtrado.to_csv(os.path.join(carpeta_datasets, 'dataset_completo.csv'), index = False)

