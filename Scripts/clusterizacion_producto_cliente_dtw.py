import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from fastdtw import fastdtw
import matplotlib.pyplot as plt

# Configuraciones iniciales
entorno = 'VM'  # Elegir "VM" o "local" para correr en entorno local
nombre_experimento = 'clusterizacion_dtw'
ventana_input = 12
ventana_output = 2
num_clusters = 20
modo_test = True  # Configura True para usar solo 1000 primeras series en modo de prueba

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

# Preparar ventas por producto y mes
ventas_producto_mes = sell_in.groupby(['periodo', 'product_id', 'customer_id'])['tn'].sum().reset_index()
ventas_producto_mes['Timestamp'] = pd.to_datetime(ventas_producto_mes['periodo'], format='%Y%m')
ventas_producto_mes.set_index('Timestamp', inplace=True)
ventas_producto_mes.drop(columns=['periodo'], inplace=True)

# Calcular la matriz de distancias DTW
productos_clientes = ventas_producto_mes.groupby(['product_id', 'customer_id'])['tn'].apply(list).reset_index()
if modo_test:
    productos_clientes = productos_clientes.head(1000)

series = productos_clientes['tn'].tolist()
distancias = np.zeros((len(series), len(series)))

for i in range(len(series)):
    for j in range(i + 1, len(series)):
        distancia, _ = fastdtw(series[i], series[j])
        distancias[i, j] = distancia
        distancias[j, i] = distancia

# Clusterizar las series de tiempo
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
productos_clientes['cluster'] = kmeans.fit_predict(distancias)

# Guardar las series con su cluster asociado
productos_clientes.to_csv(os.path.join(carpeta_exp, 'series_con_clusters.csv'), index=False)

# Generar gráficos de las series por cluster
for cluster in range(num_clusters):
    series_cluster = productos_clientes[productos_clientes['cluster'] == cluster]
    plt.figure(figsize=(12, 8))

    for _, row in series_cluster.iterrows():
        product_id = row['product_id']
        customer_id = row['customer_id']
        serie = ventas_producto_mes[(ventas_producto_mes['product_id'] == product_id) &
                                    (ventas_producto_mes['customer_id'] == customer_id)]['tn']
        serie.plot(label=f'Producto: {product_id}, Cliente: {customer_id}')

    plt.title(f'Cluster {cluster}')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.savefig(os.path.join(carpeta_exp, f'cluster_{cluster}.png'))
    plt.close()