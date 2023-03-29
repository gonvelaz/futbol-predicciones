from utils.train import train_model
from utils.functions import data_processing 
import pandas as pd

'''ESTE MAIN ESTA DEDICADO ÚNICAMENTE A LA PREDICCIÓN DE RESULTADOS'''

#Carga de datos

df_final = pd.read_csv('data/processed_files/df_datos_completos.csv')

#Creación de datos nuevos. 
#Para crear los datos nuevos hay que darle valor a una serie de variables. Se muestra un ejemplo, varían por partido, y no es necesario pasar una lista completa de lesionados
#y alineaciones pero mejorará el desempeño del modelo. Los nombres del estadio y el árbitro deben estar correctos y 100% igual escritos. Para sacarlos se puede llamar a las funciones 
# buscar_estadio y buscar_arbitro del archivo functions.py . También se pueden encontrar ids de equipos y jugadores con sus respectivas funciones. (Hay que pasar siempre id de equipo,
# y jugador)
#Para este ejemplo se usarán los siguientes datos (Lugo - Real Zaragoza)
id_equipo_local = 535
id_equipo_visitante = 537
odd_1 = 2.75
odd_x = 2.9
odd_2 = 2.5
arbitro ='Saul Ais Reig, Spain'
estadio = 'Estadio La Rosaleda (Málaga)'
season = 2022  
ids_lesionados = [10057]
ids_titulares = [
46755, 77623, 46818, 24793, 46759, 8627, 47388, 46691, 937, 47471, 47495
]
#Instacia de la clase data_preprocessing
data_processing = data_processing()

#Creación de datos nuevos
datos_nuevos = data_processing.creacion_datos_nuevos(df_final,id_equipo_local, id_equipo_visitante,
                                                     odd_1, odd_x, odd_2,
                                                     arbitro, estadio, season, ids_lesionados, ids_titulares)

#Entrenamiento del modelo. La línea de código estará comentada, se descomentará para poder reentrenar cuando haya datos nuevos

#Instacia de la clase train_model
train_model = train_model()

#Importación de modelos para el ensemble
modelo_con_cuotas = train_model.importar_modelo('model/football_predictor_cuotas.pkl')
modelo_sin_cuotas = train_model.importar_modelo('model/football_predictor_sin_cuotas.pkl')
modelo_goles_totales = train_model.importar_modelo('model/predict_goles_totales.pkl')


#Prediccion
train_model.prediccion_goles(modelo_goles_totales, datos_nuevos)
#train_model.prediccion_modelo_sin_cuotas(modelo_sin_cuotas, datos_nuevos)
#train_model.prediccion_modelo_con_cuotas(modelo_con_cuotas, datos_nuevos)
#train_model.prediccion_media_probabilidades(modelo_con_cuotas, modelo_sin_cuotas, modelo_goles_totales, datos_nuevos)


