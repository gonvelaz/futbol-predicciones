import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report



class train_model():
    def __init__(self):
        pass

    def train_xgbc_cuotas(self,df):
        #Dividimos en los datos de entrenamiento y la clasificación de los datos de entrenamiento que usaremos para entrenar el modelo
        X = df.drop(['index','fixture_id','resultado', 'goles_local','goles_totales', 'goles_visitante','goles_descanso_local','goles_descanso_visitante','fecha_timestamp' ], axis=1)
        y = df['resultado']

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA()),
            ('xgb', xgb.XGBClassifier())
        ])

        xgb_param = {
        'pca__n_components': [25,30,35],
        'xgb__n_estimators': [300, 500, 700],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [27,25],
        'xgb__subsample': [0.5, 0.8],
        'xgb__colsample_bytree': [0.5, 0.6],
        'xgb__min_child_weight': [1, 2],
        'xgb__gamma': [0]
        }

        gs_xgb = RandomizedSearchCV(
                                pipeline_xgb,
                                xgb_param,
                                n_iter=1000,
                                cv=3,
                                scoring="accuracy",
                                verbose=1,
                                n_jobs=-1
                            )
        
        modelo = gs_xgb.fit(X, y)

        with open(os.path.join('model','football_predictor_cuotas.pkl'), 'wb') as file:
            pickle.dump(modelo, file)

        return "Modelo entrenado con éxito y guardado en 'football_predictor.pkl'."
    
    def train_xgbc_sin_cuotas(self,df):
        #Dividimos en los datos de entrenamiento y la clasificación de los datos de entrenamiento que usaremos para entrenar el modelo
        X = df.drop(['index','fixture_id','resultado', 'goles_local', 'goles_visitante','goles_totales','goles_descanso_local','goles_descanso_visitante','fecha_timestamp','odd_1','odd_2','odd_x','odd_mas_25','odd_menos_25' ], axis=1)
        y = df['resultado']

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA()),
            ('xgb', xgb.XGBClassifier())
        ])

        xgb_param = {
        'pca__n_components': [25,30,35],
        'xgb__n_estimators': [300, 500, 700],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [27,25],
        'xgb__subsample': [0.5, 0.8],
        'xgb__colsample_bytree': [0.5, 0.6],
        'xgb__min_child_weight': [1, 2],
        'xgb__gamma': [0]
        }

        gs_xgb = RandomizedSearchCV(
                                pipeline_xgb,
                                xgb_param,
                                n_iter=1000,
                                cv=3,
                                scoring="accuracy",
                                verbose=1,
                                n_jobs=-1
                            )
        
        modelo = gs_xgb.fit(X, y)

        with open(os.path.join('model','football_predictor_sin_cuotas.pkl'), 'wb') as file:
            pickle.dump(modelo, file)

        return "Modelo entrenado con éxito y guardado en 'football_predictor.pkl'."


    def importar_modelo(self, ruta_modelo):
        with open(ruta_modelo, 'rb') as archivo:
            gs_xgb = pickle.load(archivo)
        return gs_xgb


    def prediccion_modelo_sin_cuotas(self, modelo, datos_nuevos):
        datos_nuevos_sin_cuotas = datos_nuevos.drop(['odd_1','odd_x','odd_2'], axis=1) 
        resultado = modelo.predict(datos_nuevos_sin_cuotas)
        probabilidades = modelo.predict_proba(datos_nuevos_sin_cuotas)
        return print(f'El resultado del partido será {resultado[0]} según el modelo sin cuotas. Las probabilidades son de X - {probabilidades[0][0]*100}%, 1 - {probabilidades[0][1]*100} y 2 - {probabilidades[0][2]*100}')

    def prediccion_modelo_con_cuotas(self, modelo, datos_nuevos):
        resultado = modelo.predict(datos_nuevos)
        probabilidades = modelo.predict_proba(datos_nuevos)
        return print(f'El resultado del partido será {resultado[0]} según el modelo con cuotas. Las probabilidades son de X - {probabilidades[0][0]*100}%, 1 - {probabilidades[0][1]*100} y 2 - {probabilidades[0][2]*100}')
    
    
    

    def train_ensamble(self, model_cuotas,model_sin_cuotas, df):
        # Separar X e y del dataframe
        X = df.drop(['index','fixture_id','resultado', 'goles_local', 'goles_visitante','goles_descanso_local','goles_descanso_visitante','fecha_timestamp'], axis=1)
        y = df['resultado']
        
        # Realizar predicciones con los modelos y añadir los resultados como columnas en X
        X['prediccion_cuotas'] = model_cuotas.predict(X)
        X['prediccion_sin_cuotas'] = model_sin_cuotas.predict(X)
        
        # Nueva columna prediccion_3
        pred_cuotas_proba = model_cuotas.predict_proba(X) # Probabilidades de cada clase para la prediccion_1
        pred_cuotas_extra = [] # Lista para almacenar los resultados de la nueva prediccion
        for i in range(len(pred_cuotas_proba)):
            if pred_cuotas_proba[i][0] + 0.15 > pred_cuotas_proba[i][1] and pred_cuotas_proba[i][0] + 0.15 > pred_cuotas_proba[i][2]:
                pred_cuotas_extra.append(0)
            else:
                pred_cuotas_extra.append(X['prediccion_cuotas'][i])
        X['pred_cuotas_extra'] = pred_cuotas_extra
        
        # Nueva columna prediccion_4
        pred_sin_cuotas_proba = model_sin_cuotas.predict_proba(X) # Probabilidades de cada clase para la prediccion_2
        pred_sin_cuotas_extra = [] # Lista para almacenar los resultados de la nueva prediccion
        for i in range(len(pred_sin_cuotas_proba)):
            if pred_sin_cuotas_proba[i][0] + 0.15 > pred_sin_cuotas_proba[i][1] and pred_sin_cuotas_proba[i][0] + 0.15 > pred_sin_cuotas_proba[i][2]:
                pred_sin_cuotas_extra.append(0)
            else:
                pred_sin_cuotas_extra.append(X['prediccion_sin_cuotas'][i])
        X['pred_sin_cuotas_extra'] = pred_sin_cuotas_extra

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA()),
            ('xgb', xgb.XGBClassifier())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_param = {
        'pca__n_components': [25,30,35],
        'xgb__n_estimators': [300, 500, 700],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [27,25],
        'xgb__subsample': [0.5, 0.8],
        'xgb__colsample_bytree': [0.5, 0.6],
        'xgb__min_child_weight': [1, 2],
        'xgb__gamma': [0]
        }

        gs_xgb = GridSearchCV(
                                pipeline_xgb,
                                xgb_param,
                                cv=3,
                                scoring="accuracy",
                                verbose=1,
                                n_jobs=-1
                            )
        
        ensemble_model = gs_xgb.fit(X_train, y_train)

        with open(os.path.join('model','football_predictor_ensemble.pkl'), 'wb') as file:
            pickle.dump(ensemble_model, file)

        y_pred = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Classification Report \n", report)

        return "Modelo entrenado con éxito y guardado en 'football_predictor_ensemble.pkl'."
    
    import numpy as np

    def prediccion_media_probabilidades(self, modelo_con_cuotas, modelo_sin_cuotas, modelo_goles_totales, datos_nuevos):
            
        # Predicción con el primer modelo
        probabilidad_1_cuotas = modelo_con_cuotas.predict_proba(datos_nuevos)
        
        # Predicción con el segundo modelo
        probabilidad_sin_cuotas = modelo_sin_cuotas.predict_proba(datos_nuevos)
        
        # Modelos adicionales que suman 0.15 a la probabilidad de 0 en modelo_con_cuotas y modelo_sin_cuotas
        probabilidad_1_cuotas_mod = probabilidad_1_cuotas.copy()
        probabilidad_sin_cuotas_mod = probabilidad_sin_cuotas.copy()
        probabilidad_1_cuotas_mod[:, 0] += 0.15
        probabilidad_sin_cuotas_mod[:, 0] += 0.15
        
        # Ajustar las sumas de probabilidades a 1
        probabilidad_1_cuotas_mod /= np.sum(probabilidad_1_cuotas_mod, axis=1)[:, np.newaxis]
        probabilidad_sin_cuotas_mod /= np.sum(probabilidad_sin_cuotas_mod, axis=1)[:, np.newaxis]

        # Calculando la media de las probabilidades de clasificación de los cuatro modelos
        probabilidades_media = (probabilidad_1_cuotas + probabilidad_sin_cuotas + probabilidad_1_cuotas_mod + probabilidad_sin_cuotas_mod) / 4
        
        # Clasificación final basada en la media de las probabilidades
        resultado_final = np.argmax(probabilidades_media, axis=1)

        prediccion_goles_totales = modelo_goles_totales.predict(datos_nuevos)
        

        print(f'El total de goles sera de {prediccion_goles_totales[0]} goles')
        
        # Imprimir los resultados finales
        return print(f'El resultado del partido será {resultado_final[0]} según la media de modelos. Las probabilidades son de X - {probabilidades_media[0][0]*100}%, 1 - {probabilidades_media[0][1]*100} y 2 - {probabilidades_media[0][2]*100}')


    def train_xgbc_goles_totales(self,df_partidos):
        X = df_partidos.drop(['index', 'fixture_id','resultado', 'goles_local','goles_totales', 'goles_visitante','goles_descanso_local','goles_descanso_visitante','fecha_timestamp' ], axis=1)
        y = df_partidos['goles_totales']

        # Columnas categóricas
        categorical_columns = ['arbitro', 'estadio']

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            #('scaler', StandardScaler()), #En este caso el escalado de variables funciona mucho peor
            ('pca', PCA()),
            ('xgb', xgb.XGBRegressor())
        ])

        param_grid = {
            'pca__n_components': [30],
            'xgb__n_estimators': [700],  # Número de árboles a construir
            'xgb__learning_rate': [ 0.01],  # Tasa de aprendizaje
            'xgb__max_depth': [25],  # Profundidad máxima de un árbol
            'xgb__min_child_weight': [5],  # Cantidad mínima de observaciones por nodo
            'xgb__subsample': [0.5],  # Porcentaje de muestras a utilizar para cada árbol
            'xgb__colsample_bytree': [1],  # Porcentaje de características a utilizar para cada árbol
            'xgb__gamma': [ 0.3],  # Reducción mínima de pérdida requerida para hacer una partición adicional
            'xgb__reg_alpha': [1],  # Regularización L1 en los pesos
            'xgb__reg_lambda': [2]  # Regularización L2 en los pesos
        }
        

        gs_xgbr = GridSearchCV(pipeline_xgb,
                            param_grid,
                            cv=3,
                            scoring="neg_mean_absolute_error",
                            verbose=1,
                            n_jobs=-1)
        
        gs_xgbr.fit(X,y)
        
        with open(os.path.join('model','predict_goles_totales.pkl'), 'wb') as file:
            pickle.dump(gs_xgbr, file)

        return "Modelo entrenado con éxito y guardado en 'predict_goles_totales.pkl'."
    

    def prediccion_goles(self, modelo_goles_totales, datos_nuevos):
        prediccion = modelo_goles_totales.predict(datos_nuevos)
        return print(f'El total de goles sera de {prediccion[0]} goles')
    
    def train_xgbc_goles_totales(self,df_partidos):
        X = df_partidos.drop(['index', 'fixture_id','resultado', 'goles_local','goles_totales', 'goles_visitante','goles_descanso_local','goles_descanso_visitante','fecha_timestamp' ], axis=1)
        y = df_partidos['goles_totales']

        # Columnas categóricas
        categorical_columns = ['arbitro', 'estadio']

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            #('scaler', StandardScaler()), #En este caso el escalado de variables funciona mucho peor
            ('pca', PCA()),
            ('xgb', xgb.XGBRegressor())
        ])

        param_grid = {
            'pca__n_components': [30],
            'xgb__n_estimators': [700],  # Número de árboles a construir
            'xgb__learning_rate': [ 0.01],  # Tasa de aprendizaje
            'xgb__max_depth': [25],  # Profundidad máxima de un árbol
            'xgb__min_child_weight': [5],  # Cantidad mínima de observaciones por nodo
            'xgb__subsample': [0.5],  # Porcentaje de muestras a utilizar para cada árbol
            'xgb__colsample_bytree': [1],  # Porcentaje de características a utilizar para cada árbol
            'xgb__gamma': [ 0.3],  # Reducción mínima de pérdida requerida para hacer una partición adicional
            'xgb__reg_alpha': [1],  # Regularización L1 en los pesos
            'xgb__reg_lambda': [2]  # Regularización L2 en los pesos
        }
        

        gs_xgbr = GridSearchCV(pipeline_xgb,
                            param_grid,
                            cv=3,
                            scoring="neg_mean_absolute_error",
                            verbose=1,
                            n_jobs=-1)
        
        gs_xgbr.fit(X,y)
        
        with open(os.path.join('model','predict_goles_totales.pkl'), 'wb') as file:
            pickle.dump(gs_xgbr, file)

        return "Modelo entrenado con éxito y guardado en 'predict_goles_totales.pkl'."
    
