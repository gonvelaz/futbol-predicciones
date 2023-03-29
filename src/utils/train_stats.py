import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

class train_model_stats():
    def __init__(self):
        pass

    def train_total_goles(self, df):
        # Crear una lista de columnas para eliminar que contengan 'titu-' o 'les-'
        columns_to_drop = df.filter(like='titu-').columns.tolist() + df.filter(like='les-').columns.tolist()

        # Añadir las columnas adicionales a la lista de columnas a eliminar
        columns_to_drop += ['index', 'fixture_id', 'resultado', 'goles_local', 'goles_totales', 'goles_visitante', 'goles_descanso_local', 'goles_descanso_visitante', 'fecha_timestamp']

        # Eliminar las columnas en la lista columns_to_drop de DataFrame df_partidos
        X = df.drop(columns_to_drop, axis=1)
        y = df['goles_totales']

        X_train, X_test, y_train, y_test = train_test_split()

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
            ('scaler', StandardScaler()), #En este caso el escalado de variables funciona mucho peor
            ('xgb', xgb.XGBRegressor())
        ])

        param_grid = {
            'xgb__n_estimators': [10,15,20],  # Número de árboles a construir
            'xgb__learning_rate': [],  # Tasa de aprendizaje
            'xgb__max_depth': [25],  # Profundidad máxima de un árbol
            'xgb__min_child_weight': [5],  # Cantidad mínima de observaciones por nodo
            'xgb__subsample': [0.5],  # Porcentaje de muestras a utilizar para cada árbol
            'xgb__colsample_bytree': [1],  # Porcentaje de características a utilizar para cada árbol
            'xgb__gamma': [ 0.3],  # Reducción mínima de pérdida requerida para hacer una partición adicional
            'xgb__reg_alpha': [1],  # Regularización L1 en los pesos
            'xgb__reg_lambda': [2]  # Regularización L2 en los pesos
        }