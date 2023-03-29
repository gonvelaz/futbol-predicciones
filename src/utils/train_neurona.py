import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoderColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.arbitro_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.estadio_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.arbitro_encoder.fit(X[['arbitro']])
        self.estadio_encoder.fit(X[['estadio']])
        return self

    def transform(self, X):
        arbitro_encoded = self.arbitro_encoder.transform(X[['arbitro']])
        estadio_encoded = self.estadio_encoder.transform(X[['estadio']])
        return np.concatenate([X.drop(['arbitro', 'estadio'], axis=1).values, arbitro_encoded, estadio_encoded], axis=1)

class train_neurona():
    def __init__(self):
        pass

    def create_modelo_neuronas(self):
        # One-hot encode the 'arbitro' column and target encode the 'estadio' column
        transformer = OneHotEncoderColumnTransformer()
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_neuronas(self, df):
        # Separación del dataframe en X e y (target)
        X = df.drop(['index', 'fixture_id', 'resultado', 'goles_local', 'goles_visitante',
                     'goles_descanso_local', 'goles_descanso_visitante', 'fecha_timestamp'], axis=1)
        y = df['resultado']

        # Eliminación de columnas de lesionados y titulares
        cols_eliminar = []
        for col in X.columns:
            if 'les-' in col or 'titu-' in col:
                cols_eliminar.append(col)

        X_ligero = X.drop(cols_eliminar, axis=1)


        # One-hot encode the 'arbitro' column and target encode the 'estadio' column
        transformer = OneHotEncoderColumnTransformer()
        X_transformed = transformer.fit_transform(X_ligero)

        # Create the model
        model = self.create_modelo_neuronas()

        # Train the model
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

        model.fit(
            X_transformed,
            y,
            batch_size=32,
            epochs=100,
            validation_split = 0.15,
            callbacks=[earlystop]
        )

     

        model.save('./model/futbol_predict_redes.h5')

        return 'Modelo entrenado con éxito'
    

    def import_model_redes(self, ruta_modelo):
        loaded_model = keras.models.load_model('./model/futbol_predict_redes.h5')
        return loaded_model

    def prediccion_redes(self,df_final, modelo, datos_nuevos):
        # One-hot encode the 'arbitro' column and target encode the 'estadio' column
        arbitro_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        estadio_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        arbitro_encoder.fit(df_final[['arbitro']])
        estadio_encoder.fit(df_final[['estadio']])
        arbitro_encoded = arbitro_encoder.transform(datos_nuevos[['arbitro']])
        estadio_encoded = estadio_encoder.transform(datos_nuevos[['estadio']])
        X = np.concatenate([datos_nuevos.drop(['arbitro', 'estadio'], axis=1).values, arbitro_encoded, estadio_encoded], axis=1)
        predictions = modelo.predict(X)
        resultado = np.argmax(predictions, axis=1)[0]
        prob_1 = predictions[0][1]
        prob_X = predictions[0][0]
        prob_2 = predictions[0][2]
        return print(f'El resultado del partido será {resultado}. La probabilidad de 1 es {prob_1}, X es {prob_X} y 2 de {prob_2}')
