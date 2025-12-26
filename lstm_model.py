"""
LSTM Model for Volume Prediction
Implementación del modelo LSTM del artículo MQL5
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class VolumeLSTM:
    """
    Modelo LSTM para predicción de volumen
    Basado en el artículo de MQL5
    """
    
    def __init__(self, hidden_size=32, lookback=10, dropout=0.2):
        """
        Inicializar modelo LSTM
        
        Args:
            hidden_size: Número de unidades en capa LSTM
            lookback: Número de períodos históricos
            dropout: Tasa de dropout para regularización
        """
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.dropout = dropout
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """Construir arquitectura del modelo LSTM"""
        
        model = keras.Sequential([
            # Capa LSTM principal
            layers.LSTM(
                self.hidden_size,
                return_sequences=True,
                input_shape=(self.lookback, 1),
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            ),
            
            # Segunda capa LSTM
            layers.LSTM(
                self.hidden_size // 2,
                return_sequences=False,
                dropout=self.dropout
            ),
            
            # Capa densa
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.dropout),
            
            # Salida
            layers.Dense(1, activation='linear')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Más robusto que MSE
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Modelo LSTM construido: {self.hidden_size} unidades, lookback={self.lookback}")
        
        return model
    
    def prepare_sequences(self, volumes):
        """
        Preparar secuencias para entrenamiento
        
        Args:
            volumes: Array de volúmenes históricos
            
        Returns:
            X, y: Features y targets
        """
        volumes = np.array(volumes).reshape(-1, 1)
        
        # Normalizar datos
        volumes_scaled = self.scaler.fit_transform(volumes)
        
        X, y = [], []
        
        for i in range(len(volumes_scaled) - self.lookback):
            X.append(volumes_scaled[i:i + self.lookback])
            y.append(volumes_scaled[i + self.lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Secuencias preparadas: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def train(self, volumes, epochs=50, validation_split=0.2, verbose=0):
        """
        Entrenar modelo con datos históricos
        
        Args:
            volumes: Lista de volúmenes históricos
            epochs: Número de épocas de entrenamiento
            validation_split: Proporción de datos para validación
            verbose: Nivel de verbosidad
            
        Returns:
            history: Historial de entrenamiento
        """
        if len(volumes) < self.lookback + 100:
            raise ValueError(f"Necesitas al menos {self.lookback + 100} datos históricos")
        
        # Construir modelo si no existe
        if self.model is None:
            self.build_model()
        
        # Preparar datos
        X, y = self.prepare_sequences(volumes)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Entrenar
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Métricas finales
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        
        logger.info(f"Entrenamiento completado:")
        logger.info(f"  Loss: {final_loss:.4f}")
        logger.info(f"  Val Loss: {final_val_loss:.4f}")
        logger.info(f"  MAE: {final_mae:.4f}")
        
        return history
    
    def predict_next_volume(self, recent_volumes):
        """
        Predecir siguiente volumen
        
        Args:
            recent_volumes: Últimos N volúmenes (N = lookback)
            
        Returns:
            Volumen predicho
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        if len(recent_volumes) < self.lookback:
            raise ValueError(f"Se necesitan al menos {self.lookback} volúmenes")
        
        # Tomar últimos lookback períodos
        recent = np.array(recent_volumes[-self.lookback:]).reshape(-1, 1)
        
        # Normalizar
        recent_scaled = self.scaler.transform(recent)
        
        # Reshape para LSTM
        X = recent_scaled.reshape(1, self.lookback, 1)
        
        # Predecir
        prediction_scaled = self.model.predict(X, verbose=0)
        
        # Desnormalizar
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return float(prediction[0, 0])
    
    def predict_derivatives(self, recent_volumes):
        """
        Predecir primera y segunda derivada del volumen
        Similar al código MQL5
        
        Args:
            recent_volumes: Volúmenes recientes
            
        Returns:
            dict con predicciones de derivadas
        """
        # Predecir siguiente volumen
        predicted_volume = self.predict_next_volume(recent_volumes)
        
        # Calcular derivadas actuales
        current_volume = recent_volumes[-1]
        prev_volume = recent_volumes[-2]
        
        current_first_der = current_volume - prev_volume
        
        # Predecir primera derivada
        predicted_first_der = predicted_volume - current_volume
        
        # Calcular segunda derivada actual
        if len(recent_volumes) >= 3:
            prev_first_der = prev_volume - recent_volumes[-3]
            current_second_der = current_first_der - prev_first_der
        else:
            current_second_der = 0
        
        # Predecir segunda derivada
        predicted_second_der = predicted_first_der - current_first_der
        
        return {
            'predicted_volume': predicted_volume,
            'current_volume': current_volume,
            'predicted_first_derivative': predicted_first_der,
            'current_first_derivative': current_first_der,
            'predicted_second_derivative': predicted_second_der,
            'current_second_derivative': current_second_der,
            'is_accelerating_positive': (
                predicted_first_der > current_first_der and
                predicted_second_der > current_second_der and
                predicted_first_der > 0
            ),
            'is_accelerating_negative': (
                predicted_first_der < current_first_der and
                predicted_second_der > current_second_der and
                predicted_first_der < 0
            )
        }
    
    def save(self, model_path='lstm_volume_model.h5', scaler_path='volume_scaler.pkl'):
        """Guardar modelo y scaler"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Guardar modelo
        self.model.save(model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        # Guardar scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler guardado: {scaler_path}")
    
    def load(self, model_path='lstm_volume_model.h5', scaler_path='volume_scaler.pkl'):
        """Cargar modelo y scaler"""
        # Cargar modelo
        self.model = keras.models.load_model(model_path)
        logger.info(f"Modelo cargado: {model_path}")
        
        # Cargar scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler cargado: {scaler_path}")
        
        self.is_trained = True
    
    def evaluate(self, volumes):
        """
        Evaluar modelo con datos de prueba
        
        Args:
            volumes: Volúmenes para evaluación
            
        Returns:
            Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X, y = self.prepare_sequences(volumes)
        
        # Evaluar
        loss, mae = self.model.evaluate(X, y, verbose=0)
        
        # Hacer predicciones
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y)
        
        # Calcular métricas adicionales
        mape = np.mean(np.abs((y_actual - predictions) / (y_actual + 1e-10))) * 100
        rmse = np.sqrt(np.mean((y_actual - predictions) ** 2))
        
        return {
            'loss': float(loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }


def create_and_train_model(volumes, hidden_size=32, lookback=10, 
                           epochs=50, save_path='models'):
    """
    Función helper para crear y entrenar modelo
    
    Args:
        volumes: Datos históricos de volumen
        hidden_size: Tamaño de capa oculta
        lookback: Períodos de lookback
        epochs: Épocas de entrenamiento
        save_path: Directorio para guardar modelo
        
    Returns:
        Modelo entrenado
    """
    # Crear directorio si no existe
    Path(save_path).mkdir(exist_ok=True)
    
    # Crear modelo
    model = VolumeLSTM(hidden_size=hidden_size, lookback=lookback)
    
    # Entrenar
    logger.info("Iniciando entrenamiento...")
    history = model.train(volumes, epochs=epochs, verbose=1)
    
    # Evaluar
    metrics = model.evaluate(volumes[-200:])  # Últimos 200 datos para test
    logger.info(f"Métricas de evaluación: {metrics}")
    
    # Guardar
    model_path = f"{save_path}/lstm_volume_model.h5"
    scaler_path = f"{save_path}/volume_scaler.pkl"
    model.save(model_path, scaler_path)
    
    return model, metrics
