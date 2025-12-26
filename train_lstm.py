"""
Script para entrenar modelo LSTM diariamente
Se ejecuta automáticamente cada madrugada via GitHub Actions
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

from kraken_trader import KrakenTrader
from telegram_notifier import TelegramNotifier
from lstm_model import VolumeLSTM, create_and_train_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMTrainingConfig:
    """Configuración para entrenamiento LSTM"""
    
    # Kraken API
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Par de trading
    KRAKEN_PAIR = 'XXRPZUSD'
    INTERVAL = 15  # Minutos
    
    # Parámetros LSTM
    LSTM_HIDDEN_SIZE = 32
    LSTM_LOOKBACK = 10
    LSTM_EPOCHS = 50
    LSTM_DROPOUT = 0.2
    
    # Datos históricos
    HISTORICAL_DAYS = 30  # Días de historia para entrenar
    
    # Rutas
    MODEL_DIR = 'models'
    MODEL_PATH = f'{MODEL_DIR}/lstm_volume_model.h5'
    SCALER_PATH = f'{MODEL_DIR}/volume_scaler.pkl'
    METRICS_PATH = f'{MODEL_DIR}/training_metrics.txt'


def download_historical_volumes(trader, config):
    """
    Descargar volúmenes históricos de Kraken
    
    Args:
        trader: Instancia de KrakenTrader
        config: Configuración
        
    Returns:
        Array de volúmenes
    """
    logger.info("Descargando datos históricos de Kraken...")
    
    # Calcular cuántas velas necesitamos
    minutes_per_day = 24 * 60
    candles_per_day = minutes_per_day // config.INTERVAL
    total_candles = candles_per_day * config.HISTORICAL_DAYS
    
    # Descargar datos OHLC
    df = trader.get_ohlc_data(
        pair=config.KRAKEN_PAIR,
        interval=config.INTERVAL
    )
    
    if df is None or len(df) < 200:
        raise ValueError("No se pudieron obtener suficientes datos históricos")
    
    # Extraer volúmenes
    volumes = df['volume'].values
    
    logger.info(f"Descargados {len(volumes)} períodos de volumen")
    logger.info(f"Rango de fechas: {df.index[0]} a {df.index[-1]}")
    logger.info(f"Volumen promedio: {volumes.mean():.2f}")
    logger.info(f"Volumen máximo: {volumes.max():.2f}")
    logger.info(f"Volumen mínimo: {volumes.min():.2f}")
    
    return volumes


def train_model(volumes, config):
    """
    Entrenar modelo LSTM con datos históricos
    
    Args:
        volumes: Array de volúmenes
        config: Configuración
        
    Returns:
        Modelo entrenado y métricas
    """
    logger.info("\n" + "="*60)
    logger.info("INICIANDO ENTRENAMIENTO LSTM")
    logger.info("="*60)
    
    # Crear directorio de modelos
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    
    # Crear y entrenar modelo
    model, metrics = create_and_train_model(
        volumes=volumes,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        lookback=config.LSTM_LOOKBACK,
        epochs=config.LSTM_EPOCHS,
        save_path=config.MODEL_DIR
    )
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*60)
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAPE: {metrics['mape']:.2f}%")
    logger.info("="*60 + "\n")
    
    return model, metrics


def validate_model(model, volumes, config):
    """
    Validar modelo con predicciones de ejemplo
    
    Args:
        model: Modelo entrenado
        volumes: Volúmenes históricos
        config: Configuración
        
    Returns:
        Resultados de validación
    """
    logger.info("Validando modelo con predicciones de ejemplo...")
    
    # Tomar últimos períodos para validación
    recent_volumes = volumes[-config.LSTM_LOOKBACK:]
    
    # Hacer predicción
    predicted_volume = model.predict_next_volume(recent_volumes)
    
    # Calcular derivadas predichas
    derivatives = model.predict_derivatives(volumes[-config.LSTM_LOOKBACK:])
    
    logger.info(f"\nPredicción de validación:")
    logger.info(f"  Volumen actual: {derivatives['current_volume']:.2f}")
    logger.info(f"  Volumen predicho: {derivatives['predicted_volume']:.2f}")
    logger.info(f"  Primera derivada actual: {derivatives['current_first_derivative']:.2f}")
    logger.info(f"  Primera derivada predicha: {derivatives['predicted_first_derivative']:.2f}")
    logger.info(f"  Segunda derivada actual: {derivatives['current_second_derivative']:.2f}")
    logger.info(f"  Segunda derivada predicha: {derivatives['predicted_second_derivative']:.2f}")
    logger.info(f"  ¿Acelerando positivo?: {derivatives['is_accelerating_positive']}")
    logger.info(f"  ¿Acelerando negativo?: {derivatives['is_accelerating_negative']}")
    
    return derivatives


def save_metrics(metrics, derivatives, config):
    """Guardar métricas de entrenamiento"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(config.METRICS_PATH, 'w') as f:
        f.write(f"LSTM Training Metrics\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"="*60 + "\n\n")
        
        f.write(f"Training Metrics:\n")
        f.write(f"  MAE: {metrics['mae']:.4f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"  Loss: {metrics['loss']:.4f}\n\n")
        
        f.write(f"Validation Example:\n")
        f.write(f"  Current Volume: {derivatives['current_volume']:.2f}\n")
        f.write(f"  Predicted Volume: {derivatives['predicted_volume']:.2f}\n")
        f.write(f"  Accelerating Positive: {derivatives['is_accelerating_positive']}\n")
        f.write(f"  Accelerating Negative: {derivatives['is_accelerating_negative']}\n")
    
    logger.info(f"Métricas guardadas en: {config.METRICS_PATH}")


def notify_training_complete(telegram, metrics, derivatives, config):
    """Enviar notificación de entrenamiento completado"""
    
    message = f"""
🧠 <b>LSTM MODEL TRAINING COMPLETED</b>

📅 <b>Training Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💱 <b>Pair:</b> {config.KRAKEN_PAIR}
⏱ <b>Interval:</b> {config.INTERVAL} min
🔢 <b>Lookback:</b> {config.LSTM_LOOKBACK} periods

📊 <b>Model Performance:</b>
• MAE: {metrics['mae']:.4f}
• RMSE: {metrics['rmse']:.4f}
• MAPE: {metrics['mape']:.2f}%

🧪 <b>Validation Test:</b>
• Current Vol: {derivatives['current_volume']:.0f}
• Predicted Vol: {derivatives['predicted_volume']:.0f}
• Accelerating ⬆️: {'✅' if derivatives['is_accelerating_positive'] else '❌'}
• Accelerating ⬇️: {'✅' if derivatives['is_accelerating_negative'] else '❌'}

✅ <b>Model ready for predictions!</b>
    """
    
    telegram.send_message(message.strip())


def main():
    """Función principal de entrenamiento"""
    try:
        logger.info("\n" + "🧠"*30)
        logger.info("LSTM DAILY TRAINING - STARTING")
        logger.info("🧠"*30 + "\n")
        
        # Validar variables de entorno
        config = LSTMTrainingConfig()
        
        required_vars = [
            'KRAKEN_API_KEY',
            'KRAKEN_API_SECRET',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Variables de entorno faltantes: {missing}")
            sys.exit(1)
        
        # Inicializar servicios
        trader = KrakenTrader(
            config.KRAKEN_API_KEY,
            config.KRAKEN_API_SECRET
        )
        
        telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID
        )
        
        # Notificar inicio
        telegram.send_message(
            f"🧠 <b>LSTM Training Started</b>\n\n"
            f"Downloading {config.HISTORICAL_DAYS} days of data..."
        )
        
        # Descargar datos históricos
        volumes = download_historical_volumes(trader, config)
        
        # Entrenar modelo
        model, metrics = train_model(volumes, config)
        
        # Validar modelo
        derivatives = validate_model(model, volumes, config)
        
        # Guardar métricas
        save_metrics(metrics, derivatives, config)
        
        # Notificar completado
        notify_training_complete(telegram, metrics, derivatives, config)
        
        logger.info("\n" + "✅"*30)
        logger.info("LSTM TRAINING COMPLETED SUCCESSFULLY")
        logger.info("✅"*30 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}", exc_info=True)
        
        # Notificar error
        try:
            telegram = TelegramNotifier(
                os.getenv('TELEGRAM_BOT_TOKEN'),
                os.getenv('TELEGRAM_CHAT_ID')
            )
            telegram.send_message(
                f"❌ <b>LSTM Training Failed</b>\n\n"
                f"Error: {str(e)}\n\n"
                f"Check logs for details."
            )
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
