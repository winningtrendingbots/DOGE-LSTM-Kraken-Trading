"""
Script para entrenar modelo LSTM diariamente
Usa YFINANCE para obtener 2 años de datos horarios
Se ejecuta automáticamente cada madrugada via GitHub Actions
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

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
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Símbolo para yfinance
    SYMBOL = 'DOGE-USD'
    
    # Configuración temporal - 2 años con intervalos de 1 hora
    INTERVAL = '1h'
    HISTORICAL_PERIOD = '2y'
    
    # Parámetros LSTM
    LSTM_HIDDEN_SIZE = 32
    LSTM_LOOKBACK = 10
    LSTM_EPOCHS = 50
    LSTM_DROPOUT = 0.2
    
    # Rutas
    MODEL_DIR = 'models'
    MODEL_PATH = f'{MODEL_DIR}/lstm_volume_model.h5'
    SCALER_PATH = f'{MODEL_DIR}/volume_scaler.pkl'
    METRICS_PATH = f'{MODEL_DIR}/training_metrics.txt'


def download_historical_data(config):
    """
    Descargar datos históricos usando yfinance
    2 años con intervalos de 1 hora
    """
    logger.info("="*80)
    logger.info(f"DESCARGANDO DATOS HISTÓRICOS CON YFINANCE")
    logger.info("="*80)
    
    try:
        logger.info(f"Símbolo: {config.SYMBOL}")
        logger.info(f"Período: {config.HISTORICAL_PERIOD}")
        logger.info(f"Intervalo: {config.INTERVAL}")
        logger.info("")
        logger.info("Descargando desde Yahoo Finance...")
        
        ticker = yf.Ticker(config.SYMBOL)
        
        df = ticker.history(
            period=config.HISTORICAL_PERIOD,
            interval=config.INTERVAL
        )
        
        if df is None or len(df) == 0:
            raise ValueError("No se pudieron descargar datos de yfinance")
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"DATOS DESCARGADOS EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"Total de períodos: {len(df)}")
        logger.info(f"Fecha inicial: {df.index[0]}")
        logger.info(f"Fecha final: {df.index[-1]}")
        logger.info(f"Duración: {df.index[-1] - df.index[0]}")
        logger.info("")
        
        logger.info("Estadísticas de Volumen:")
        logger.info(f"  Promedio: {df['Volume'].mean():,.0f}")
        logger.info(f"  Máximo: {df['Volume'].max():,.0f}")
        logger.info(f"  Mínimo: {df['Volume'].min():,.0f}")
        logger.info(f"  Mediana: {df['Volume'].median():,.0f}")
        logger.info("="*80)
        logger.info("")
        
        min_required = config.LSTM_LOOKBACK + 200
        if len(df) < min_required:
            logger.warning(f"⚠️ Se recomienda al menos {min_required} períodos")
            logger.warning(f"   Solo se descargaron {len(df)} períodos")
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Error descargando datos de yfinance: {e}")
        raise


def prepare_volume_data(df):
    """Preparar datos de volumen para entrenamiento LSTM"""
    logger.info("Preparando datos de volumen para LSTM...")
    
    volumes = df['Volume'].values
    
    if np.isnan(volumes).any():
        logger.warning("Encontrados valores NaN en volumen, limpiando...")
        volumes = pd.Series(volumes).fillna(method='ffill').fillna(0).values
    
    zero_count = (volumes == 0).sum()
    if zero_count > 0:
        logger.warning(f"Encontrados {zero_count} volúmenes en cero")
        min_nonzero = volumes[volumes > 0].min() if (volumes > 0).any() else 1.0
        volumes[volumes == 0] = min_nonzero
    
    logger.info(f"Datos de volumen preparados: {len(volumes)} puntos")
    
    return volumes


def train_model(volumes, config):
    """Entrenar modelo LSTM"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO ENTRENAMIENTO LSTM")
    logger.info("="*80)
    logger.info(f"Puntos de datos: {len(volumes)}")
    logger.info(f"Lookback: {config.LSTM_LOOKBACK}")
    logger.info(f"Hidden size: {config.LSTM_HIDDEN_SIZE}")
    logger.info(f"Épocas: {config.LSTM_EPOCHS}")
    logger.info("="*80 + "\n")
    
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    
    model, metrics = create_and_train_model(
        volumes=volumes,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        lookback=config.LSTM_LOOKBACK,
        epochs=config.LSTM_EPOCHS,
        save_path=config.MODEL_DIR
    )
    
    logger.info("\n" + "="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    logger.info(f"  MAE: {metrics['mae']:,.2f}")
    logger.info(f"  RMSE: {metrics['rmse']:,.2f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info("="*80 + "\n")
    
    return model, metrics


def validate_model(model, volumes, config):
    """Validar modelo con predicciones"""
    logger.info("="*80)
    logger.info("VALIDACIÓN DEL MODELO")
    logger.info("="*80)
    
    recent_volumes = volumes[-config.LSTM_LOOKBACK:]
    derivatives = model.predict_derivatives(volumes[-config.LSTM_LOOKBACK:])
    
    logger.info(f"📊 Volumen actual: {derivatives['current_volume']:,.0f}")
    logger.info(f"🔮 Volumen predicho: {derivatives['predicted_volume']:,.0f}")
    
    pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                  / derivatives['current_volume'] * 100)
    logger.info(f"📈 Cambio esperado: {pct_change:+.2f}%")
    
    if derivatives['is_accelerating_positive']:
        logger.info("  🟢 ACELERACIÓN POSITIVA")
    elif derivatives['is_accelerating_negative']:
        logger.info("  🔴 ACELERACIÓN NEGATIVA")
    else:
        logger.info("  ⚪ SIN ACELERACIÓN CLARA")
    
    logger.info("="*80 + "\n")
    
    return derivatives


def save_metrics(metrics, derivatives, config):
    """Guardar métricas"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(config.METRICS_PATH, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSTM TRAINING METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Symbol: {config.SYMBOL}\n")
        f.write(f"Interval: {config.INTERVAL}\n")
        f.write(f"Period: {config.HISTORICAL_PERIOD}\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Lookback: {config.LSTM_LOOKBACK}\n")
        f.write(f"  Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"  Epochs: {config.LSTM_EPOCHS}\n\n")
        
        f.write("Training Metrics:\n")
        f.write(f"  MAE: {metrics['mae']:,.2f}\n")
        f.write(f"  RMSE: {metrics['rmse']:,.2f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n\n")
        
        f.write("Validation:\n")
        f.write(f"  Current Volume: {derivatives['current_volume']:,.0f}\n")
        f.write(f"  Predicted Volume: {derivatives['predicted_volume']:,.0f}\n")
        f.write(f"  Accelerating Positive: {derivatives['is_accelerating_positive']}\n")
        f.write(f"  Accelerating Negative: {derivatives['is_accelerating_negative']}\n")
    
    logger.info(f"✅ Métricas guardadas en: {config.METRICS_PATH}")


def notify_training_complete(telegram, metrics, derivatives, config, training_time):
    """Notificar completado"""
    
    mape = metrics['mape']
    quality = "Excelente ✅" if mape < 5 else ("Bueno ✅" if mape < 10 else "Aceptable ⚠️")
    
    pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                  / derivatives['current_volume'] * 100)
    
    message = f"""
🧠 <b>LSTM MODEL TRAINING COMPLETED</b>

📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💱 <b>Symbol:</b> {config.SYMBOL}
⏱️ <b>Data:</b> {config.HISTORICAL_PERIOD} @ {config.INTERVAL}
⏳ <b>Time:</b> {training_time}

📈 <b>Performance:</b>
• MAE: {metrics['mae']:,.2f}
• RMSE: {metrics['rmse']:,.2f}
• MAPE: {metrics['mape']:.2f}%
• Quality: {quality}

🧪 <b>Validation:</b>
• Current: {derivatives['current_volume']:,.0f}
• Predicted: {derivatives['predicted_volume']:,.0f}
• Change: {pct_change:+.2f}%
• Accel ⬆️: {'✅' if derivatives['is_accelerating_positive'] else '❌'}
• Accel ⬇️: {'✅' if derivatives['is_accelerating_negative'] else '❌'}

✅ <b>Model ready!</b>
    """
    
    telegram.send_message(message.strip())


def main():
    """Función principal"""
    start_time = datetime.now()
    
    try:
        logger.info("\n" + "🧠 "*40)
        logger.info("LSTM TRAINING - 2 YEARS @ 1H INTERVALS")
        logger.info("🧠 "*40 + "\n")
        
        config = LSTMTrainingConfig()
        
        required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Variables faltantes: {missing}")
            sys.exit(1)
        
        telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID
        )
        
        telegram.send_message(
            f"🧠 <b>LSTM Training Started</b>\n\n"
            f"📊 {config.SYMBOL}\n"
            f"⏱️ {config.HISTORICAL_PERIOD} @ {config.INTERVAL}\n\n"
            f"Downloading data..."
        )
        
        # Descargar con yfinance
        df = download_historical_data(config)
        
        # Preparar volúmenes
        volumes = prepare_volume_data(df)
        
        # Entrenar
        model, metrics = train_model(volumes, config)
        
        # Validar
        derivatives = validate_model(model, volumes, config)
        
        # Guardar
        save_metrics(metrics, derivatives, config)
        
        # Notificar
        training_time = str(datetime.now() - start_time).split('.')[0]
        notify_training_complete(telegram, metrics, derivatives, config, training_time)
        
        logger.info("\n" + "✅ "*40)
        logger.info("TRAINING COMPLETED")
        logger.info("✅ "*40 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        
        try:
            telegram = TelegramNotifier(
                os.getenv('TELEGRAM_BOT_TOKEN'),
                os.getenv('TELEGRAM_CHAT_ID')
            )
            telegram.send_message(
                f"❌ <b>LSTM Training Failed</b>\n\n"
                f"Error: {str(e)}\n\n"
                f"Check logs."
            )
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
