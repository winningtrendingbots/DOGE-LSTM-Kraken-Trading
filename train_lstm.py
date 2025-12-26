"""
Script para entrenar modelo LSTM diariamente
Usa YFINANCE para obtener datos históricos profundos
Se ejecuta automáticamente cada madrugada via GitHub Actions
"""

import os
import sys
import logging
from datetime import datetime, timedelta
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
    
    # Telegram (para notificaciones)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Símbolo de trading
    # Para DOGE/USD en yfinance usamos 'DOGE-USD'
    # Para XRP/USD sería 'XRP-USD'
    # Para BTC/USD sería 'BTC-USD'
    SYMBOL = 'DOGE-USD'
    
    # Marco temporal
    # Opciones en yfinance: '1m', '5m', '15m', '30m', '1h', '1d'
    # Nota: datos de minutos solo están disponibles para los últimos 7-60 días
    # Para datos históricos más largos, usa '1h' o '1d'
    INTERVAL = '1h'  # Usar 1 hora para tener 2 años de historia
    
    # Parámetros LSTM
    LSTM_HIDDEN_SIZE = 32       # Neuronas en capa LSTM
    LSTM_LOOKBACK = 10          # Períodos históricos para predicción
    LSTM_EPOCHS = 50            # Épocas de entrenamiento
    LSTM_DROPOUT = 0.2          # Dropout para regularización
    
    # Datos históricos
    # Con INTERVAL='1h', podemos pedir 2 años completos
    # Esto nos da aproximadamente 17,520 puntos de datos
    HISTORICAL_PERIOD = '2y'    # 2 años ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
    
    # Rutas
    MODEL_DIR = 'models'
    MODEL_PATH = f'{MODEL_DIR}/lstm_volume_model.h5'
    SCALER_PATH = f'{MODEL_DIR}/volume_scaler.pkl'
    METRICS_PATH = f'{MODEL_DIR}/training_metrics.txt'


def download_historical_data(config):
    """
    Descargar datos históricos usando yfinance
    
    Esta función es mucho más simple y poderosa que usar Kraken directamente
    porque yfinance puede descargar años de datos en una sola llamada.
    
    Args:
        config: Configuración del entrenamiento
        
    Returns:
        DataFrame con columnas: Open, High, Low, Close, Volume
    """
    logger.info("="*80)
    logger.info(f"DESCARGANDO DATOS HISTÓRICOS CON YFINANCE")
    logger.info("="*80)
    
    try:
        # Descargar datos usando yfinance
        # Esto es increíblemente simple comparado con hacer llamadas
        # múltiples a la API de Kraken con paginación
        logger.info(f"Símbolo: {config.SYMBOL}")
        logger.info(f"Período: {config.HISTORICAL_PERIOD}")
        logger.info(f"Intervalo: {config.INTERVAL}")
        logger.info("")
        logger.info("Descargando desde Yahoo Finance...")
        
        ticker = yf.Ticker(config.SYMBOL)
        
        # Descargar datos históricos
        # period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
        # interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo'
        df = ticker.history(
            period=config.HISTORICAL_PERIOD,
            interval=config.INTERVAL
        )
        
        if df is None or len(df) == 0:
            raise ValueError("No se pudieron descargar datos de yfinance")
        
        # Información sobre los datos descargados
        logger.info("")
        logger.info("="*80)
        logger.info(f"DATOS DESCARGADOS EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"Total de períodos: {len(df)}")
        logger.info(f"Fecha inicial: {df.index[0]}")
        logger.info(f"Fecha final: {df.index[-1]}")
        logger.info(f"Duración: {df.index[-1] - df.index[0]}")
        logger.info("")
        
        # Información sobre el volumen
        logger.info("Estadísticas de Volumen:")
        logger.info(f"  Promedio: {df['Volume'].mean():,.0f}")
        logger.info(f"  Máximo: {df['Volume'].max():,.0f}")
        logger.info(f"  Mínimo: {df['Volume'].min():,.0f}")
        logger.info(f"  Mediana: {df['Volume'].median():,.0f}")
        logger.info("="*80)
        logger.info("")
        
        # Verificar que tenemos suficientes datos
        min_required = config.LSTM_LOOKBACK + 200  # Mínimo para entrenar bien
        if len(df) < min_required:
            logger.warning(f"⚠️ Se recomienda al menos {min_required} períodos")
            logger.warning(f"   Solo se descargaron {len(df)} períodos")
            logger.warning(f"   Considera usar un período más largo o intervalo más corto")
        
        # Retornar solo las columnas que necesitamos
        # yfinance ya proporciona las columnas en el formato correcto
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Error descargando datos de yfinance: {e}")
        raise


def prepare_volume_data(df):
    """
    Preparar datos de volumen para entrenamiento LSTM
    
    Args:
        df: DataFrame con datos OHLCV
        
    Returns:
        Array de numpy con volúmenes
    """
    logger.info("Preparando datos de volumen para LSTM...")
    
    # Extraer volumen
    volumes = df['Volume'].values
    
    # Verificar que no haya valores NaN
    if np.isnan(volumes).any():
        logger.warning("Encontrados valores NaN en volumen, limpiando...")
        volumes = pd.Series(volumes).fillna(method='ffill').fillna(0).values
    
    # Verificar que no haya volúmenes cero (pueden causar problemas)
    zero_count = (volumes == 0).sum()
    if zero_count > 0:
        logger.warning(f"Encontrados {zero_count} volúmenes en cero")
        # Reemplazar ceros con el mínimo no-cero
        min_nonzero = volumes[volumes > 0].min() if (volumes > 0).any() else 1.0
        volumes[volumes == 0] = min_nonzero
    
    logger.info(f"Datos de volumen preparados: {len(volumes)} puntos")
    
    return volumes


def train_model(volumes, config):
    """
    Entrenar modelo LSTM con datos históricos
    
    Args:
        volumes: Array de volúmenes históricos
        config: Configuración
        
    Returns:
        Modelo entrenado y métricas
    """
    logger.info("\n" + "="*80)
    logger.info("INICIANDO ENTRENAMIENTO LSTM")
    logger.info("="*80)
    logger.info(f"Puntos de datos: {len(volumes)}")
    logger.info(f"Lookback: {config.LSTM_LOOKBACK}")
    logger.info(f"Hidden size: {config.LSTM_HIDDEN_SIZE}")
    logger.info(f"Épocas: {config.LSTM_EPOCHS}")
    logger.info(f"Dropout: {config.LSTM_DROPOUT}")
    logger.info("="*80 + "\n")
    
    # Crear directorio de modelos
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    
    # Crear y entrenar modelo
    # La función create_and_train_model maneja todo el pipeline:
    # - Crear arquitectura LSTM
    # - Preparar secuencias de entrenamiento
    # - Normalizar datos
    # - Entrenar con early stopping
    # - Evaluar performance
    # - Guardar modelo y scaler
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
    logger.info("Métricas de Performance:")
    logger.info(f"  MAE (Mean Absolute Error): {metrics['mae']:,.2f}")
    logger.info(f"  RMSE (Root Mean Square Error): {metrics['rmse']:,.2f}")
    logger.info(f"  MAPE (Mean Absolute % Error): {metrics['mape']:.2f}%")
    logger.info("")
    logger.info("Interpretación de MAPE:")
    if metrics['mape'] < 5:
        logger.info("  ✅ Excelente (<5%) - Predicciones muy precisas")
    elif metrics['mape'] < 10:
        logger.info("  ✅ Bueno (5-10%) - Predicciones confiables")
    elif metrics['mape'] < 20:
        logger.info("  ⚠️ Aceptable (10-20%) - Predicciones útiles pero con margen de error")
    else:
        logger.info("  ❌ Necesita mejora (>20%) - Considera más datos o ajustar hiperparámetros")
    logger.info("="*80 + "\n")
    
    return model, metrics


def validate_model(model, volumes, config):
    """
    Validar modelo con predicciones de ejemplo
    
    Esta función hace una predicción real con los últimos datos
    para verificar que el modelo está funcionando correctamente.
    
    Args:
        model: Modelo entrenado
        volumes: Volúmenes históricos
        config: Configuración
        
    Returns:
        Resultados de validación
    """
    logger.info("="*80)
    logger.info("VALIDACIÓN DEL MODELO")
    logger.info("="*80)
    
    # Tomar últimos períodos para validación
    recent_volumes = volumes[-config.LSTM_LOOKBACK:]
    
    logger.info(f"Usando últimos {config.LSTM_LOOKBACK} períodos para predicción:")
    for i, vol in enumerate(recent_volumes, 1):
        logger.info(f"  Período {i}: {vol:,.0f}")
    
    logger.info("")
    logger.info("Generando predicción...")
    
    # Hacer predicción de volumen
    predicted_volume = model.predict_next_volume(recent_volumes)
    
    # Calcular derivadas predichas
    derivatives = model.predict_derivatives(volumes[-config.LSTM_LOOKBACK:])
    
    logger.info("")
    logger.info("RESULTADOS DE LA PREDICCIÓN:")
    logger.info("="*80)
    logger.info(f"📊 Volumen actual: {derivatives['current_volume']:,.0f}")
    logger.info(f"🔮 Volumen predicho: {derivatives['predicted_volume']:,.0f}")
    logger.info("")
    
    # Calcular cambio porcentual
    pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                  / derivatives['current_volume'] * 100)
    logger.info(f"📈 Cambio esperado: {pct_change:+.2f}%")
    logger.info("")
    
    logger.info("Primera Derivada (velocidad de cambio):")
    logger.info(f"  Actual: {derivatives['current_first_derivative']:,.0f}")
    logger.info(f"  Predicha: {derivatives['predicted_first_derivative']:,.0f}")
    logger.info("")
    
    logger.info("Segunda Derivada (aceleración):")
    logger.info(f"  Actual: {derivatives['current_second_derivative']:,.0f}")
    logger.info(f"  Predicha: {derivatives['predicted_second_derivative']:,.0f}")
    logger.info("")
    
    logger.info("Señales de Trading:")
    if derivatives['is_accelerating_positive']:
        logger.info("  🟢 ACELERACIÓN POSITIVA - Señal alcista")
        logger.info("     El volumen está aumentando y acelerando al alza")
    elif derivatives['is_accelerating_negative']:
        logger.info("  🔴 ACELERACIÓN NEGATIVA - Señal bajista")
        logger.info("     El volumen está disminuyendo y acelerando a la baja")
    else:
        logger.info("  ⚪ SIN ACELERACIÓN CLARA - Sin señal fuerte")
        logger.info("     El volumen no muestra patrón de aceleración definido")
    
    logger.info("="*80 + "\n")
    
    return derivatives


def save_metrics(metrics, derivatives, config):
    """Guardar métricas de entrenamiento en archivo"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(config.METRICS_PATH, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSTM TRAINING METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Symbol: {config.SYMBOL}\n")
        f.write(f"Interval: {config.INTERVAL}\n")
        f.write(f"Historical Period: {config.HISTORICAL_PERIOD}\n")
        f.write("\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Lookback: {config.LSTM_LOOKBACK}\n")
        f.write(f"  Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"  Epochs: {config.LSTM_EPOCHS}\n")
        f.write(f"  Dropout: {config.LSTM_DROPOUT}\n")
        f.write("\n")
        
        f.write("Training Metrics:\n")
        f.write(f"  MAE: {metrics['mae']:,.2f}\n")
        f.write(f"  RMSE: {metrics['rmse']:,.2f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"  Loss: {metrics['loss']:.4f}\n")
        f.write("\n")
        
        f.write("Validation Example:\n")
        f.write(f"  Current Volume: {derivatives['current_volume']:,.0f}\n")
        f.write(f"  Predicted Volume: {derivatives['predicted_volume']:,.0f}\n")
        f.write(f"  Change: {((derivatives['predicted_volume'] - derivatives['current_volume']) / derivatives['current_volume'] * 100):+.2f}%\n")
        f.write(f"  Accelerating Positive: {derivatives['is_accelerating_positive']}\n")
        f.write(f"  Accelerating Negative: {derivatives['is_accelerating_negative']}\n")
        f.write("\n")
        f.write("="*80 + "\n")
    
    logger.info(f"✅ Métricas guardadas en: {config.METRICS_PATH}")


def notify_training_complete(telegram, metrics, derivatives, config, training_time):
    """Enviar notificación de entrenamiento completado"""
    
    # Determinar calidad del modelo
    mape = metrics['mape']
    if mape < 5:
        quality = "Excelente ✅"
    elif mape < 10:
        quality = "Bueno ✅"
    elif mape < 20:
        quality = "Aceptable ⚠️"
    else:
        quality = "Necesita mejora ❌"
    
    # Calcular cambio esperado
    pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                  / derivatives['current_volume'] * 100)
    
    message = f"""
🧠 <b>LSTM MODEL TRAINING COMPLETED</b>

📅 <b>Training Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
💱 <b>Symbol:</b> {config.SYMBOL}
⏱️ <b>Interval:</b> {config.INTERVAL}
📊 <b>Historical Data:</b> {config.HISTORICAL_PERIOD}
⏳ <b>Training Time:</b> {training_time}

📈 <b>Model Performance:</b>
• MAE: {metrics['mae']:,.2f}
• RMSE: {metrics['rmse']:,.2f}
• MAPE: {metrics['mape']:.2f}%
• Quality: {quality}

🧪 <b>Validation Test:</b>
• Current Vol: {derivatives['current_volume']:,.0f}
• Predicted Vol: {derivatives['predicted_volume']:,.0f}
• Expected Change: {pct_change:+.2f}%
• Accelerating ⬆️: {'✅' if derivatives['is_accelerating_positive'] else '❌'}
• Accelerating ⬇️: {'✅' if derivatives['is_accelerating_negative'] else '❌'}

✅ <b>Model ready for trading predictions!</b>
    """
    
    telegram.send_message(message.strip())


def main():
    """Función principal de entrenamiento"""
    
    start_time = datetime.now()
    
    try:
        logger.info("\n" + "🧠 "*40)
        logger.info("LSTM DAILY TRAINING - STARTING")
        logger.info("Using yfinance for deep historical data")
        logger.info("🧠 "*40 + "\n")
        
        # Cargar configuración
        config = LSTMTrainingConfig()
        
        # Validar variables de entorno requeridas
        required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            logger.error(f"Variables de entorno faltantes: {missing}")
            sys.exit(1)
        
        # Inicializar notificador de Telegram
        telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID
        )
        
        # Notificar inicio
        telegram.send_message(
            f"🧠 <b>LSTM Training Started</b>\n\n"
            f"📊 Symbol: {config.SYMBOL}\n"
            f"⏱️ Interval: {config.INTERVAL}\n"
            f"📅 Period: {config.HISTORICAL_PERIOD}\n\n"
            f"Downloading data from Yahoo Finance..."
        )
        
        # Descargar datos históricos usando yfinance
        df = download_historical_data(config)
        
        # Preparar datos de volumen
        volumes = prepare_volume_data(df)
        
        # Entrenar modelo
        model, metrics = train_model(volumes, config)
        
        # Validar modelo
        derivatives = validate_model(model, volumes, config)
        
        # Guardar métricas
        save_metrics(metrics, derivatives, config)
        
        # Calcular tiempo de entrenamiento
        training_time = str(datetime.now() - start_time).split('.')[0]
        
        # Notificar completado
        notify_training_complete(telegram, metrics, derivatives, config, training_time)
        
        logger.info("\n" + "✅ "*40)
        logger.info("LSTM TRAINING COMPLETED SUCCESSFULLY")
        logger.info("✅ "*40 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR EN ENTRENAMIENTO: {e}", exc_info=True)
        
        # Notificar error
        try:
            telegram = TelegramNotifier(
                os.getenv('TELEGRAM_BOT_TOKEN'),
                os.getenv('TELEGRAM_CHAT_ID')
            )
            telegram.send_message(
                f"❌ <b>LSTM Training Failed</b>\n\n"
                f"Error: {str(e)}\n\n"
                f"Check GitHub Actions logs for details."
            )
        except:
            logger.error("No se pudo enviar notificación de error")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
