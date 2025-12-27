"""
Script para entrenar modelo LSTM diariamente
Usa CoinGecko para obtener datos históricos profundos de criptomonedas
Se ejecuta automáticamente cada madrugada via GitHub Actions
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from pycoingecko import CoinGeckoAPI
import time

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
    """Configuración para entrenamiento LSTM con CoinGecko"""
    
    # Telegram (para notificaciones)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Símbolo de trading
    # Para CoinGecko, usamos el ID de la moneda, no el símbolo del ticker
    # Dogecoin: 'dogecoin'
    # Bitcoin: 'bitcoin'
    # Ethereum: 'ethereum'
    # Ripple: 'ripple'
    COIN_ID = 'dogecoin'
    COIN_SYMBOL = 'DOGE'  # Para display en notificaciones
    
    # Moneda de cotización
    VS_CURRENCY = 'usd'
    
    # Período histórico
    # CoinGecko proporciona toda la historia disponible de la moneda
    # Podemos especificar cuántos días queremos hacia atrás
    HISTORICAL_DAYS = 730  # 2 años completos de datos diarios
    
    # Parámetros LSTM
    LSTM_HIDDEN_SIZE = 32       # Neuronas en capa LSTM
    LSTM_LOOKBACK = 10          # Períodos históricos para predicción
    LSTM_EPOCHS = 50            # Épocas de entrenamiento
    LSTM_DROPOUT = 0.2          # Dropout para regularización
    
    # Rutas
    MODEL_DIR = 'models'
    MODEL_PATH = f'{MODEL_DIR}/lstm_volume_model.h5'
    SCALER_PATH = f'{MODEL_DIR}/volume_scaler.pkl'
    METRICS_PATH = f'{MODEL_DIR}/training_metrics.txt'


def download_historical_data_coingecko(config):
    """
    Descargar datos históricos usando CoinGecko
    
    CoinGecko es superior a Yahoo Finance para criptomonedas porque:
    1. Especializado en cripto, no en acciones tradicionales
    2. Datos más limpios y consistentes
    3. Cobertura completa de miles de altcoins
    4. API gratuita sin autenticación requerida
    5. Datos históricos profundos disponibles
    
    Args:
        config: Configuración del entrenamiento
        
    Returns:
        DataFrame con columnas: timestamp, price, volume
    """
    logger.info("="*80)
    logger.info(f"DESCARGANDO DATOS HISTÓRICOS CON COINGECKO")
    logger.info("="*80)
    
    try:
        # Inicializar cliente de CoinGecko
        # La API gratuita no requiere clave, pero tiene rate limits
        cg = CoinGeckoAPI()
        
        logger.info(f"Coin ID: {config.COIN_ID}")
        logger.info(f"Moneda: {config.VS_CURRENCY.upper()}")
        logger.info(f"Período: {config.HISTORICAL_DAYS} días")
        logger.info("")
        logger.info("Descargando desde CoinGecko API...")
        
        # Calcular timestamp de inicio
        # CoinGecko usa timestamps Unix (segundos desde 1970)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.HISTORICAL_DAYS)
        
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        logger.info(f"Rango de fechas:")
        logger.info(f"  Desde: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Hasta: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Descargar datos usando market_chart_range
        # Este endpoint proporciona precios, volúmenes y market caps históricos
        # Para períodos largos, CoinGecko agrupa automáticamente en intervalos diarios
        logger.info("Realizando llamada a la API...")
        data = cg.get_coin_market_chart_range_by_id(
            id=config.COIN_ID,
            vs_currency=config.VS_CURRENCY,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        
        # Pequeña pausa para respetar rate limits de la API
        time.sleep(1)
        
        # Verificar que recibimos datos
        if not data or 'prices' not in data or 'total_volumes' not in data:
            raise ValueError("CoinGecko no retornó datos válidos")
        
        # Los datos vienen en formato:
        # prices: [[timestamp_ms, price], ...]
        # total_volumes: [[timestamp_ms, volume], ...]
        
        # Convertir a DataFrame
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Los timestamps pueden no coincidir exactamente, así que hacemos merge
        df = pd.merge(prices_df, volumes_df, on='timestamp', how='inner')
        
        # Convertir timestamp de milisegundos a datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        
        # Ordenar por fecha
        df = df.sort_index()
        
        # Remover duplicados si los hay
        df = df[~df.index.duplicated(keep='first')]
        
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
        
        # Información sobre el precio
        logger.info("Estadísticas de Precio:")
        logger.info(f"  Promedio: ${df['price'].mean():.6f}")
        logger.info(f"  Máximo: ${df['price'].max():.6f}")
        logger.info(f"  Mínimo: ${df['price'].min():.6f}")
        logger.info(f"  Último: ${df['price'].iloc[-1]:.6f}")
        logger.info("")
        
        # Información sobre el volumen
        logger.info("Estadísticas de Volumen:")
        logger.info(f"  Promedio: ${df['volume'].mean():,.0f}")
        logger.info(f"  Máximo: ${df['volume'].max():,.0f}")
        logger.info(f"  Mínimo: ${df['volume'].min():,.0f}")
        logger.info(f"  Mediana: ${df['volume'].median():,.0f}")
        logger.info("="*80)
        logger.info("")
        
        # Verificar que tenemos suficientes datos
        min_required = config.LSTM_LOOKBACK + 200  # Mínimo para entrenar bien
        if len(df) < min_required:
            logger.warning(f"⚠️ Se recomienda al menos {min_required} períodos")
            logger.warning(f"   Solo se descargaron {len(df)} períodos")
            logger.warning(f"   Considera aumentar HISTORICAL_DAYS")
        
        # Retornar DataFrame con las columnas necesarias
        return df[['price', 'volume']]
        
    except Exception as e:
        logger.error(f"Error descargando datos de CoinGecko: {e}")
        logger.error(f"Asegúrate de que el COIN_ID '{config.COIN_ID}' es válido")
        logger.error("Ejemplos válidos: 'bitcoin', 'ethereum', 'dogecoin', 'ripple'")
        raise


def prepare_volume_data(df):
    """
    Preparar datos de volumen para entrenamiento LSTM
    
    CoinGecko proporciona volúmenes en USD, lo cual es perfecto para nuestro análisis.
    Los volúmenes representan el valor total negociado en dólares durante cada período.
    
    Args:
        df: DataFrame con columnas price y volume
        
    Returns:
        Array de numpy con volúmenes
    """
    logger.info("Preparando datos de volumen para LSTM...")
    
    # Extraer volumen
    volumes = df['volume'].values
    
    # Verificar que no haya valores NaN
    if np.isnan(volumes).any():
        logger.warning("Encontrados valores NaN en volumen, limpiando...")
        volumes = pd.Series(volumes).fillna(method='ffill').fillna(0).values
    
    # Verificar que no haya volúmenes cero
    zero_count = (volumes == 0).sum()
    if zero_count > 0:
        logger.warning(f"Encontrados {zero_count} volúmenes en cero")
        # Reemplazar ceros con el mínimo no-cero
        min_nonzero = volumes[volumes > 0].min() if (volumes > 0).any() else 1.0
        volumes[volumes == 0] = min_nonzero
        logger.info(f"Ceros reemplazados con valor mínimo: {min_nonzero:,.0f}")
    
    # Verificar que los volúmenes son razonables
    # Para criptomonedas, volúmenes demasiado bajos pueden indicar datos defectuosos
    if volumes.mean() < 100:
        logger.warning("⚠️ Volúmenes promedio muy bajos, verifica los datos")
    
    logger.info(f"Datos de volumen preparados: {len(volumes)} puntos")
    logger.info(f"Rango de volúmenes: ${volumes.min():,.0f} - ${volumes.max():,.0f}")
    
    return volumes


def train_model(volumes, config):
    """
    Entrenar modelo LSTM con datos históricos
    
    El modelo aprenderá a predecir volúmenes futuros basándose en patrones
    históricos. Esto es útil para detectar aceleraciones de volumen antes
    de que ocurran movimientos significativos de precio.
    
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
    logger.info(f"  MAE (Mean Absolute Error): ${metrics['mae']:,.2f}")
    logger.info(f"  RMSE (Root Mean Square Error): ${metrics['rmse']:,.2f}")
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
    
    Hace una predicción real con los últimos datos disponibles para
    verificar que el modelo está funcionando correctamente y proporcionar
    una muestra de cómo se comportará en producción.
    
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
        logger.info(f"  Período {i}: ${vol:,.0f}")
    
    logger.info("")
    logger.info("Generando predicción...")
    
    # Hacer predicción completa con derivadas
    derivatives = model.predict_derivatives(volumes[-config.LSTM_LOOKBACK:])
    
    logger.info("")
    logger.info("RESULTADOS DE LA PREDICCIÓN:")
    logger.info("="*80)
    logger.info(f"📊 Volumen actual: ${derivatives['current_volume']:,.0f}")
    logger.info(f"🔮 Volumen predicho: ${derivatives['predicted_volume']:,.0f}")
    logger.info("")
    
    # Calcular cambio porcentual
    pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                  / derivatives['current_volume'] * 100)
    logger.info(f"📈 Cambio esperado: {pct_change:+.2f}%")
    logger.info("")
    
    logger.info("Primera Derivada (velocidad de cambio):")
    logger.info(f"  Actual: ${derivatives['current_first_derivative']:,.0f}")
    logger.info(f"  Predicha: ${derivatives['predicted_first_derivative']:,.0f}")
    logger.info("")
    
    logger.info("Segunda Derivada (aceleración):")
    logger.info(f"  Actual: ${derivatives['current_second_derivative']:,.0f}")
    logger.info(f"  Predicha: ${derivatives['predicted_second_derivative']:,.0f}")
    logger.info("")
    
    logger.info("Señales de Trading:")
    if derivatives['is_accelerating_positive']:
        logger.info("  🟢 ACELERACIÓN POSITIVA - Señal alcista")
        logger.info("     El volumen está aumentando y acelerando al alza")
        logger.info("     Esto típicamente precede movimientos de precio significativos")
    elif derivatives['is_accelerating_negative']:
        logger.info("  🔴 ACELERACIÓN NEGATIVA - Señal bajista")
        logger.info("     El volumen está disminuyendo y acelerando a la baja")
        logger.info("     Puede indicar pérdida de interés o consolidación")
    else:
        logger.info("  ⚪ SIN ACELERACIÓN CLARA - Sin señal fuerte")
        logger.info("     El volumen no muestra patrón de aceleración definido")
        logger.info("     Esperar confirmación antes de operar")
    
    logger.info("="*80 + "\n")
    
    return derivatives


def save_metrics(metrics, derivatives, config):
    """Guardar métricas de entrenamiento en archivo"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(config.METRICS_PATH, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSTM TRAINING METRICS (CoinGecko Data)\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Coin: {config.COIN_SYMBOL} ({config.COIN_ID})\n")
        f.write(f"VS Currency: {config.VS_CURRENCY.upper()}\n")
        f.write(f"Historical Days: {config.HISTORICAL_DAYS}\n")
        f.write("\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Lookback: {config.LSTM_LOOKBACK}\n")
        f.write(f"  Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"  Epochs: {config.LSTM_EPOCHS}\n")
        f.write(f"  Dropout: {config.LSTM_DROPOUT}\n")
        f.write("\n")
        
        f.write("Training Metrics:\n")
        f.write(f"  MAE: ${metrics['mae']:,.2f}\n")
        f.write(f"  RMSE: ${metrics['rmse']:,.2f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"  Loss: {metrics['loss']:.4f}\n")
        f.write("\n")
        
        f.write("Validation Example:\n")
        f.write(f"  Current Volume: ${derivatives['current_volume']:,.0f}\n")
        f.write(f"  Predicted Volume: ${derivatives['predicted_volume']:,.0f}\n")
        pct_change = ((derivatives['predicted_volume'] - derivatives['current_volume']) 
                      / derivatives['current_volume'] * 100)
        f.write(f"  Change: {pct_change:+.2f}%\n")
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
💎 <b>Coin:</b> {config.COIN_SYMBOL} ({config.COIN_ID})
💱 <b>Currency:</b> {config.VS_CURRENCY.upper()}
📊 <b>Historical Data:</b> {config.HISTORICAL_DAYS} days
⏳ <b>Training Time:</b> {training_time}
🔧 <b>Data Source:</b> CoinGecko API

📈 <b>Model Performance:</b>
• MAE: ${metrics['mae']:,.2f}
• RMSE: ${metrics['rmse']:,.2f}
• MAPE: {metrics['mape']:.2f}%
• Quality: {quality}

🧪 <b>Validation Test:</b>
• Current Vol: ${derivatives['current_volume']:,.0f}
• Predicted Vol: ${derivatives['predicted_volume']:,.0f}
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
        logger.info("Using CoinGecko for deep cryptocurrency data")
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
            f"💎 Coin: {config.COIN_SYMBOL}\n"
            f"📅 Period: {config.HISTORICAL_DAYS} days\n"
            f"🔧 Source: CoinGecko API\n\n"
            f"Downloading historical data..."
        )
        
        # Descargar datos históricos usando CoinGecko
        df = download_historical_data_coingecko(config)
        
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
