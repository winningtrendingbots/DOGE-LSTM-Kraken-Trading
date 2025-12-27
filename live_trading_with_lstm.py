"""
LIVE TRADING BOT FOR KRAKEN WITH LSTM PREDICTIONS
Ejecuta la estrategia de volumen + OHLC + LSTM en producción
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import ta
import requests

from kraken_trader import KrakenTrader
from telegram_notifier import TelegramNotifier
from state_manager import StateManager
from lstm_model import VolumeLSTM

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN CON LSTM
# ============================================================================

class ProductionConfig:
    """Configuración para trading en vivo con LSTM"""
    
    # API Kraken
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Trading #
    SYMBOL = 'DOGE-USD'
    KRAKEN_PAIR = 'XDOGZUSD'
    INTERVAL = 60
    LOOKBACK_PERIODS = 200
    
    # ===== CONFIGURACIÓN LSTM =====
    USE_LSTM = True                    # Activar predicciones LSTM
    LSTM_LOOKBACK = 10                 # Períodos para LSTM
    LSTM_WEIGHT = 0.5                  # Peso de LSTM vs derivadas (0-1)
    LSTM_CONFIRMATION_REQUIRED = False  # Requiere confirmación de LSTM
    MODEL_PATH = 'models/lstm_volume_model.h5'
    SCALER_PATH = 'models/volume_scaler.pkl'
    
    # Estrategia (igual que antes)
    VOLUME_SMOOTH_PERIODS = 4
    ACCEL_BARS_REQUIRED = 2
    
    # Confirmaciones
    USE_ADX = False
    ADX_THRESHOLD = 20
    USE_OBV = False
    USE_PRICE_MA = False
    USE_RSI_FILTER = False
    USE_BB_FILTER = False
    MIN_CONFIRMATIONS_RATIO = 0.25
    
    # Risk Management
    RISK_PER_TRADE = 0.05
    TP_POINTS = 100
    ATR_STOP_MULTIPLIER = 1.0
    
    # Trailing Stop
    USE_TRAILING_STOP = True
    TRAILING_START = 25
    TRAILING_STEP = 15
    
    # Limits
    PROFIT_CLOSE = 50
    MAX_DAILY_LOSS = -20
    MAX_POSITIONS = 1
    SAME_DIRECTION_ONLY = False
    MAX_BARS_IN_TRADE = 2
    
    # Leverage
    LEVERAGE_MIN = 2
    LEVERAGE_MAX = 10
    
    # Trading Hours
    USE_TRADING_HOURS = False
    TRADE_EUROPEAN_SESSION = True
    TRADE_AMERICAN_SESSION = True
    TRADE_ASIAN_SESSION = True
    SESSION_BUFFER = 15
    
    # Días
    TRADE_MONDAY = True
    TRADE_TUESDAY = True
    TRADE_WEDNESDAY = True
    TRADE_THURSDAY = True
    TRADE_FRIDAY = True


# ============================================================================
# FUNCIONES DE ANÁLISIS TÉCNICO (con LSTM integrado)
# ============================================================================

def calculate_volume_derivatives(df, config):
    """Calcular derivadas de volumen"""
    df = df.copy()
    
    if 'volume' not in df.columns:
        logger.error(f"Columna 'volume' no encontrada. Columnas: {df.columns.tolist()}")
        raise ValueError("Columna 'volume' requerida")
    
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['volume'] = df['volume'].ffill().fillna(0)
    
    if df['volume'].sum() == 0 or df['volume'].mean() < 1:
        logger.warning("Volumen insuficiente, sintetizando...")
        price_range = df['high'] - df['low']
        df['volume'] = price_range * 100000
    
    df['Volume_Smoothed'] = df['volume'].rolling(
        window=config.VOLUME_SMOOTH_PERIODS, 
        min_periods=1
    ).mean()
    
    df['Vol_1st_Der'] = df['Volume_Smoothed'].diff()
    df['Vol_2nd_Der'] = df['Vol_1st_Der'].diff()
    
    df['Vol_1st_Der_Norm'] = (df['Vol_1st_Der'] - df['Vol_1st_Der'].mean()) / (df['Vol_1st_Der'].std() + 1e-10)
    df['Vol_2nd_Der_Norm'] = (df['Vol_2nd_Der'] - df['Vol_2nd_Der'].mean()) / (df['Vol_2nd_Der'].std() + 1e-10)
    
    df['Accel_Positive'] = (
        (df['Vol_1st_Der_Norm'] > 0.1) & 
        (df['Vol_2nd_Der_Norm'] > 0.1)
    ).astype(int)
    
    df['Accel_Negative'] = (
        (df['Vol_1st_Der_Norm'] < -0.1) & 
        (df['Vol_2nd_Der_Norm'] < -0.1)
    ).astype(int)
    
    df['Consecutive_Accel'] = 0
    consec = 0
    for i in range(1, len(df)):
        if df['Accel_Positive'].iloc[i]:
            consec = max(0, consec) + 1
        elif df['Accel_Negative'].iloc[i]:
            consec = min(0, consec) - 1
        else:
            consec = 0
        df.iloc[i, df.columns.get_loc('Consecutive_Accel')] = consec
    
    return df


def add_technical_indicators(df):
    """Agregar indicadores técnicos"""
    df = df.copy()
    
    for col in ['high', 'low', 'close', 'open', 'volume']:
        if col not in df.columns:
            logger.error(f"Columna {col} no encontrada")
            raise ValueError(f"Columna requerida {col} no existe")
        
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[['high', 'low', 'close', 'open', 'volume']] = df[['high', 'low', 'close', 'open', 'volume']].ffill().bfill()
    
    if df[['high', 'low', 'close']].isnull().any().any():
        raise ValueError("Datos inválidos con NaN")
    
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['ADX_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['ADX_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
    
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['OBV_MA'] = df['OBV'].rolling(window=3).mean()
    
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    
    df['Price_Momentum'] = df['close'].diff(5)
    
    return df


def generate_signal_with_lstm(df, lstm_model, config):
    """
    Generar señal de trading con predicciones LSTM
    
    Args:
        df: DataFrame con datos OHLC
        lstm_model: Modelo LSTM cargado
        config: Configuración
        
    Returns:
        signal, indicators, lstm_prediction
    """
    # Señal tradicional de volumen
    last_accel = df['Consecutive_Accel'].iloc[-1]
    
    if last_accel >= config.ACCEL_BARS_REQUIRED:
        vol_signal = 1
    elif last_accel <= -config.ACCEL_BARS_REQUIRED:
        vol_signal = -1
    else:
        vol_signal = 0
    
    # Predicción LSTM si está activada
    lstm_signal = 0
    lstm_prediction = None
    
    if config.USE_LSTM and lstm_model is not None:
        try:
            # Obtener volúmenes recientes
            recent_volumes = df['volume'].iloc[-config.LSTM_LOOKBACK:].values
            
            # Predecir derivadas
            lstm_prediction = lstm_model.predict_derivatives(recent_volumes)
            
            logger.info(f"📊 LSTM Prediction:")
            logger.info(f"   Current Vol: {lstm_prediction['current_volume']:.2f}")
            logger.info(f"   Predicted Vol: {lstm_prediction['predicted_volume']:.2f}")
            logger.info(f"   Accel Positive: {lstm_prediction['is_accelerating_positive']}")
            logger.info(f"   Accel Negative: {lstm_prediction['is_accelerating_negative']}")
            
            # Determinar señal LSTM
            if lstm_prediction['is_accelerating_positive']:
                lstm_signal = 1
            elif lstm_prediction['is_accelerating_negative']:
                lstm_signal = -1
            
        except Exception as e:
            logger.error(f"Error en predicción LSTM: {e}")
    
    # Combinar señales tradicionales y LSTM
    if config.USE_LSTM and lstm_signal != 0:
        if config.LSTM_CONFIRMATION_REQUIRED:
            # Requiere que ambas señales coincidan
            if vol_signal == lstm_signal:
                final_signal = vol_signal
                logger.info(f"✅ LSTM confirma señal tradicional: {final_signal}")
            else:
                final_signal = 0
                logger.info(f"❌ LSTM no confirma señal tradicional")
        else:
            # Promedio ponderado de señales
            combined = vol_signal * (1 - config.LSTM_WEIGHT) + lstm_signal * config.LSTM_WEIGHT
            final_signal = 1 if combined > 0.5 else (-1 if combined < -0.5 else 0)
            logger.info(f"🔀 Señal combinada (tradicional + LSTM): {final_signal}")
    else:
        final_signal = vol_signal
    
    # Confirmaciones adicionales
    if final_signal != 0:
        confirmations = 0
        required = 0
        
        if config.USE_ADX:
            required += 1
            if df['ADX'].iloc[-1] > config.ADX_THRESHOLD:
                if final_signal == 1 and df['ADX_pos'].iloc[-1] > df['ADX_neg'].iloc[-1]:
                    confirmations += 1
                elif final_signal == -1 and df['ADX_neg'].iloc[-1] > df['ADX_pos'].iloc[-1]:
                    confirmations += 1
        
        if config.USE_OBV:
            required += 1
            if final_signal == 1 and df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
                confirmations += 1
            elif final_signal == -1 and df['OBV'].iloc[-1] < df['OBV'].iloc[-2]:
                confirmations += 1
        
        if config.USE_RSI_FILTER:
            rsi = df['RSI'].iloc[-1]
            if final_signal == 1 and rsi > 65:
                return 0, {}, lstm_prediction
            elif final_signal == -1 and rsi < 35:
                return 0, {}, lstm_prediction
        
        if required == 0 or confirmations >= required * config.MIN_CONFIRMATIONS_RATIO:
            indicators = {
                'accel': last_accel,
                'adx': df['ADX'].iloc[-1] if 'ADX' in df.columns else 0,
                'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 0,
                'obv': df['OBV'].iloc[-1] if 'OBV' in df.columns else 0,
                'lstm_signal': lstm_signal,
                'vol_signal': vol_signal
            }
            return final_signal, indicators, lstm_prediction
    
    return 0, {}, lstm_prediction


def can_trade_now(config):
    """Verificar si podemos tradear según horario"""
    if not config.USE_TRADING_HOURS:
        return True
    
    now = datetime.utcnow()
    hour = now.hour
    minute = now.minute
    day_of_week = now.weekday()
    
    days_allowed = [
        config.TRADE_MONDAY,
        config.TRADE_TUESDAY,
        config.TRADE_WEDNESDAY,
        config.TRADE_THURSDAY,
        config.TRADE_FRIDAY,
        False, False
    ]
    
    if not days_allowed[day_of_week]:
        return False
    
    time_minutes = hour * 60 + minute
    buffer = config.SESSION_BUFFER
    
    if config.TRADE_ASIAN_SESSION:
        if buffer <= time_minutes <= (8 * 60 - buffer):
            return True
    
    if config.TRADE_EUROPEAN_SESSION:
        if (7 * 60 + buffer) <= time_minutes <= (16 * 60 - buffer):
            return True
    
    if config.TRADE_AMERICAN_SESSION:
        if (13 * 60 + buffer) <= time_minutes <= (22 * 60 - buffer):
            return True
    
    return False


# ============================================================================
# TRADING LOGIC CON LSTM
# ============================================================================

class LiveTrader:
    """Gestor principal de trading en vivo con LSTM"""
    
    def __init__(self, config):
        self.config = config
        
        # Inicializar módulos
        self.kraken = KrakenTrader(
            config.KRAKEN_API_KEY,
            config.KRAKEN_API_SECRET,
            leverage_min=config.LEVERAGE_MIN,
            leverage_max=config.LEVERAGE_MAX
        )
        
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID
        )
        
        self.state = StateManager()
        
        # Cargar modelo LSTM si está activado
        self.lstm_model = None
        if config.USE_LSTM:
            self.load_lstm_model()
        
        logger.info("LiveTrader inicializado")
    
    def load_lstm_model(self):
        """Cargar modelo LSTM entrenado"""
        try:
            if not Path(self.config.MODEL_PATH).exists():
                logger.warning(f"Modelo LSTM no encontrado en {self.config.MODEL_PATH}")
                logger.warning("Ejecuta train_lstm.py primero o desactiva USE_LSTM")
                return
            
            self.lstm_model = VolumeLSTM(
                hidden_size=32,
                lookback=self.config.LSTM_LOOKBACK
            )
            
            self.lstm_model.load(
                model_path=self.config.MODEL_PATH,
                scaler_path=self.config.SCALER_PATH
            )
            
            logger.info("✅ Modelo LSTM cargado correctamente")
            self.telegram.send_message(
                "🧠 <b>LSTM Model Loaded</b>\n\n"
                "Volume predictions active!"
            )
            
        except Exception as e:
            logger.error(f"Error cargando modelo LSTM: {e}")
            self.lstm_model = None
    
    def run(self):
        """Ejecutar ciclo de trading con LSTM"""
        try:
            logger.info("="*80)
            logger.info(f"Iniciando ciclo de trading: {datetime.now()}")
            logger.info("="*80)
            
            if not self.state.can_trade():
                logger.warning("Trading deshabilitado por límite de pérdida diaria")
                return
            
            if not can_trade_now(self.config):
                logger.info("Fuera de horario de trading")
                return
            
            # Obtener datos
            logger.info("Descargando datos de mercado...")
            df = self.kraken.get_ohlc_data(
                pair=self.config.KRAKEN_PAIR,
                interval=self.config.INTERVAL
            )
            
            if df is None or len(df) < 100:
                logger.error("No se pudieron obtener datos suficientes")
                self.telegram.notify_error("Error obteniendo datos de mercado")
                return
            
            logger.info(f"Datos descargados: {len(df)} velas")
            
            # Calcular indicadores
            logger.info("Calculando indicadores...")
            df = calculate_volume_derivatives(df, self.config)
            df = add_technical_indicators(df)
            
            # Actualizar posiciones existentes
            self.update_open_positions(df)
            
            # Generar señal con LSTM
            signal, indicators, lstm_pred = generate_signal_with_lstm(
                df, self.lstm_model, self.config
            )
            
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            logger.info(f"Precio actual: ${current_price:.2f}")
            logger.info(f"Señal: {signal}")
            
            if signal != 0:
                signal_type = "BUY" if signal > 0 else "SELL"
                logger.info(f"🎯 Señal detectada: {signal_type}")
                
                # Notificación mejorada con LSTM
                indicators_text = {**indicators}
                if lstm_pred:
                    indicators_text['lstm_vol_pred'] = lstm_pred['predicted_volume']
                
                self.telegram.notify_signal(signal_type, current_price, indicators_text)
                
                if self.can_open_position(signal):
                    self.open_position(signal, current_price, atr)
            
            self.state.increment_bars_open()
            
            logger.info("Ciclo completado correctamente")
            
        except Exception as e:
            logger.error(f"Error en ciclo de trading: {e}", exc_info=True)
            self.telegram.notify_error(f"Error en ciclo: {str(e)}")
    
    # ... (resto de métodos igual que antes: can_open_position, open_position, 
    # update_open_positions, etc.)
    
    def can_open_position(self, signal):
        """Verificar si podemos abrir una nueva posición"""
        positions = self.state.get_all_positions()
        
        if len(positions) >= self.config.MAX_POSITIONS:
            logger.info("Máximo de posiciones alcanzado")
            return False
        
        if self.config.SAME_DIRECTION_ONLY and len(positions) > 0:
            first_direction = list(positions.values())[0]['direction']
            new_direction = 'long' if signal > 0 else 'short'
            
            if first_direction != new_direction:
                logger.info("Same direction only: dirección no permitida")
                return False
        
        return True
    
    def open_position(self, signal, current_price, atr):
        """Abrir nueva posición"""
        try:
            logger.info("Abriendo posición...")
            
            balance = self.kraken.get_tradable_balance()
            logger.info(f"Balance disponible: ${balance:.2f}")
            
            if balance < 10:
                logger.warning("Balance insuficiente para operar")
                self.telegram.notify_error("Balance insuficiente")
                return
            
            stop_loss_points = self.config.ATR_STOP_MULTIPLIER * atr / 0.0001
            
            position_calc = self.kraken.calculate_position_size(
                balance=balance,
                risk_percent=self.config.RISK_PER_TRADE,
                stop_loss_points=stop_loss_points,
                current_price=current_price,
                pair=self.config.KRAKEN_PAIR
            )
            
            if position_calc is None:
                logger.error("No se pudo calcular tamaño de posición")
                return
            
            logger.info(f"Tamaño calculado: {position_calc['size']:.4f}")
            logger.info(f"Apalancamiento: {position_calc['leverage']}x")
            
            direction = 'long' if signal > 0 else 'short'
            
            if direction == 'long':
                stop_loss = current_price - (self.config.ATR_STOP_MULTIPLIER * atr)
                take_profit = current_price + (self.config.TP_POINTS * 0.0001)
            else:
                stop_loss = current_price + (self.config.ATR_STOP_MULTIPLIER * atr)
                take_profit = current_price - (self.config.TP_POINTS * 0.0001)
            
            side = 'buy' if signal > 0 else 'sell'
            
            order_result = self.kraken.place_margin_order(
                pair=self.config.KRAKEN_PAIR,
                side=side,
                size=position_calc['size'],
                leverage=position_calc['leverage'],
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_result is None:
                logger.error("Error colocando orden")
                return
            
            position_data = {
                'entry_price': current_price,
                'size': position_calc['size'],
                'direction': direction,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': position_calc['leverage'],
                'trailing_stop': None
            }
            
            self.state.add_position(order_result['txid'], position_data)
            
            order_details = {
                'txid': order_result['txid'],
                'side': side,
                'price': current_price,
                'size': position_calc['size'],
                'cost': position_calc['cost'],
                'leverage': position_calc['leverage'],
                'margin': position_calc['margin_required'],
                'tp': take_profit,
                'sl': stop_loss
            }
            
            self.telegram.notify_order_placed(order_details)
            logger.info(f"✅ Posición abierta: {order_result['txid']}")
            
        except Exception as e:
            logger.error(f"Error abriendo posición: {e}", exc_info=True)
    
    def update_open_positions(self, df):
        """Actualizar posiciones abiertas"""
        positions = self.state.get_all_positions()
        
        if len(positions) == 0:
            return
        
        current_price = df['close'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        for position_id, position in list(positions.items()):
            try:
                self.update_single_position(position_id, position, current_price, high, low, atr)
            except Exception as e:
                logger.error(f"Error actualizando posición {position_id}: {e}")
    
    def update_single_position(self, position_id, position, current_price, high, low, atr):
        """Actualizar una posición individual"""
        
        if position['direction'] == 'long':
            pnl = (current_price - position['entry_price']) * position['size']
            profit_points = (current_price - position['entry_price']) / 0.0001
        else:
            pnl = (position['entry_price'] - current_price) * position['size']
            profit_points = (position['entry_price'] - current_price) / 0.0001
        
        self.state.update_position_extremes(position_id, current_price)
        
        if profit_points >= self.config.PROFIT_CLOSE:
            logger.info(f"Profit target alcanzado para {position_id}")
            self.close_position(position_id, current_price, 'profit_target')
            return
        
        if self.config.USE_TRAILING_STOP and profit_points >= self.config.TRAILING_START:
            self.handle_trailing_stop(position_id, position, profit_points, current_price)
        
        if position['direction'] == 'long':
            if low <= position['stop_loss']:
                self.close_position(position_id, position['stop_loss'], 'stop_loss')
                return
            elif high >= position['take_profit']:
                self.close_position(position_id, position['take_profit'], 'take_profit')
                return
        else:
            if high >= position['stop_loss']:
                self.close_position(position_id, position['stop_loss'], 'stop_loss')
                return
            elif low <= position['take_profit']:
                self.close_position(position_id, position['take_profit'], 'take_profit')
                return
        
        if position['bars_open'] >= self.config.MAX_BARS_IN_TRADE:
            logger.info(f"Time limit alcanzado para {position_id}")
            self.close_position(position_id, current_price, 'time_limit')
            return
    
    def handle_trailing_stop(self, position_id, position, profit_points, current_price):
        """Manejar trailing stop"""
        
        if not position.get('trailing_activated'):
            position['trailing_activated'] = True
            position['highest_profit'] = profit_points
            self.state.update_position(position_id, position)
            logger.info(f"Trailing stop activado para {position_id}")
        else:
            if profit_points > position['highest_profit']:
                position['highest_profit'] = profit_points
                
                trail_level = position['highest_profit'] - self.config.TRAILING_STEP
                
                if position['direction'] == 'long':
                    new_sl = position['entry_price'] + (trail_level * 0.0001)
                    if new_sl > position['stop_loss']:
                        position['stop_loss'] = new_sl
                        self.state.update_position(position_id, position)
                        
                        pnl = (current_price - position['entry_price']) * position['size']
                        self.telegram.notify_trailing_stop_update(position_id, new_sl, pnl)
                else:
                    new_sl = position['entry_price'] - (trail_level * 0.0001)
                    if new_sl < position['stop_loss']:
                        position['stop_loss'] = new_sl
                        self.state.update_position(position_id, position)
                        
                        pnl = (position['entry_price'] - current_price) * position['size']
                        self.telegram.notify_trailing_stop_update(position_id, new_sl, pnl)
    
    def close_position(self, position_id, exit_price, reason):
        """Cerrar posición"""
        try:
            position = self.state.get_position(position_id)
            
            if position is None:
                return
            
            if position['direction'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['size']
            
            return_pct = (pnl / (position['entry_price'] * position['size'])) * 100
            
            success = self.kraken.close_position(
                pair=self.config.KRAKEN_PAIR,
                position_type=position['direction']
            )
            
            if not success:
                logger.error(f"Error cerrando posición en Kraken: {position_id}")
            
            self.state.add_trade(pnl)
            self.state.remove_position(position_id)
            
            if self.state.is_daily_loss_limit_hit(self.config.MAX_DAILY_LOSS):
                self.telegram.notify_daily_loss_limit(
                    self.state.get_daily_profit(),
                    self.config.MAX_DAILY_LOSS
                )
            
            entry_time = datetime.fromisoformat(position['entry_time'])
            duration = str(datetime.now() - entry_time).split('.')[0]
            
            close_details = {
                'txid': position_id,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': return_pct,
                'reason': reason,
                'duration': duration,
                'balance': self.kraken.get_tradable_balance()
            }
            
            self.telegram.notify_order_closed(close_details)
            logger.info(f"✅ Posición cerrada: {position_id} | P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error cerrando posición: {e}", exc_info=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal"""
    
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
    
    config = ProductionConfig()
    
    trader = LiveTrader(config)
    
    trader.run()


if __name__ == "__main__":
    main()
