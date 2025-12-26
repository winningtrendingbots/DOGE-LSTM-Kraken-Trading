"""
Módulo para interactuar con la API de Kraken
Maneja la obtención de datos OHLC, balance, y ejecución de órdenes
"""

import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


class KrakenTrader:
    """Cliente para operar con Kraken"""
    
    def __init__(self, api_key, api_secret, leverage_min=2, leverage_max=10):
        """
        Inicializar conexión con Kraken
        
        Args:
            api_key: Clave API de Kraken
            api_secret: Secreto API de Kraken
            leverage_min: Apalancamiento mínimo
            leverage_max: Apalancamiento máximo
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.leverage_min = leverage_min
        self.leverage_max = leverage_max
        
        # Inicializar cliente Kraken
        self.api = krakenex.API(key=api_key, secret=api_secret)
        self.kraken = KrakenAPI(self.api)
        
        logger.info("KrakenTrader inicializado")
    
    def get_ohlc_data(self, pair='XXRPZUSD', interval=15, since=None):
        """
        Obtener datos OHLC de Kraken
        
        Args:
            pair: Par de trading (ej: 'XXRPZUSD')
            interval: Intervalo en minutos (1, 5, 15, 30, 60, 240, 1440)
            since: Timestamp desde donde obtener datos
            
        Returns:
            DataFrame con columnas: time, open, high, low, close, volume
        """
        try:
            logger.info(f"Obteniendo datos OHLC para {pair}, intervalo={interval}")
            
            # Obtener datos de Kraken
            ohlc, last = self.kraken.get_ohlc_data(pair, interval=interval, since=since)
            
            # Renombrar columnas para consistencia
            ohlc.columns = ['time', 'open', 'high', 'low', 'close', 
                           'vwap', 'volume', 'count']
            
            # Asegurar tipos numéricos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')
            
            # Eliminar NaN
            ohlc = ohlc.dropna()
            
            logger.info(f"Obtenidos {len(ohlc)} registros OHLC")
            
            return ohlc
            
        except Exception as e:
            logger.error(f"Error obteniendo datos OHLC: {e}")
            return None
    
    def get_tradable_balance(self):
        """
        Obtener balance disponible para trading
        
        Returns:
            Balance en USD
        """
        try:
            balance = self.kraken.get_account_balance()
            
            # Buscar balance en USD o equivalente
            if 'ZUSD' in balance:
                usd_balance = float(balance['ZUSD'])
            else:
                # Si no hay USD directo, buscar otras monedas
                usd_balance = 0
                for currency, amount in balance.items():
                    if float(amount) > 0:
                        logger.info(f"Balance {currency}: {amount}")
                        # Convertir a USD aproximado (necesitarías hacer llamadas adicionales para precisión)
                        usd_balance += float(amount)
            
            logger.info(f"Balance disponible: ${usd_balance:.2f}")
            return usd_balance
            
        except Exception as e:
            logger.error(f"Error obteniendo balance: {e}")
            return 0
    
    def calculate_position_size(self, balance, risk_percent, stop_loss_points, 
                               current_price, pair='XXRPZUSD'):
        """
        Calcular tamaño de posición basado en riesgo
        
        Args:
            balance: Balance disponible
            risk_percent: Porcentaje de riesgo (0.01 = 1%)
            stop_loss_points: Puntos de stop loss
            current_price: Precio actual
            pair: Par de trading
            
        Returns:
            Dict con size, leverage, cost, margin_required
        """
        try:
            # Cantidad a arriesgar
            risk_amount = balance * risk_percent
            
            # Calcular tamaño de posición
            # stop_loss_points está en pips (0.0001)
            stop_loss_price = stop_loss_points * 0.0001
            
            # Tamaño sin apalancamiento
            position_size = risk_amount / stop_loss_price
            
            # Ajustar por precio actual
            position_cost = position_size * current_price
            
            # Calcular apalancamiento óptimo
            leverage = min(
                self.leverage_max,
                max(self.leverage_min, int(position_cost / balance) + 1)
            )
            
            # Ajustar tamaño con apalancamiento
            position_size = position_size / leverage
            
            # Margen requerido
            margin_required = (position_size * current_price) / leverage
            
            result = {
                'size': round(position_size, 4),
                'leverage': leverage,
                'cost': round(position_cost, 2),
                'margin_required': round(margin_required, 2)
            }
            
            logger.info(f"Posición calculada: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculando posición: {e}")
            return None
    
    def place_margin_order(self, pair, side, size, leverage, 
                          stop_loss=None, take_profit=None):
        """
        Colocar orden de margen en Kraken
        
        Args:
            pair: Par de trading
            side: 'buy' o 'sell'
            size: Tamaño de la orden
            leverage: Apalancamiento
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
            
        Returns:
            Dict con información de la orden o None si falla
        """
        try:
            logger.info(f"Colocando orden {side} para {pair}")
            logger.info(f"Tamaño: {size}, Leverage: {leverage}x")
            
            # Preparar parámetros de la orden
            order_params = {
                'pair': pair,
                'type': side,
                'ordertype': 'market',
                'volume': str(size),
                'leverage': str(leverage)
            }
            
            # Agregar stop loss si está especificado
            if stop_loss:
                order_params['close[ordertype]'] = 'stop-loss'
                order_params['close[price]'] = str(stop_loss)
            
            # Nota: Kraken no permite take profit y stop loss simultáneos
            # en una sola orden, necesitarías hacer órdenes separadas
            
            # Ejecutar orden (COMENTADO PARA EVITAR TRADES REALES EN TESTING)
            # DESCOMENTAR CUANDO ESTÉS LISTO PARA OPERAR EN REAL
            
            # response = self.api.query_private('AddOrder', order_params)
            # txid = response['result']['txid'][0]
            
            # Para testing, simular respuesta
            txid = f"TEST-{int(time.time())}"
            logger.info(f"Orden simulada colocada: {txid}")
            
            return {
                'txid': txid,
                'size': size,
                'leverage': leverage,
                'status': 'simulated'
            }
            
        except Exception as e:
            logger.error(f"Error colocando orden: {e}")
            return None
    
    def close_position(self, pair, position_type):
        """
        Cerrar posición abierta
        
        Args:
            pair: Par de trading
            position_type: 'long' o 'short'
            
        Returns:
            True si se cerró exitosamente
        """
        try:
            logger.info(f"Cerrando posición {position_type} para {pair}")
            
            # Obtener posiciones abiertas
            # positions = self.kraken.get_open_positions()
            
            # Cerrar posición correspondiente
            # side = 'sell' si position_type == 'long' else 'buy'
            
            # Para testing, simular cierre
            logger.info("Posición simulada cerrada")
            return True
            
        except Exception as e:
            logger.error(f"Error cerrando posición: {e}")
            return False
