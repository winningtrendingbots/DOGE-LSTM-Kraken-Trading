"""
Módulo para enviar notificaciones a Telegram
Informa sobre señales, órdenes, cierres y errores
"""

import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Cliente para enviar notificaciones a Telegram"""
    
    def __init__(self, bot_token, chat_id):
        """
        Inicializar notificador de Telegram
        
        Args:
            bot_token: Token del bot de Telegram
            chat_id: ID del chat donde enviar mensajes
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        logger.info("TelegramNotifier inicializado")
    
    def send_message(self, message, parse_mode='HTML'):
        """
        Enviar mensaje a Telegram
        
        Args:
            message: Texto del mensaje
            parse_mode: Modo de parseo ('HTML' o 'Markdown')
            
        Returns:
            True si se envió exitosamente
        """
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Mensaje enviado a Telegram")
                return True
            else:
                logger.error(f"Error enviando mensaje: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error enviando mensaje a Telegram: {e}")
            return False
    
    def notify_signal(self, signal_type, price, indicators):
        """
        Notificar señal de trading detectada
        
        Args:
            signal_type: 'BUY' o 'SELL'
            price: Precio actual
            indicators: Dict con indicadores técnicos
        """
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        
        message = f"""
{emoji} <b>SEÑAL DETECTADA: {signal_type}</b>

💰 <b>Precio:</b> ${price:.4f}
📊 <b>Aceleración:</b> {indicators.get('accel', 0):.2f}
📈 <b>ADX:</b> {indicators.get('adx', 0):.2f}
📉 <b>RSI:</b> {indicators.get('rsi', 0):.2f}
"""
        
        # Agregar info LSTM si está disponible
        if 'lstm_signal' in indicators:
            lstm_emoji = "🟢" if indicators['lstm_signal'] > 0 else ("🔴" if indicators['lstm_signal'] < 0 else "⚪")
            message += f"\n🧠 <b>LSTM Signal:</b> {lstm_emoji}"
        
        if 'lstm_vol_pred' in indicators:
            message += f"\n📊 <b>Vol Predicho:</b> {indicators['lstm_vol_pred']:.0f}"
        
        message += f"\n\n⏳ <b>Evaluando entrada...</b>"
        
        self.send_message(message.strip())
    
    def notify_order_placed(self, order_details):
        """
        Notificar que se colocó una orden
        
        Args:
            order_details: Dict con detalles de la orden
        """
        side_emoji = "🟢" if order_details['side'] == 'buy' else "🔴"
        
        message = f"""
{side_emoji} <b>ORDEN EJECUTADA</b>

🆔 <b>ID:</b> <code>{order_details['txid']}</code>
📊 <b>Tipo:</b> {order_details['side'].upper()}
💰 <b>Precio:</b> ${order_details['price']:.4f}
📦 <b>Tamaño:</b> {order_details['size']:.4f}
💵 <b>Costo:</b> ${order_details['cost']:.2f}
📈 <b>Apalancamiento:</b> {order_details['leverage']}x
💳 <b>Margen:</b> ${order_details['margin']:.2f}

🎯 <b>Take Profit:</b> ${order_details['tp']:.4f}
🛡️ <b>Stop Loss:</b> ${order_details['sl']:.4f}

⏰ <b>Tiempo:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.send_message(message.strip())
    
    def notify_order_closed(self, close_details):
        """
        Notificar que se cerró una posición
        
        Args:
            close_details: Dict con detalles del cierre
        """
        pnl = close_details['pnl']
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
{emoji} <b>POSICIÓN CERRADA</b>

🆔 <b>ID:</b> <code>{close_details['txid']}</code>
📊 <b>Dirección:</b> {close_details['direction'].upper()}
💰 <b>Entrada:</b> ${close_details['entry_price']:.4f}
💰 <b>Salida:</b> ${close_details['exit_price']:.4f}

{'💚' if pnl > 0 else '❤️'} <b>P&L:</b> ${pnl:.2f} ({close_details['return_pct']:.2f}%)
📝 <b>Razón:</b> {close_details['reason']}
⏱️ <b>Duración:</b> {close_details['duration']}

💰 <b>Balance:</b> ${close_details['balance']:.2f}
"""
        
        self.send_message(message.strip())
    
    def notify_trailing_stop_update(self, position_id, new_sl, current_pnl):
        """
        Notificar actualización de trailing stop
        
        Args:
            position_id: ID de la posición
            new_sl: Nuevo stop loss
            current_pnl: P&L actual
        """
        message = f"""
📈 <b>TRAILING STOP ACTUALIZADO</b>

🆔 <b>Posición:</b> <code>{position_id}</code>
🛡️ <b>Nuevo SL:</b> ${new_sl:.4f}
💰 <b>P&L Actual:</b> ${current_pnl:.2f}
"""
        
        self.send_message(message.strip())
    
    def notify_daily_loss_limit(self, daily_pnl, max_loss):
        """
        Notificar que se alcanzó el límite de pérdida diaria
        
        Args:
            daily_pnl: P&L del día
            max_loss: Pérdida máxima permitida
        """
        message = f"""
⛔ <b>LÍMITE DE PÉRDIDA ALCANZADO</b>

❌ <b>Pérdida del día:</b> ${daily_pnl:.2f}
🛑 <b>Límite:</b> ${max_loss:.2f}

⚠️ Trading deshabilitado hasta mañana
"""
        
        self.send_message(message.strip())
    
    def notify_error(self, error_message):
        """
        Notificar error del sistema
        
        Args:
            error_message: Mensaje de error
        """
        message = f"""
❌ <b>ERROR DEL SISTEMA</b>

⚠️ {error_message}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.send_message(message.strip())
    
    def notify_startup(self, config_info):
        """
        Notificar inicio del bot
        
        Args:
            config_info: Información de configuración
        """
        message = f"""
🤖 <b>BOT INICIADO</b>

📊 <b>Par:</b> {config_info.get('pair', 'N/A')}
⏱️ <b>Intervalo:</b> {config_info.get('interval', 'N/A')} min
💰 <b>Riesgo:</b> {config_info.get('risk', 0) * 100:.1f}%
📈 <b>Apalancamiento:</b> {config_info.get('leverage_min', 0)}-{config_info.get('leverage_max', 0)}x

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.send_message(message.strip())
