"""
Módulo para gestionar el estado del bot
Maneja posiciones abiertas, P&L diario, y persistencia de datos
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class StateManager:
    """Gestor de estado del bot de trading"""
    
    def __init__(self, state_file='trading_state.json'):
        """
        Inicializar gestor de estado
        
        Args:
            state_file: Archivo donde guardar el estado
        """
        self.state_file = state_file
        self.state = self._load_state()
        
        logger.info("StateManager inicializado")
    
    def _load_state(self):
        """Cargar estado desde archivo"""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    logger.info(f"Estado cargado desde {self.state_file}")
                    return state
            else:
                logger.info("No existe estado previo, iniciando nuevo")
                return self._create_new_state()
                
        except Exception as e:
            logger.error(f"Error cargando estado: {e}")
            return self._create_new_state()
    
    def _create_new_state(self):
        """Crear nuevo estado vacío"""
        return {
            'positions': {},
            'daily_trades': [],
            'last_reset_date': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': 0,
            'total_profit': 0,
            'bars_since_trade': 0
        }
    
    def _save_state(self):
        """Guardar estado en archivo"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.info(f"Estado guardado en {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def _check_daily_reset(self):
        """Verificar si necesitamos resetear el P&L diario"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.state['last_reset_date'] != today:
            logger.info(f"Nuevo día detectado, reseteando P&L diario")
            self.state['daily_trades'] = []
            self.state['last_reset_date'] = today
            self._save_state()
    
    def add_position(self, position_id, position_data):
        """
        Agregar nueva posición abierta
        
        Args:
            position_id: ID único de la posición
            position_data: Dict con datos de la posición
        """
        position_data['entry_time'] = datetime.now().isoformat()
        position_data['bars_open'] = 0
        position_data['highest_price'] = position_data['entry_price']
        position_data['lowest_price'] = position_data['entry_price']
        
        self.state['positions'][position_id] = position_data
        self.state['bars_since_trade'] = 0
        self._save_state()
        
        logger.info(f"Posición agregada: {position_id}")
    
    def remove_position(self, position_id):
        """
        Remover posición cerrada
        
        Args:
            position_id: ID de la posición a remover
        """
        if position_id in self.state['positions']:
            del self.state['positions'][position_id]
            self._save_state()
            logger.info(f"Posición removida: {position_id}")
    
    def get_position(self, position_id):
        """
        Obtener datos de una posición
        
        Args:
            position_id: ID de la posición
            
        Returns:
            Dict con datos de la posición o None
        """
        return self.state['positions'].get(position_id)
    
    def get_all_positions(self):
        """
        Obtener todas las posiciones abiertas
        
        Returns:
            Dict con todas las posiciones
        """
        return self.state['positions']
    
    def update_position(self, position_id, updated_data):
        """
        Actualizar datos de una posición
        
        Args:
            position_id: ID de la posición
            updated_data: Dict con datos actualizados
        """
        if position_id in self.state['positions']:
            self.state['positions'][position_id].update(updated_data)
            self._save_state()
    
    def update_position_extremes(self, position_id, current_price):
        """
        Actualizar precios máximos/mínimos de una posición
        
        Args:
            position_id: ID de la posición
            current_price: Precio actual
        """
        if position_id in self.state['positions']:
            position = self.state['positions'][position_id]
            
            position['highest_price'] = max(
                position.get('highest_price', current_price),
                current_price
            )
            
            position['lowest_price'] = min(
                position.get('lowest_price', current_price),
                current_price
            )
            
            self._save_state()
    
    def increment_bars_open(self):
        """Incrementar contador de barras para posiciones abiertas"""
        for position_id in self.state['positions']:
            self.state['positions'][position_id]['bars_open'] += 1
        
        self.state['bars_since_trade'] += 1
        self._save_state()
    
    def add_trade(self, pnl):
        """
        Agregar trade cerrado al historial diario
        
        Args:
            pnl: Ganancia/pérdida del trade
        """
        self._check_daily_reset()
        
        trade_data = {
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        
        self.state['daily_trades'].append(trade_data)
        self.state['total_trades'] += 1
        self.state['total_profit'] += pnl
        
        self._save_state()
        
        logger.info(f"Trade agregado: P&L=${pnl:.2f}")
    
    def get_daily_profit(self):
        """
        Calcular ganancia/pérdida del día
        
        Returns:
            P&L total del día
        """
        self._check_daily_reset()
        
        daily_pnl = sum(trade['pnl'] for trade in self.state['daily_trades'])
        return daily_pnl
    
    def is_daily_loss_limit_hit(self, max_loss):
        """
        Verificar si se alcanzó el límite de pérdida diaria
        
        Args:
            max_loss: Pérdida máxima permitida (número negativo)
            
        Returns:
            True si se alcanzó el límite
        """
        daily_pnl = self.get_daily_profit()
        
        if daily_pnl <= max_loss:
            logger.warning(f"Límite de pérdida alcanzado: ${daily_pnl:.2f}")
            return True
        
        return False
    
    def can_trade(self):
        """
        Verificar si el bot puede tradear
        
        Returns:
            True si puede tradear
        """
        # Por ahora solo verifica si existe el estado
        # Puedes agregar más verificaciones aquí
        return True
    
    def get_statistics(self):
        """
        Obtener estadísticas del bot
        
        Returns:
            Dict con estadísticas
        """
        self._check_daily_reset()
        
        daily_pnl = self.get_daily_profit()
        
        stats = {
            'total_trades': self.state['total_trades'],
            'total_profit': self.state['total_profit'],
            'daily_trades': len(self.state['daily_trades']),
            'daily_profit': daily_pnl,
            'open_positions': len(self.state['positions']),
            'bars_since_trade': self.state['bars_since_trade']
        }
        
        return stats
    
    def reset_daily(self):
        """Forzar reset del P&L diario"""
        self.state['daily_trades'] = []
        self.state['last_reset_date'] = datetime.now().strftime('%Y-%m-%d')
        self._save_state()
        logger.info("P&L diario reseteado manualmente")
    
    def clear_all_positions(self):
        """Limpiar todas las posiciones (usar con cuidado)"""
        self.state['positions'] = {}
        self._save_state()
        logger.warning("Todas las posiciones limpiadas")
