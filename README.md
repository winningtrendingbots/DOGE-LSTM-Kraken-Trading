# 🤖 Bot de Trading Automatizado para Kraken

Bot de trading automatizado que ejecuta la estrategia de Volumen + OHLC en Kraken con margin trading, usando GitHub Actions para ejecución cada 15 minutos.

## 📋 Características

✅ **Trading en Margin de Kraken** con apalancamiento inteligente  
✅ **Ejecución automática** cada 15 minutos vía GitHub Actions  
✅ **Predicción LSTM** de volumen futuro (entrenamiento diario)  
✅ **Trailing Stop** dinámico  
✅ **Control de pérdidas** diarias máximas  
✅ **Notificaciones** a Telegram en tiempo real  
✅ **Gestión de riesgo** avanzada  
✅ **Persistencia de estado** entre ejecuciones  

---

## 🧠 Sistema LSTM Integrado

El bot incluye un sistema de predicción de volumen con LSTM que:

- 🔮 **Predice** el volumen de la siguiente vela
- 📊 **Calcula** derivadas predichas (aceleración de volumen)
- ✅ **Confirma** o filtra señales de trading tradicionales
- 🔄 **Se entrena** automáticamente cada madrugada (03:00 UTC)

**Para más información:** Ver [LSTM_README.md](LSTM_README.md)  

---

## 🚀 Configuración Inicial

### 1. Crear API Key en Kraken

1. Entra en tu cuenta de Kraken
2. Ve a **Settings** → **API**
3. Crea una nueva API Key con los siguientes permisos:
   - ✅ Query Funds
   - ✅ Query Open Orders & Trades
   - ✅ Query Closed Orders & Trades
   - ✅ Create & Modify Orders
   - ✅ Cancel/Close Orders
4. Guarda tu **API Key** y **Private Key** (no las compartas nunca)

### 2. Crear Bot de Telegram

1. Abre Telegram y busca [@BotFather](https://t.me/botfather)
2. Envía `/newbot` y sigue las instrucciones
3. Guarda el **Bot Token** que te da
4. Para obtener tu **Chat ID**:
   - Envía un mensaje a tu bot
   - Abre en navegador: `https://api.telegram.org/bot<TU_BOT_TOKEN>/getUpdates`
   - Busca tu `chat_id` en el JSON

### 3. Configurar GitHub Repository

1. **Fork o clona este repositorio**

2. **Configura los Secrets** en GitHub:
   - Ve a tu repositorio → **Settings** → **Secrets and variables** → **Actions**
   - Crea los siguientes secrets:

```
KRAKEN_API_KEY         → Tu API Key de Kraken
KRAKEN_API_SECRET      → Tu Private Key de Kraken
TELEGRAM_BOT_TOKEN     → Token de tu bot de Telegram
TELEGRAM_CHAT_ID       → Tu chat ID de Telegram
```

3. **Habilita GitHub Actions**:
   - Ve a **Actions** en tu repositorio
   - Si está deshabilitado, habilítalo

### 4. Estructura de Archivos

Asegúrate de tener esta estructura en tu repositorio:

```
tu-repo/
├── .github/
│   └── workflows/
│       └── trading.yml          # Workflow de GitHub Actions
├── kraken_trader.py             # API de Kraken
├── telegram_notifier.py         # Notificaciones Telegram
├── state_manager.py             # Gestión de estado
├── live_trading.py              # Script principal
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

---

## ⚙️ Configuración de la Estrategia

Edita los parámetros en `live_trading.py` clase `ProductionConfig`:

### Trading Básico
```python
SYMBOL = 'ETH-USD'              # Par a tradear
KRAKEN_PAIR = 'XETHZUSD'        # Formato Kraken
INTERVAL = 15                    # Minutos entre ejecuciones
```

### Gestión de Riesgo
```python
RISK_PER_TRADE = 0.05           # 5% de riesgo por trade
TP_POINTS = 100                  # Take profit en puntos
ATR_STOP_MULTIPLIER = 2.0       # Stop loss = 2 x ATR
```

### Trailing Stop
```python
USE_TRAILING_STOP = True        # Activar trailing stop
TRAILING_START = 25             # Activar al alcanzar +25 puntos
TRAILING_STEP = 15              # Seguir cada 15 puntos
```

### Límites
```python
PROFIT_CLOSE = 50               # Cerrar al alcanzar +50 puntos
MAX_DAILY_LOSS = -200           # Detener si pérdida > $200/día
MAX_POSITIONS = 15              # Máximo de posiciones simultáneas
MAX_BARS_IN_TRADE = 48          # Cerrar después de 48 barras (12h con TF 15m)
```

### Apalancamiento
```python
LEVERAGE_MIN = 2                # Mínimo para evitar comisiones mínimas
LEVERAGE_MAX = 5                # Máximo permitido
```

### Horario de Trading
```python
USE_TRADING_HOURS = True
TRADE_EUROPEAN_SESSION = True   # 07:00-16:00 GMT
TRADE_AMERICAN_SESSION = True   # 13:00-22:00 GMT
TRADE_ASIAN_SESSION = False     # 00:00-08:00 GMT
```

---

## 🎯 Cómo Funciona

### Flujo de Ejecución (Cada 15 minutos)

1. **GitHub Actions ejecuta** el workflow
2. **Descarga datos** de Kraken (últimas 200 velas)
3. **Calcula indicadores** técnicos y derivadas de volumen
4. **Actualiza posiciones** abiertas (trailing stops, SL/TP)
5. **Genera señales** de trading
6. **Ejecuta órdenes** si hay señales válidas
7. **Guarda estado** y envía notificaciones a Telegram

### Lógica de Trading

**Señal de Compra cuando:**
- Aceleración de volumen positiva durante 2+ barras consecutivas
- Confirmaciones opcionales: ADX, OBV, Price MA, RSI

**Señal de Venta cuando:**
- Aceleración de volumen negativa durante 2+ barras consecutivas
- Confirmaciones opcionales invertidas

**Gestión de Posiciones:**
- Apalancamiento calculado automáticamente según balance y riesgo
- Stop loss dinámico basado en ATR
- Trailing stop que sigue el precio favorable
- Cierre automático por: TP, SL, profit target, time limit o pérdida diaria

---

## 📱 Notificaciones de Telegram

Recibirás mensajes para:

- ✅ Inicio del bot
- 🟢 Señales detectadas (BUY/SELL)
- 📊 Órdenes ejecutadas
- 💰 Posiciones cerradas con P&L
- 📈 Actualizaciones de trailing stop
- ⚠️ Límite de pérdida diaria alcanzado
- ❌ Errores críticos
- 📊 Resumen diario (opcional)

---

## 🔍 Monitoreo y Logs

### Ver logs en GitHub Actions

1. Ve a tu repositorio → **Actions**
2. Selecciona el último workflow run
3. Abre **trade** → **Ejecutar bot de trading**
4. Verás los logs en tiempo real

### Descargar logs históricos

Los logs se guardan como artifacts en cada ejecución:
1. Ve al workflow run
2. Scroll hasta abajo → **Artifacts**
3. Descarga `trading-logs-XXXX`

### Estado Persistente

El bot guarda su estado en `trading_state.json` que incluye:
- Posiciones abiertas actuales
- Estadísticas del día
- Capital actual
- Configuración de trailing stops

---

## 🛡️ Seguridad

### ✅ Buenas Prácticas

1. **NUNCA** subas tus API keys al código
2. **SIEMPRE** usa GitHub Secrets
3. Activa **2FA** en Kraken y GitHub
4. Revisa los **permisos de API** regularmente
5. Limita el **balance en la cuenta** de trading
6. Monitorea las **notificaciones de Telegram**

### ⚠️ Advertencias

- Este bot opera con dinero real
- Las pérdidas son posibles y pueden ser significativas
- Prueba primero en una cuenta demo/pequeña
- No dejes el bot sin supervisión por períodos largos
- Revisa los logs regularmente

---

## 🔧 Solución de Problemas

### Error: "'>=' not supported between instances of 'float' and 'NoneType'"

Este error ocurre cuando los datos de Kraken contienen valores nulos. **Solución:**

1. **Ejecuta el diagnóstico de datos:**
   ```bash
   python debug_data.py
   ```
   Esto te mostrará exactamente qué datos están disponibles y si hay problemas.

2. **El código ya maneja este error automáticamente:**
   - Limpia valores nulos
   - Rellena gaps en los datos
   - Valida antes de calcular indicadores

3. **Si persiste, verifica:**
   - Que el par de trading existe en Kraken (`XETHZUSD` para ETH/USD)
   - Que el intervalo es válido (1, 5, 15, 30, 60, 240, 1440)
   - Que hay liquidez en el par seleccionado

### El bot no ejecuta órdenes

1. Verifica que tienes balance suficiente en Kraken
2. Comprueba que las API keys tienen los permisos correctos
3. Revisa los logs de GitHub Actions para errores
4. Verifica que no se alcanzó el límite de pérdida diaria

### No recibo notificaciones en Telegram

1. Verifica que el Bot Token es correcto
2. Comprueba que el Chat ID es el correcto
3. Asegúrate de haber enviado `/start` a tu bot
4. Revisa los logs para errores de API

### El workflow falla en GitHub Actions

1. Verifica que todos los secrets están configurados
2. Comprueba que el código está actualizado
3. Revisa los logs de error específicos
4. Verifica que GitHub Actions está habilitado

### Pérdidas inesperadas

1. Reduce `RISK_PER_TRADE`
2. Ajusta `MAX_DAILY_LOSS` más conservador
3. Activa más confirmaciones (ADX, RSI, etc.)
4. Reduce `MAX_POSITIONS`
5. Considera pausar el bot y revisar la estrategia

---

## 📊 Optimización y Backtesting

Antes de usar en producción:

1. **Backtest completo** con datos históricos
2. **Forward testing** en cuenta demo
3. **Optimización de parámetros** para tu mercado
4. **Análisis de drawdown** y gestión de riesgo

Usa el script original de backtest para probar configuraciones:

```bash
python backtest_strategy_v2.py
```

---

## 🚦 Control Manual

### Pausar el bot

1. Ve a **Actions** → **Kraken Trading Bot**
2. Click en **Disable workflow**

### Ejecutar manualmente

1. Ve a **Actions** → **Kraken Trading Bot**
2. Click en **Run workflow**

### Cerrar todas las posiciones

Edita `live_trading.py` y ejecuta manualmente:

```python
# Al final del archivo main(), antes de sys.exit()
trader.kraken.close_position('XETHZUSD', 'long')
trader.kraken.close_position('XETHZUSD', 'short')
```

---

## 📝 Notas Importantes

1. **GitHub Actions tiene límites**:
   - 2000 minutos/mes en plan gratuito
   - Con 96 ejecuciones/día = ~960 min/mes
   - Considera un plan de pago si es necesario

2. **Kraken API limits**:
   - Respetar rate limits (el código ya lo hace)
   - Verificar fees de margin trading

3. **Comisiones**:
   - Las comisiones impactan el rendimiento
   - El bot usa apalancamiento para reducir comisiones mínimas

---

## 📧 Soporte

Si tienes problemas:

1. Revisa esta documentación completa
2. Verifica los logs de GitHub Actions
3. Comprueba las notificaciones de Telegram
4. Revisa el código y los comentarios

---

## ⚖️ Disclaimer

Este bot es para fines educativos. El trading conlleva riesgos significativos y puedes perder tu capital. Siempre:

- Opera responsablemente
- Solo arriesga lo que puedas permitirte perder
- Haz tu propia investigación
- Considera consultar a un asesor financiero

**El autor no se hace responsable de pérdidas financieras.**

---

## 📜 Licencia

MIT License - Usa bajo tu propio riesgo

---

**¡Feliz Trading! 🚀📈**
