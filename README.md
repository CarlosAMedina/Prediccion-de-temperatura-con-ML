# Predictor de Temperatura - Lindavista, CDMX

Sistema de predicción de temperatura actual usando Machine Learning con datos históricos de 10 años. Utiliza HistGradientBoostingRegressor de scikit-learn para alcanzar una precisión de ±0.2°C.

## Características

- Descarga automática de datos históricos (Open-Meteo API)
- Ingeniería de características avanzada (lags, promedios móviles, codificación cíclica)
- Modelo de Machine Learning (HistGradientBoostingRegressor)
- Predicción en tiempo real con validación cruzada de APIs
- Objetivo: MAE ≤ 0.2°C

## Instalación

### Prerequisitos
- Python 3.8 o superior

### Pasos

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd "Práctica 7 - Predicción de temperatura actual"
```

2. **Obtener API Key de OpenWeatherMap**
   - Registrarse en [OpenWeatherMap](https://openweathermap.org/api)
   - Ir a la sección "API keys" en tu cuenta
   - Copiar tu API key gratuita
   - Abrir el archivo `predict_temperature.py` y pegar tu API key en la línea:
   ```python
   OPENWEATHER_API_KEY = "tu_api_key_aqui"
   ```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

Para entrenar el modelo con 10 años de datos históricos:

```bash
python predict_temperature.py --train
```

Para entrenar con un número específico de años:

```bash
python predict_temperature.py --train --years 5
```

### Hacer predicción

Una vez entrenado el modelo, para predecir la temperatura actual:

```bash
python predict_temperature.py --predict
```

## Estructura del Proyecto

```
.
├── predict_temperature.py      # Script principal
├── requirements.txt            # Dependencias del proyecto
├── temperatura_lindavistav3/   # Directorio de datos y modelo
│   ├── datos_10years.csv      # Datos históricos descargados
│   └── modelo_temperatura.pkl # Modelo entrenado guardado
└── README.md                   # Este archivo
```

## Configuración

Las siguientes constantes pueden ser modificadas en `predict_temperature.py`:

```python
LATITUDE = 19.4868              # Latitud de Lindavista, CDMX
LONGITUDE = -99.1310            # Longitud de Lindavista, CDMX
TIMEZONE = "America/Mexico_City"
TARGET_MAE = 0.2                # Objetivo de error absoluto medio
OPENWEATHER_API_KEY = "..."    # API key de OpenWeatherMap
```

## Características del Modelo

El modelo utiliza las siguientes características:

### Temporales
- Hora del día (codificación cíclica)
- Mes del año (codificación cíclica)
- Día del año

### Meteorológicas
- Humedad relativa
- Presión atmosférica
- Velocidad del viento
- Nubosidad
- Temperatura aparente
- Punto de rocío
- Precipitación
- Radiación de onda corta

### Derivadas
- Lags de temperatura (1h, 3h, 6h, 12h, 24h)
- Promedios móviles (6h, 12h, 24h)
- Temperatura histórica promedio por hora-mes
- Desviación de temperatura histórica
- VPD (Déficit de presión de vapor)
- Indicadores (día/noche, lluvia)

## Rendimiento

- **MAE objetivo**: ≤ 0.2°C
- **Datos de entrenamiento**: ~87,600 registros (10 años de datos horarios)
- **División**: 80% entrenamiento, 20% prueba
- **Validación**: Dual API (Open-Meteo + OpenWeatherMap)

## APIs Utilizadas

1. **Open-Meteo Archive API**: Datos históricos
   - URL: https://archive-api.open-meteo.com/v1/archive
   - Gratis, sin API key requerida

2. **Open-Meteo Forecast API**: Datos actuales y pronóstico
   - URL: https://api.open-meteo.com/v1/forecast
   - Gratis, sin API key requerida

3. **OpenWeatherMap API**: Validación de temperatura actual
   - URL: https://api.openweathermap.org/data/2.5/weather
   - Requiere API key (gratis para uso básico)

## Tecnologías

- **Python 3.8+**
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos
- **scikit-learn**: Machine Learning (HistGradientBoostingRegressor, StandardScaler)
- **joblib**: Persistencia del modelo
- **requests**: Consultas HTTP a APIs

## Ejemplo de Salida

```
PREDICCIÓN DE TEMPERATURA ACTUAL
============================================================
Consultando APIs...

LINDAVISTA, CDMX
2025-11-28 14:30:00

 RESULTADOS:
   Predicción del modelo: 23.45°C
   Open-Meteo: 23.50°C
   OpenWeatherMap: 23.40°C
   Promedio APIs: 23.45°C
   Diferencia vs Promedio: 0.00°C

Dentro del objetivo (±0.2°C)

RENDIMIENTO DEL MODELO:
   MAE histórico: 0.1856°C
   Precisión histórica: 89.3%
   Fecha entrenamiento: 2025-11-28
============================================================
```

## Autor

Carlos Armando Medina

Proyecto desarrollado como práctica de Machine Learning para predicción meteorológica.
