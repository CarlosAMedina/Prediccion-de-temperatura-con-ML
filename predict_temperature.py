"""
Predicción de Temperatura Actual - Lindavista, CDMX
Sistema de predicción de temperatura usando Machine Learning
con datos históricos de 10 años y predicción en tiempo real.
"""

import os
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configuración de warnings
warnings.filterwarnings('ignore')

# Constantes de configuración
LATITUDE = 19.4868
LONGITUDE = -99.1310
TIMEZONE = "America/Mexico_City"
TARGET_MAE = 0.2
OPENWEATHER_API_KEY = "" # CONFIGURA AQUÍ TU API KEY

# Directorios
BASE_DIR = os.path.join(os.path.dirname(__file__), "temperatura_lindavistav3")
os.makedirs(BASE_DIR, exist_ok=True)


def download_year(year):
    """
    Descargar datos históricos de un año desde Open-Meteo API.
    
    Args:
        year (int): Año a descargar
        
    Returns:
        pd.DataFrame: DataFrame con datos horarios del año
    """
    # Para 2025, solo hasta septiembre
    end_date = f"{year}-09-30" if year == 2025 else f"{year}-12-31"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": f"{year}-01-01",
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover,apparent_temperature,dew_point_2m,precipitation,shortwave_radiation",
        "timezone": TIMEZONE
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:
            if attempt == 2:
                raise Exception(f"Error descargando datos de {year}: {e}")
            time.sleep(2)


def download_historical_data(years=10):
    """
    Descargar y consolidar datos históricos de múltiples años.
    
    Args:
        years (int): Número de años hacia atrás a descargar
        
    Returns:
        pd.DataFrame: DataFrame consolidado con todos los datos
    """
    print("Descargando datos históricos...")
    all_data = []
    current_year = datetime.now().year
    start_year = current_year - years + 1

    for year in range(start_year, current_year + 1):
        print(f"  Descargando {year}...")
        df_year = download_year(year)
        all_data.append(df_year)
        time.sleep(1)

    # Combinar todos los años
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Guardar
    csv_path = os.path.join(BASE_DIR, "datos_10years.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Descargados {len(df):,} registros")
    print(f"   Período: {df['time'].min()} a {df['time'].max()}")
    
    return df


def create_features(df):
    """
    Crear características de ingeniería para el modelo.
    
    Args:
        df (pd.DataFrame): DataFrame con datos brutos
        
    Returns:
        pd.DataFrame: DataFrame con características adicionales
    """
    print("\nCreando características...")
    
    # Características temporales
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear

    # Codificación cíclica
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Características de retardo (lag)
    for lag in [1, 3, 6, 12, 24]:
        df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)

    # Promedios móviles
    for window in [6, 12, 24]:
        df[f'temp_ma_{window}h'] = df['temperature_2m'].rolling(window=window, min_periods=1).mean()

    # Interacciones simples
    if 'apparent_temperature' in df.columns:
        df['temp_apparent_diff'] = df['temperature_2m'] - df['apparent_temperature']

    if 'dew_point_2m' in df.columns:
        df['vpd'] = df['temperature_2m'] - df['dew_point_2m']

    # Categorías de clima
    df['is_day'] = df['hour'].between(6, 18).astype(int)
    df['is_rainy'] = (df['precipitation'] > 0).astype(int) if 'precipitation' in df.columns else 0

    # Temperatura histórica promedio por hora-mes (CRÍTICO)
    df['temp_hist'] = df.groupby(['hour', 'month'])['temperature_2m'].transform('mean')
    df['temp_deviation'] = df['temperature_2m'] - df['temp_hist']

    # Eliminar NaN
    df = df.dropna()

    print(f"Características creadas")
    print(f"   Shape final: {df.shape}")
    
    return df


def prepare_training_data(df):
    """
    Preparar datos para entrenamiento del modelo.
    
    Args:
        df (pd.DataFrame): DataFrame con características
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    print("\nPreparando datos para entrenamiento...")
    
    # Definir características
    feature_cols = [
        'hour', 'month', 'day_of_year',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'relative_humidity_2m', 'pressure_msl', 'wind_speed_10m', 'cloud_cover',
        'temp_lag_1h', 'temp_lag_3h', 'temp_lag_6h', 'temp_lag_12h', 'temp_lag_24h',
        'temp_ma_6h', 'temp_ma_12h', 'temp_ma_24h',
        'temp_hist', 'temp_deviation',
        'is_day'
    ]

    # Agregar características opcionales si existen
    optional_features = ['apparent_temperature', 'dew_point_2m', 'precipitation',
                        'shortwave_radiation', 'temp_apparent_diff', 'vpd', 'is_rainy']

    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)

    # Preparar X e y
    X = df[feature_cols].values
    y = df['temperature_2m'].values

    # División temporal (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Datos preparados")
    print(f"   Entrenamiento: {len(X_train):,} muestras")
    print(f"   Prueba: {len(X_test):,} muestras")
    print(f"   Características: {len(feature_cols)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_model(X_train, X_test, y_train, y_test):
    """
    Entrenar modelo de predicción de temperatura.
    
    Args:
        X_train, X_test: Datos de entrenamiento y prueba
        y_train, y_test: Etiquetas de entrenamiento y prueba
        
    Returns:
        tuple: model, train_mae, test_mae, within_target_pct
    """
    print("\nEntrenando modelo...")
    start_time = time.time()

    # Modelo principal - HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=42
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Evaluar
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Calcular porcentaje dentro del objetivo
    test_errors = np.abs(y_test - y_pred_test)
    within_target = (test_errors <= TARGET_MAE).mean() * 100

    train_time = time.time() - start_time

    print(f"\nEntrenamiento completado en {train_time:.1f} segundos")
    print(f"   MAE Entrenamiento: {train_mae:.4f}°C")
    print(f"   MAE Prueba: {test_mae:.4f}°C")
    print(f"   Predicciones dentro de ±{TARGET_MAE}°C: {within_target:.1f}%")

    if test_mae <= TARGET_MAE:
        print(f"\n¡OBJETIVO ALCANZADO! Precisión ≤ ±{TARGET_MAE}°C")
    else:
        print(f"\nMAE actual: {test_mae:.4f}°C (objetivo: {TARGET_MAE}°C)")
    
    return model, train_mae, test_mae, within_target


def save_model(model, scaler, feature_cols, test_mae, within_target):
    """
    Guardar modelo entrenado y datos auxiliares.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        feature_cols: Lista de nombres de características
        test_mae: Error medio absoluto en test
        within_target: Porcentaje dentro del objetivo
    """
    print("\nGuardando modelo...")
    
    # Crear lookup table de temp_hist
    csv_path = os.path.join(BASE_DIR, "datos_10years.csv")
    df_for_lookup = pd.read_csv(csv_path, parse_dates=['time'])
    df_for_lookup['hour'] = df_for_lookup['time'].dt.hour
    df_for_lookup['month'] = df_for_lookup['time'].dt.month

    # Crear diccionario de temperaturas históricas promedio
    temp_hist_lookup = df_for_lookup.groupby(['hour', 'month'])['temperature_2m'].mean().to_dict()

    # Preparar datos del modelo
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'mae': test_mae,
        'within_target_pct': within_target,
        'training_date': datetime.now().isoformat(),
        'temp_hist_lookup': temp_hist_lookup
    }

    # Guardar
    model_path = os.path.join(BASE_DIR, "modelo_temperatura.pkl")
    joblib.dump(model_data, model_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"   Tamaño del archivo: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")


def get_openweather_data():
    """
    Obtener datos actuales de OpenWeatherMap API.
    
    Returns:
        dict: Diccionario con datos meteorológicos o None si hay error
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": LATITUDE,
            "lon": LONGITUDE,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'] * 3.6,  # m/s a km/h
            'clouds': data['clouds']['all'],
            'feels_like': data['main']['feels_like']
        }
    except Exception as e:
        print(f"Error OpenWeatherMap: {e}")
        return None


def predict_current():
    """
    Hacer predicción de temperatura actual usando el modelo entrenado.
    
    Returns:
        tuple: prediction, real_temp_avg, open_meteo_temp, owm_data
    """
    print("\nPREDICCIÓN DE TEMPERATURA ACTUAL")
    print("=" * 60)
    
    # Cargar modelo
    model_path = os.path.join(BASE_DIR, "modelo_temperatura.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo no encontrado. Ejecute primero el entrenamiento con --train")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    temp_hist_lookup = model_data['temp_hist_lookup']

    # Obtener datos de ambas APIs
    print("Consultando APIs...")

    # API 1: Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover,apparent_temperature,precipitation",
        "hourly": "temperature_2m",
        "past_days": 2,
        "timezone": TIMEZONE
    }

    response = requests.get(url, params=params)
    data = response.json()
    current = data['current']
    current_time = pd.to_datetime(current['time'])

    # API 2: OpenWeatherMap
    owm_data = get_openweather_data()

    # Datos históricos para lags
    hourly = pd.DataFrame(data['hourly'])
    hourly['time'] = pd.to_datetime(hourly['time'])

    # Obtener temp_hist del lookup table
    lookup_key = (current_time.hour, current_time.month)
    temp_hist_value = temp_hist_lookup.get(lookup_key, current['temperature_2m'])

    # Crear DataFrame para predicción
    pred_data = pd.DataFrame([{
        'hour': current_time.hour,
        'month': current_time.month,
        'day_of_year': current_time.dayofyear,
        'hour_sin': np.sin(2 * np.pi * current_time.hour / 24),
        'hour_cos': np.cos(2 * np.pi * current_time.hour / 24),
        'month_sin': np.sin(2 * np.pi * current_time.month / 12),
        'month_cos': np.cos(2 * np.pi * current_time.month / 12),
        'relative_humidity_2m': current.get('relative_humidity_2m', 50),
        'pressure_msl': current.get('pressure_msl', 1013),
        'wind_speed_10m': current.get('wind_speed_10m', 5),
        'cloud_cover': current.get('cloud_cover', 50),
        'is_day': 1 if 6 <= current_time.hour <= 18 else 0,
        'apparent_temperature': current.get('apparent_temperature', 20),
        'precipitation': current.get('precipitation', 0)
    }])

    # Calcular lags desde datos históricos
    temps = hourly['temperature_2m'].values
    for lag in [1, 3, 6, 12, 24]:
        if len(temps) > lag:
            pred_data[f'temp_lag_{lag}h'] = temps[-lag]
        else:
            pred_data[f'temp_lag_{lag}h'] = temps[-1] if len(temps) > 0 else 20

    # Calcular promedios móviles
    for window in [6, 12, 24]:
        if len(temps) >= window:
            pred_data[f'temp_ma_{window}h'] = temps[-window:].mean()
        else:
            pred_data[f'temp_ma_{window}h'] = temps.mean() if len(temps) > 0 else 20

    # Usar temp_hist real del lookup
    pred_data['temp_hist'] = temp_hist_value
    pred_data['temp_deviation'] = current['temperature_2m'] - temp_hist_value

    # Otras características calculadas
    pred_data['temp_apparent_diff'] = current['temperature_2m'] - current.get('apparent_temperature', current['temperature_2m'])
    pred_data['vpd'] = 5
    pred_data['is_rainy'] = 1 if pred_data['precipitation'].iloc[0] > 0 else 0
    pred_data['dew_point_2m'] = pred_data['apparent_temperature'] - 2
    pred_data['shortwave_radiation'] = 0 if pred_data['is_day'].iloc[0] == 0 else 100

    # Seleccionar solo las características usadas en el modelo
    X_pred = pred_data[feature_cols].values

    # Normalizar
    X_pred_scaled = scaler.transform(X_pred)

    # Predecir
    prediction = model.predict(X_pred_scaled)[0]

    # Calcular temperatura real promediando ambas APIs
    if owm_data:
        real_temp_avg = (current['temperature_2m'] + owm_data['temperature']) / 2
    else:
        real_temp_avg = current['temperature_2m']

    # Mostrar resultados
    print("\n📍 LINDAVISTA, CDMX")
    print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n RESULTADOS:")
    print(f"   Predicción del modelo: {prediction:.2f}°C")
    print(f"   Open-Meteo: {current['temperature_2m']:.2f}°C")

    if owm_data:
        print(f"   OpenWeatherMap: {owm_data['temperature']:.2f}°C")
        print(f"   Promedio APIs: {real_temp_avg:.2f}°C")
        print(f"   Diferencia vs Promedio: {abs(prediction - real_temp_avg):.2f}°C")
    else:
        print(f"   Diferencia: {abs(prediction - current['temperature_2m']):.2f}°C")

    error = abs(prediction - real_temp_avg)
    if error <= TARGET_MAE:
        print(f"\n Dentro del objetivo (±{TARGET_MAE}°C)")
    else:
        print(f"\n Fuera del objetivo (±{TARGET_MAE}°C)")

    print(f"\n RENDIMIENTO DEL MODELO:")
    print(f"   MAE histórico: {model_data['mae']:.4f}°C")
    print(f"   Precisión histórica: {model_data['within_target_pct']:.1f}%")
    print(f"   Fecha entrenamiento: {model_data['training_date'][:10]}")
    print("=" * 60)

    return prediction, real_temp_avg, current['temperature_2m'], owm_data


def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Sistema de Predicción de Temperatura - Lindavista, CDMX'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Entrenar el modelo con datos históricos'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Hacer predicción de temperatura actual'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Número de años de datos históricos para entrenamiento (default: 10)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" SISTEMA DE PREDICCIÓN DE TEMPERATURA")
    print(" Lindavista, CDMX")
    print("=" * 60)

    if args.train:
        # Flujo de entrenamiento
        df = download_historical_data(years=args.years)
        df = create_features(df)
        X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_training_data(df)
        model, train_mae, test_mae, within_target = train_model(X_train, X_test, y_train, y_test)
        save_model(model, scaler, feature_cols, test_mae, within_target)
        print("\nEntrenamiento completado exitosamente")
        
    elif args.predict:
        # Flujo de predicción
        predict_current()
        
    else:
        # Sin argumentos, mostrar ayuda
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  python predict_temperature.py --train          # Entrenar modelo")
        print("  python predict_temperature.py --predict        # Hacer predicción")
        print("  python predict_temperature.py --train --years 5  # Entrenar con 5 años de datos")


if __name__ == "__main__":
    main()
