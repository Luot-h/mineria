import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from datetime import datetime

# Directorio para guardar modelos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Carga los datos de los archivos parquet en data_sampled."""
    print("Cargando datos...")
    data_files = [f for f in os.listdir('data_sampled') if f.endswith('_reduced.parquet')]
    
    if not data_files:
        raise Exception("No se encontraron archivos de datos")
    
    # Para pruebas rápidas, solo usar los primeros 2 archivos
    data_files = data_files[:2]
    print(f"Usando {len(data_files)} archivos: {data_files}")
    
    # Cargar y concatenar archivos
    dfs = []
    for file in data_files:
        file_path = os.path.join('data_sampled', file)
        print(f"Cargando {file_path}...")
        df = pd.read_parquet(file_path)
        print(f"  Cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        dfs.append(df)
    
    # Concatenar todos los DataFrames
    df = pd.concat(dfs, ignore_index=True)
    print(f"Datos combinados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df

def train_fare_model(df):
    """Entrena un modelo para predecir el costo del viaje (driver_pay)."""
    print("\n==== Entrenando modelo de predicción de costo (DRIVER_PAY) ====")
    
    if 'driver_pay' not in df.columns:
        print("Columna 'driver_pay' no encontrada.")
        return None
    
    # Seleccionar características
    numeric_cols = ['trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 
                    'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee']
    categorical_cols = ['pickup_hour', 'pickup_weekday', 'pickup_month']
    
    # Verificar que las columnas existen
    for col in numeric_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            numeric_cols.remove(col)
    
    for col in categorical_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            categorical_cols.remove(col)
    
    feature_cols = numeric_cols + categorical_cols
    
    if len(feature_cols) == 0:
        print("No hay suficientes columnas para entrenar el modelo.")
        return None
    
    # Eliminar filas con valores nulos
    df_clean = df.dropna(subset=feature_cols + ['driver_pay'])
    
    # One-hot encoding para variables categóricas
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    # Preparar datos
    X_cols = [col for col in df_encoded.columns if col in numeric_cols or 
              any(col.startswith(cat_col) for cat_col in categorical_cols)]
    X = df_encoded[X_cols]
    y = df_encoded['driver_pay']
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # Crear pipeline con escalado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Modelo entrenado - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Guardar modelo
    model_path = os.path.join(MODEL_DIR, 'fare_predictor.joblib')
    joblib.dump(pipeline, model_path)
    
    # Guardar también los nombres de las características
    feature_path = os.path.join(MODEL_DIR, 'fare_features.joblib')
    joblib.dump(X_cols, feature_path)
    
    print(f"Modelo guardado en {model_path}")
    return pipeline

def train_airport_model(df):
    """Entrena un modelo para clasificar si un viaje es hacia o desde un aeropuerto."""
    print("\n==== Entrenando modelo de clasificación de aeropuertos ====")
    
    # Crear una característica sintética para si el viaje es al aeropuerto
    # Basado en las zonas conocidas de los aeropuertos
    if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
        # JFK (132), LaGuardia (138), Newark (1)
        airport_zones = [1, 132, 138]
        
        df['to_airport'] = df['DOLocationID'].isin(airport_zones)
        df['from_airport'] = df['PULocationID'].isin(airport_zones)
        df['airport_trip'] = df['to_airport'] | df['from_airport']
        
        target_col = 'airport_trip'
        
        print(f"Creada la característica 'airport_trip'. Total de viajes de aeropuerto: {df[target_col].sum()}")
    else:
        print("No se pudieron encontrar columnas para identificar viajes de aeropuerto.")
        return None
    
    # Seleccionar características
    numeric_cols = ['trip_miles', 'trip_time', 'base_passenger_fare']
    categorical_cols = ['pickup_hour', 'pickup_weekday', 'pickup_month']
    
    # Verificar que las columnas existen
    for col in numeric_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            numeric_cols.remove(col)
    
    for col in categorical_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            categorical_cols.remove(col)
    
    feature_cols = numeric_cols + categorical_cols
    
    if len(feature_cols) == 0:
        print("No hay suficientes columnas para entrenar el modelo.")
        return None
    
    # Eliminar filas con valores nulos
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    # One-hot encoding para variables categóricas
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    # Preparar datos
    X_cols = [col for col in df_encoded.columns if col in numeric_cols or 
              any(col.startswith(cat_col) for cat_col in categorical_cols)]
    X = df_encoded[X_cols]
    y = df_encoded[target_col]
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # Crear pipeline con escalado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Modelo entrenado - Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo
    model_path = os.path.join(MODEL_DIR, 'airport_classifier.joblib')
    joblib.dump(pipeline, model_path)
    
    # Guardar también los nombres de las características
    feature_path = os.path.join(MODEL_DIR, 'airport_features.joblib')
    joblib.dump(X_cols, feature_path)
    
    print(f"Modelo guardado en {model_path}")
    return pipeline

def train_trip_time_model(df):
    """Entrena un modelo para predecir la duración del viaje (trip_time)."""
    print("\n==== Entrenando modelo de predicción de duración (TRIP_TIME) ====")
    
    if 'trip_time' not in df.columns:
        print("Columna 'trip_time' no encontrada.")
        return None
    
    # Seleccionar características
    numeric_cols = ['trip_miles', 'base_passenger_fare']
    categorical_cols = ['pickup_hour', 'pickup_weekday', 'pickup_month']
    
    if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
        categorical_cols.extend(['PULocationID', 'DOLocationID'])
    
    # Verificar que las columnas existen
    for col in numeric_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            numeric_cols.remove(col)
    
    for col in categorical_cols.copy():
        if col not in df.columns:
            print(f"Columna {col} no encontrada. Se omitirá.")
            categorical_cols.remove(col)
    
    feature_cols = numeric_cols + categorical_cols
    
    if len(feature_cols) == 0:
        print("No hay suficientes columnas para entrenar el modelo.")
        return None
    
    # Eliminar filas con valores nulos
    df_clean = df.dropna(subset=feature_cols + ['trip_time'])
    
    # Limitar el número de categorías para evitar demasiadas columnas one-hot
    for col in categorical_cols.copy():
        if col in ['PULocationID', 'DOLocationID'] and col in df_clean.columns:
            # Para PULocationID y DOLocationID, agrupar las zonas menos comunes
            value_counts = df_clean[col].value_counts()
            top_zones = value_counts.nlargest(20).index.tolist()
            df_clean[col] = df_clean[col].apply(lambda x: x if x in top_zones else -1)
    
    # One-hot encoding para variables categóricas
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    # Preparar datos
    X_cols = [col for col in df_encoded.columns if col in numeric_cols or 
              any(col.startswith(cat_col) for cat_col in categorical_cols)]
    X = df_encoded[X_cols]
    y = df_encoded['trip_time']
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # Crear pipeline con escalado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Modelo entrenado - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Guardar modelo
    model_path = os.path.join(MODEL_DIR, 'trip_time_predictor.joblib')
    joblib.dump(pipeline, model_path)
    
    # Guardar también los nombres de las características
    feature_path = os.path.join(MODEL_DIR, 'trip_time_features.joblib')
    joblib.dump(X_cols, feature_path)
    
    print(f"Modelo guardado en {model_path}")
    return pipeline

def analyze_feature_importance(models_dict):
    """Analiza la importancia de las características para cada modelo."""
    print("\n==== Analizando importancia de características ====")
    
    for model_name, model_info in models_dict.items():
        if model_info['model'] is None:
            continue
        
        print(f"\nImportancia de características para {model_name}:")
        
        # Extraer el modelo interno desde el pipeline
        pipeline = model_info['model']
        model = pipeline.named_steps['model']
        features = model_info['features']
        
        # Obtener importancias
        importances = model.feature_importances_
        
        # Crear DataFrame de importancias
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Mostrar las 10 características más importantes
        print(importance_df.head(10))
        
        # Guardar el DataFrame completo
        output_path = os.path.join(MODEL_DIR, f"{model_name}_feature_importance.csv")
        importance_df.to_csv(output_path, index=False)
        print(f"Importancia de características guardada en {output_path}")

def main():
    print("Iniciando entrenamiento de modelos...")
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Crear directorio de modelos si no existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Cargar datos
        df = load_data()
        
        # Entrenar modelos
        fare_model = train_fare_model(df)
        airport_model = train_airport_model(df)
        trip_time_model = train_trip_time_model(df)
        
        # Analizar importancia de características
        models_info = {
            'fare_predictor': {
                'model': fare_model,
                'features': joblib.load(os.path.join(MODEL_DIR, 'fare_features.joblib')) if fare_model else None
            },
            'airport_classifier': {
                'model': airport_model,
                'features': joblib.load(os.path.join(MODEL_DIR, 'airport_features.joblib')) if airport_model else None
            },
            'trip_time_predictor': {
                'model': trip_time_model,
                'features': joblib.load(os.path.join(MODEL_DIR, 'trip_time_features.joblib')) if trip_time_model else None
            }
        }
        
        analyze_feature_importance(models_info)
        
        print("\nEntrenamiento de modelos completado exitosamente.")
        
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
