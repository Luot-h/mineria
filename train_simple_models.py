import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

# Directorio para guardar modelos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def simple_train():
    """Entrenar modelos simples con un conjunto básico de columnas."""
    print("Iniciando entrenamiento de modelos básicos...")
    
    # Cargar datos
    print("Cargando datos...")
    data_files = [f for f in os.listdir('data_sampled') if f.endswith('_reduced.parquet')]
    
    if not data_files:
        print("No se encontraron archivos de datos")
        return
    
    # Usar solo el primer archivo para probar
    print(f"Usando archivo: {data_files[0]}")
    df = pd.read_parquet(os.path.join('data_sampled', data_files[0]))
    print(f"Archivo cargado con éxito: {df.shape}")
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Mostrar columnas
    print("\nColumnas disponibles:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    
    print("\nEntrenando modelo simple de regresión...")
    
    # 1. Entrenar modelo simple de regresión (si hay driver_pay)
    if 'driver_pay' in df.columns:
        # Seleccionar características básicas
        numeric_cols = ['trip_miles', 'trip_time'] 
        
        # Verificar que las columnas existen
        for col in numeric_cols.copy():
            if col not in df.columns:
                print(f"Columna {col} no encontrada. Se omitirá.")
                numeric_cols.remove(col)
        
        if len(numeric_cols) == 0:
            print("No hay suficientes columnas para entrenar el modelo de regresión.")
        else:
            # Eliminar filas con valores nulos
            df_clean = df.dropna(subset=numeric_cols + ['driver_pay'])
            
            # Preparar datos
            X = df_clean[numeric_cols]
            y = df_clean['driver_pay']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Modelo de regresión - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Guardar modelo
            joblib.dump(model, os.path.join(MODEL_DIR, 'simple_fare_predictor.joblib'))
            
            # Guardar también los nombres de las características
            joblib.dump(numeric_cols, os.path.join(MODEL_DIR, 'simple_fare_features.joblib'))
            
            print("Modelo de regresión guardado exitosamente.")
    else:
        print("No se encontró la columna 'driver_pay'. No se puede entrenar el modelo de regresión.")
    
    # 2. Entrenar modelo simple de clasificación para aeropuertos
    print("\nEntrenando modelo simple de clasificación...")
    if 'to_airport' in df.columns:
        # Seleccionar características básicas
        feature_cols = ['trip_miles', 'trip_time', 'pickup_hour']
        
        # Verificar que las columnas existen
        for col in feature_cols.copy():
            if col not in df.columns:
                print(f"Columna {col} no encontrada. Se omitirá.")
                feature_cols.remove(col)
        
        if len(feature_cols) == 0:
            print("No hay suficientes columnas para entrenar el modelo de clasificación.")
        else:
            # Eliminar filas con valores nulos
            df_clean = df.dropna(subset=feature_cols + ['to_airport'])
            
            # Preparar datos
            X = df_clean[feature_cols]
            y = df_clean['to_airport']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Modelo de clasificación - Accuracy: {accuracy:.4f}")
            
            # Guardar modelo
            joblib.dump(model, os.path.join(MODEL_DIR, 'simple_airport_classifier.joblib'))
            
            # Guardar también los nombres de las características
            joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'simple_airport_features.joblib'))
            
            print("Modelo de clasificación guardado exitosamente.")
    else:
        print("No se encontró la columna 'to_airport'. No se puede entrenar el modelo de clasificación.")
    
    print("\nEntrenamiento completado.")

if __name__ == "__main__":
    simple_train()
