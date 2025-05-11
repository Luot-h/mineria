import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib

# Directorio para guardar modelos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    """Entrenamiento muy básico de modelos con datos mínimos para hacer funcionar la pestaña ML."""
    print("Creando modelos básicos para la demostración...")
    print(f"Directorio de modelos: {MODEL_DIR}")
    
    try:
        print(f"Verificando si existe el directorio: {os.path.exists(MODEL_DIR)}")
        if not os.path.exists(MODEL_DIR):
            print(f"Creando directorio de modelos: {MODEL_DIR}")
            os.makedirs(MODEL_DIR)
        print(f"Directorio de modelos existe: {os.path.exists(MODEL_DIR)}")
        # Cargar solo un archivo de datos de muestra
        file_path = os.path.join('data_sampled', '2024 - 1_reduced.parquet')
        df = pd.read_parquet(file_path)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mantener solo 10,000 filas para rapidez
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            print(f"Muestra reducida a {df.shape[0]} filas")
        
        # 1. Modelo de predicción de tarifa (driver_pay)
        if 'driver_pay' in df.columns and 'trip_miles' in df.columns:
            print("\nEntrenando modelo de predicción de tarifa...")
            # Características básicas
            features = ['trip_miles', 'trip_time', 'pickup_hour']
            features = [f for f in features if f in df.columns]
            
            # Limpiar datos
            data = df[features + ['driver_pay']].dropna()
            
            # Dividir datos
            X = data[features]
            y = data['driver_pay']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Modelo de tarifa - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Guardar modelo y métricas
            joblib.dump(model, os.path.join(MODEL_DIR, 'driver_pay_predictor.joblib'))
            metrics = {'rmse': rmse, 'r2': r2, 'best_model': 'RandomForest'}
            joblib.dump(metrics, os.path.join(MODEL_DIR, 'driver_pay_predictor_metrics.joblib'))
            print("Modelo de tarifa guardado.")
            
            # Guardar importancia de características
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            imp_df.to_csv(os.path.join(MODEL_DIR, 'driver_pay_predictor_feature_importance.csv'), index=False)
        
        # 2. Modelo de clasificación de aeropuertos
        print("\nEntrenando modelo de clasificación de aeropuertos...")
        
        # Crear etiqueta sintética para viajes a aeropuertos
        if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
            # JFK (132), LaGuardia (138), Newark (1)
            airport_zones = [1, 132, 138]
            df['airport_trip'] = df['DOLocationID'].isin(airport_zones) | df['PULocationID'].isin(airport_zones)
            
            # Características
            features = ['trip_miles', 'trip_time', 'pickup_hour']
            features = [f for f in features if f in df.columns]
            
            # Limpiar datos
            data = df[features + ['airport_trip']].dropna()
            
            # Dividir datos
            X = data[features]
            y = data['airport_trip']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Modelo de aeropuertos - Accuracy: {accuracy:.4f}")
            
            # Guardar modelo y métricas
            joblib.dump(model, os.path.join(MODEL_DIR, 'airport_classifier.joblib'))
            metrics = {'accuracy': accuracy, 'best_model': 'RandomForest'}
            joblib.dump(metrics, os.path.join(MODEL_DIR, 'airport_classifier_metrics.joblib'))
            print("Modelo de aeropuertos guardado.")
            
            # Guardar importancia de características
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            imp_df.to_csv(os.path.join(MODEL_DIR, 'airport_classifier_feature_importance.csv'), index=False)
        
        # 3. Modelo de predicción de duración (trip_time)
        if 'trip_time' in df.columns and 'trip_miles' in df.columns:
            print("\nEntrenando modelo de predicción de duración...")
            # Características básicas
            features = ['trip_miles', 'base_passenger_fare', 'pickup_hour']
            features = [f for f in features if f in df.columns]
            
            # Limpiar datos
            data = df[features + ['trip_time']].dropna()
            
            # Dividir datos
            X = data[features]
            y = data['trip_time']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Modelo de duración - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Guardar modelo y métricas
            joblib.dump(model, os.path.join(MODEL_DIR, 'trip_time_predictor.joblib'))
            metrics = {'rmse': rmse, 'r2': r2, 'best_model': 'RandomForest'}
            joblib.dump(metrics, os.path.join(MODEL_DIR, 'trip_time_predictor_metrics.joblib'))
            print("Modelo de duración guardado.")
            
            # Guardar importancia de características
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            imp_df.to_csv(os.path.join(MODEL_DIR, 'trip_time_predictor_feature_importance.csv'), index=False)
        
        print("\nEntrenamiento completado con éxito. Archivos guardados en:", MODEL_DIR)
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
