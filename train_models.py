import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.impute import SimpleImputer
import joblib
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Verificar si hay una GPU disponible y configurarla
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Info: ", tf.config.list_physical_devices('GPU'))

# Configuración para utilizar GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU configurada correctamente")

# Directorio para guardar modelos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Cargar datos
def load_data():
    """Carga los datos de los archivos parquet reducidos."""
    try:
        # Usar los archivos de muestra de data_sampled
        data_files = [f for f in os.listdir('data_sampled') if f.endswith('_reduced.parquet')]
        
        if not data_files:
            raise Exception("No se encontraron archivos de datos")
        
        print(f"Encontrados {len(data_files)} archivos de datos reducidos")
        
        # Cargar y concatenar todos los archivos
        dfs = []
        for file in data_files:
            df = pd.read_parquet(os.path.join('data_sampled', file))
            print(f"Cargado {file}: {df.shape[0]} filas, {df.shape[1]} columnas")
            dfs.append(df)
        
        # Concatenar todos los DataFrames
        df = pd.concat(dfs, ignore_index=True)
        print(f"Datos combinados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise

# Preparación de datos
def prepare_data(df, target_variable, task='regression'):
    """
    Prepara los datos para modelado.
    
    Args:
        df: DataFrame con los datos
        target_variable: Variable objetivo (ej: 'driver_pay', 'trip_time')
        task: 'regression' o 'classification'
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    print(f"Preparando datos para {task} con target '{target_variable}'")
    
    # Verificar si la columna objetivo existe
    if target_variable not in df.columns:
        raise ValueError(f"La columna {target_variable} no existe en el DataFrame")
    
    # Eliminar filas con valores nulos en la variable objetivo
    if df[target_variable].isnull().sum() > 0:
        print(f"Eliminando {df[target_variable].isnull().sum()} filas con valores nulos en la variable objetivo")
        df = df.dropna(subset=[target_variable])
    
    # Identificar tipos de columnas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    # Eliminar la variable objetivo de las listas de características
    if target_variable in numeric_features:
        numeric_features.remove(target_variable)
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)
    
    # Eliminar columnas que no deberían usarse para modelado
    cols_to_drop = ['pickup_datetime', 'dropoff_datetime', 'pickup_date']
    
    # Agregar columnas con demasiados valores únicos (pueden causar problemas)
    cols_to_drop.extend(['pickup_zone', 'dropoff_zone', 'pickup_borough', 'dropoff_borough'])
    
    for col in cols_to_drop:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    # Eliminar columnas con demasiados valores nulos
    for col in numeric_features.copy():
        if df[col].isnull().mean() > 0.3:  # Si más del 30% son nulos
            print(f"Eliminando columna {col} con {df[col].isnull().mean()*100:.1f}% de valores nulos")
            numeric_features.remove(col)
    
    for col in categorical_features.copy():
        if df[col].isnull().mean() > 0.3:  # Si más del 30% son nulos
            print(f"Eliminando columna {col} con {df[col].isnull().mean()*100:.1f}% de valores nulos")
            categorical_features.remove(col)
    
    print(f"Características numéricas ({len(numeric_features)}): {numeric_features}")
    print(f"Características categóricas ({len(categorical_features)}): {categorical_features}")
    
    # Preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Dividir en conjunto de entrenamiento y prueba
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # Para clasificación, convertir a categoría si es necesario
    if task == 'classification':
        y = pd.Categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test, preprocessor

# Entrenamiento de modelos de regresión
def train_regression_model(X_train, X_test, y_train, y_test, preprocessor, model_name='driver_pay_predictor'):
    """Entrena un modelo de regresión para predecir una variable continua."""
    print(f"Entrenando modelo de regresión: {model_name}")
    
    # Pipeline con preprocesamiento y modelo
    model_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Pipeline con XGBoost
    model_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', 
                                       n_estimators=100, 
                                       learning_rate=0.1,
                                       random_state=42,
                                       tree_method='gpu_hist' if len(tf.config.list_physical_devices('GPU')) > 0 else 'auto'))
    ])
    
    # Entrenar RandomForest
    print("Entrenando RandomForest...")
    model_rf.fit(X_train, y_train)
    rf_pred = model_rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"RandomForest - RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
    
    # Entrenar XGBoost
    print("Entrenando XGBoost...")
    model_xgb.fit(X_train, y_train)
    xgb_pred = model_xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"XGBoost - RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")
    
    # Elegir el mejor modelo
    best_model = model_xgb if xgb_rmse < rf_rmse else model_rf
    best_model_name = "XGBoost" if xgb_rmse < rf_rmse else "RandomForest"
    
    print(f"Mejor modelo: {best_model_name}")
    
    # Guardar el modelo
    model_filename = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Modelo guardado como {model_filename}")
    
    # Guardar métricas
    metrics = {
        'rf_rmse': rf_rmse,
        'rf_r2': rf_r2,
        'xgb_rmse': xgb_rmse,
        'xgb_r2': xgb_r2,
        'best_model': best_model_name
    }
    
    metrics_filename = os.path.join(MODEL_DIR, f"{model_name}_metrics.joblib")
    joblib.dump(metrics, metrics_filename)
    
    return best_model, metrics

# Entrenamiento de modelos de clasificación
def train_classification_model(X_train, X_test, y_train, y_test, preprocessor, model_name='trip_type_classifier'):
    """Entrena un modelo de clasificación."""
    print(f"Entrenando modelo de clasificación: {model_name}")
    
    # Pipeline con RandomForest
    model_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Pipeline con XGBoost
    model_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(objective='multi:softprob', 
                                         n_estimators=100, 
                                         learning_rate=0.1,
                                         random_state=42,
                                         tree_method='gpu_hist' if len(tf.config.list_physical_devices('GPU')) > 0 else 'auto'))
    ])
    
    # Entrenar RandomForest
    print("Entrenando RandomForest Classifier...")
    model_rf.fit(X_train, y_train)
    rf_pred = model_rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"RandomForest - Accuracy: {rf_accuracy:.4f}")
    print(classification_report(y_test, rf_pred))
    
    # Entrenar XGBoost
    print("Entrenando XGBoost Classifier...")
    model_xgb.fit(X_train, y_train)
    xgb_pred = model_xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}")
    print(classification_report(y_test, xgb_pred))
    
    # Elegir el mejor modelo
    best_model = model_xgb if xgb_accuracy > rf_accuracy else model_rf
    best_model_name = "XGBoost" if xgb_accuracy > rf_accuracy else "RandomForest"
    
    print(f"Mejor modelo: {best_model_name}")
    
    # Guardar el modelo
    model_filename = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Modelo guardado como {model_filename}")
    
    # Guardar métricas
    metrics = {
        'rf_accuracy': rf_accuracy,
        'xgb_accuracy': xgb_accuracy,
        'best_model': best_model_name
    }
    
    metrics_filename = os.path.join(MODEL_DIR, f"{model_name}_metrics.joblib")
    joblib.dump(metrics, metrics_filename)
    
    return best_model, metrics

# Función para entrenar modelo de red neuronal (aprovechando GPU)
def train_neural_network(X_train, X_test, y_train, y_test, preprocessor, model_name='nn_predictor', task='regression', max_epochs=50):
    """Entrena un modelo de red neuronal usando TensorFlow/Keras."""
    print(f"Entrenando red neuronal para {task}: {model_name}")
    
    # Aplicar preprocesamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Obtener número de features después del preprocesamiento
    n_features = X_train_processed.shape[1]
    
    # Definir arquitectura según la tarea
    if task == 'regression':
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)  # Sin activación para regresión
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    else:  # classification
        # Para clasificación binaria
        if len(np.unique(y_train)) == 2:
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
        else:  # Para clasificación multiclase
            n_classes = len(np.unique(y_train))
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(n_classes, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callback para early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        patience=5,  # Reducido para entrenamientos más cortos
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train_processed, y_train,
        epochs=max_epochs,  # Usar el nuevo parámetro
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluar el modelo
    print("Evaluando modelo...")
    
    if task == 'regression':
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"NN - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
    else:  # classification
        if len(np.unique(y_train)) == 2:  # binaria
            y_pred = (model.predict(X_test_processed) > 0.5).astype(int)
        else:  # multiclase
            y_pred = np.argmax(model.predict(X_test_processed), axis=1)
            
        accuracy = accuracy_score(y_test, y_pred)
        print(f"NN - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy
        }
    
    # Guardar el modelo Keras
    model_save_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_save_path)
    print(f"Modelo guardado en {model_save_path}")
    
    # Guardar el preprocesador y métricas
    preprocessor_filename = os.path.join(MODEL_DIR, f"{model_name}_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_filename)
    
    metrics_filename = os.path.join(MODEL_DIR, f"{model_name}_metrics.joblib")
    joblib.dump(metrics, metrics_filename)
    
    # Plot de la historia de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    if task == 'regression':
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('MAE del modelo')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    else:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_training_history.png"))
    
    return model, metrics

# Función principal para ejecutar el entrenamiento
def main():
    print("Iniciando entrenamiento de modelos...")
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Crear directorio de modelos si no existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Cargar datos
    df = load_data()
    
    # Mostrar información de las columnas disponibles
    print("\nColumnas disponibles en el dataset:")
    for col in df.columns:
        missing = df[col].isnull().mean() * 100
        print(f"- {col}: {df[col].dtype} ({missing:.1f}% valores nulos)")
    
    # 1. Modelo para predecir el costo del viaje (driver_pay)
    if 'driver_pay' in df.columns:
        print("\n" + "="*50)
        print("ENTRENANDO MODELO DE PREDICCIÓN DE COSTO (DRIVER_PAY)")
        print("="*50)
        try:
            X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, 'driver_pay', 'regression')
            
            # Entrenar RandomForest y XGBoost
            train_regression_model(X_train, X_test, y_train, y_test, preprocessor, 'driver_pay_predictor')
            
            # Entrenar red neuronal con menos epochs para datos más pequeños
            train_neural_network(X_train, X_test, y_train, y_test, preprocessor, 'driver_pay_nn', 'regression', max_epochs=30)
        except Exception as e:
            print(f"Error al entrenar modelo de predicción de costo: {e}")
    else:
        print("\nColumna 'driver_pay' no encontrada. No se puede entrenar modelo de predicción de costo.")
    
    # 2. Modelo para predecir si un viaje es hacia/desde un aeropuerto
    if 'to_airport' in df.columns:
        print("\n" + "="*50)
        print("ENTRENANDO MODELO DE CLASIFICACIÓN DE VIAJES A AEROPUERTOS")
        print("="*50)
        try:
            X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, 'to_airport', 'classification')
            
            # Entrenar RandomForest y XGBoost
            train_classification_model(X_train, X_test, y_train, y_test, preprocessor, 'airport_classifier')
            
            # Entrenar red neuronal con menos epochs para datos más pequeños
            train_neural_network(X_train, X_test, y_train, y_test, preprocessor, 'airport_nn', 'classification', max_epochs=30)
        except Exception as e:
            print(f"Error al entrenar modelo de clasificación de aeropuertos: {e}")
    else:
        print("\nColumna 'to_airport' no encontrada. No se puede entrenar modelo de clasificación de aeropuertos.")
    
    # 3. Modelo para predecir la duración del viaje (trip_time)
    if 'trip_time' in df.columns:
        print("\n" + "="*50)
        print("ENTRENANDO MODELO DE PREDICCIÓN DE DURACIÓN (TRIP_TIME)")
        print("="*50)
        try:
            X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, 'trip_time', 'regression')
            
            # Entrenar RandomForest y XGBoost
            train_regression_model(X_train, X_test, y_train, y_test, preprocessor, 'trip_time_predictor')
            
            # Entrenar red neuronal con menos epochs para datos más pequeños
            train_neural_network(X_train, X_test, y_train, y_test, preprocessor, 'trip_time_nn', 'regression', max_epochs=30)
        except Exception as e:
            print(f"Error al entrenar modelo de predicción de duración: {e}")
    else:
        print("\nColumna 'trip_time' no encontrada. No se puede entrenar modelo de predicción de duración.")
    
    print("\nEntrenamiento de modelos completado.")

if __name__ == "__main__":
    main()
