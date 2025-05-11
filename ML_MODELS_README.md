# Implementación de Modelos ML para Dashboard de Transporte

Este documento explica la implementación de modelos de machine learning para el análisis predictivo de datos de viajes de transporte en NYC.

## Modelos Implementados

Se han implementado tres tipos principales de modelos:

1. **Predicción de Costo de Viaje**: Modelo para predecir la tarifa (`driver_pay`) basado en características del viaje.
2. **Clasificación de Viajes a Aeropuertos**: Modelo para identificar si un viaje es hacia o desde un aeropuerto.
3. **Predicción de Duración de Viaje**: Modelo para estimar la duración del viaje (`trip_time`) en segundos.

## Estructura de Archivos

- `train_models.py`: Script principal para entrenar modelos utilizando TensorFlow con GPU.
- `train_models_no_tf.py`: Versión alternativa que no requiere TensorFlow (solo scikit-learn y XGBoost).
- `train_final_models.py`: Script optimizado para entrenar modelos con datos de muestra.
- `create_demo_models.py`: Script simplificado para crear modelos de demostración rápidamente.
- `model_utils.py`: Biblioteca de utilidades para cargar y utilizar los modelos entrenados.
- `models/`: Directorio que contiene los modelos guardados y métricas asociadas.

## Entrenamiento de Modelos

Para entrenar los modelos, ejecuta uno de los siguientes scripts:

```bash
# Versión completa con TensorFlow (requiere GPU)
python train_models.py

# Versión sin dependencia de TensorFlow
python train_models_no_tf.py

# Versión simplificada para demostración rápida
python create_demo_models.py
```

Los modelos se guardan en la carpeta `models/` en formato joblib.

## Uso de Modelos en el Dashboard

La pestaña "🤖 Modelos ML" del dashboard permite:

1. **Predicción de Tarifa**: Ingresa características del viaje para predecir el costo.
2. **Clasificación de Aeropuertos**: Determina si un viaje es hacia/desde un aeropuerto.
3. **Análisis de Features**: Visualiza la importancia de las características para cada modelo.

## Estructura de Modelos

Cada modelo se compone de:
- Archivo `.joblib` con el modelo entrenado
- Archivo `_metrics.joblib` con métricas de desempeño
- Archivo `_feature_importance.csv` con la importancia de características

## Requisitos

- Python 3.10+
- scikit-learn
- pandas
- numpy
- joblib
- TensorFlow (opcional, para redes neuronales)
- XGBoost (opcional, para modelos avanzados)

## Resolución de Problemas

Si encuentras errores con TensorFlow y protobuf, ejecuta:
```bash
pip install protobuf==3.20.3
```

Para ejecutar sin TensorFlow, usa los scripts alternativos que solo utilizan scikit-learn.

## Referencias

Para más información sobre los métodos de entrenamiento utilizados, consulta:
- [Documentación de scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Documentación de TensorFlow](https://www.tensorflow.org/tutorials/keras/regression)
