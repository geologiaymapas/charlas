# Preparacion de datos

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Simularemos un dataset geotécnico realista
np.random.seed(42)

def generar_dataset_geotecnico(n_samples=500):
    """
    Genera un dataset sintético basado en correlaciones geotécnicas reales
    """
    # Propiedades índice
    limite_liquido = np.random.uniform(20, 80, n_samples)
    limite_plastico = limite_liquido * np.random.uniform(0.4, 0.7, n_samples)
    indice_plasticidad = limite_liquido - limite_plastico
    
    contenido_humedad = np.random.uniform(10, 35, n_samples)
    peso_especifico = np.random.uniform(16, 21, n_samples)  # kN/m³
    
    # Porcentaje de finos (pasa tamiz #200)
    finos = np.random.uniform(5, 85, n_samples)
    
    # Propiedades mecánicas (con correlaciones simplificadas)
    # Ángulo de fricción: inversamente proporcional a la plasticidad
    angulo_friccion = 40 - (indice_plasticidad * 0.3) + np.random.normal(0, 2, n_samples)
    angulo_friccion = np.clip(angulo_friccion, 15, 42)
    
    # Cohesión: directamente proporcional a la plasticidad
    cohesion = (indice_plasticidad * 0.5) + (finos * 0.1) + np.random.normal(0, 3, n_samples)
    cohesion = np.clip(cohesion, 0, 50)  # kPa
    
    df = pd.DataFrame({
        'limite_liquido': limite_liquido,
        'limite_plastico': limite_plastico,
        'indice_plasticidad': indice_plasticidad,
        'contenido_humedad': contenido_humedad,
        'peso_especifico': peso_especifico,
        'finos_pct': finos,
        'angulo_friccion': angulo_friccion,
        'cohesion': cohesion
    })
    
    return df

# Generar dataset
df = generar_dataset_geotecnico(500)
print("Dataset Geotécnico Generado:")
print(df.head())
print(f"\nDimensiones: {df.shape}")
print(f"\nEstadísticas descriptivas:")
print(df.describe())

# Analisis Descriptivo
# Matriz de correlación
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación - Propiedades Geotécnicas')
plt.tight_layout()
plt.savefig('correlacion_geotecnica.png', dpi=300)
plt.show()

# Visualización de relaciones clave
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df['indice_plasticidad'], df['angulo_friccion'], alpha=0.6)
axes[0, 0].set_xlabel('Índice de Plasticidad (%)')
axes[0, 0].set_ylabel('Ángulo de Fricción (°)')
axes[0, 0].set_title('IP vs Ángulo de Fricción')

axes[0, 1].scatter(df['indice_plasticidad'], df['cohesion'], alpha=0.6, color='orange')
axes[0, 1].set_xlabel('Índice de Plasticidad (%)')
axes[0, 1].set_ylabel('Cohesión (kPa)')
axes[0, 1].set_title('IP vs Cohesión')

axes[1, 0].scatter(df['finos_pct'], df['angulo_friccion'], alpha=0.6, color='green')
axes[1, 0].set_xlabel('Contenido de Finos (%)')
axes[1, 0].set_ylabel('Ángulo de Fricción (°)')
axes[1, 0].set_title('Finos vs Ángulo de Fricción')

axes[1, 1].scatter(df['peso_especifico'], df['angulo_friccion'], alpha=0.6, color='red')
axes[1, 1].set_xlabel('Peso Específico (kN/m³)')
axes[1, 1].set_ylabel('Ángulo de Fricción (°)')
axes[1, 1].set_title('Peso Específico vs Ángulo de Fricción')

plt.tight_layout()
plt.savefig('relaciones_geotecnicas.png', dpi=300)
plt.show()

#Modelo 1: Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Preparar datos
features = ['limite_liquido', 'limite_plastico', 'indice_plasticidad', 
            'contenido_humedad', 'peso_especifico', 'finos_pct']

X = df[features]
y_friccion = df['angulo_friccion']
y_cohesion = df['cohesion']

# Split datos
X_train, X_test, y_fric_train, y_fric_test = train_test_split(
    X, y_friccion, test_size=0.2, random_state=42
)

X_train, X_test, y_coh_train, y_coh_test = train_test_split(
    X, y_cohesion, test_size=0.2, random_state=42
)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo Random Forest para Ángulo de Fricción
rf_friccion = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_friccion.fit(X_train_scaled, y_fric_train)
y_fric_pred = rf_friccion.predict(X_test_scaled)

# Métricas
print("=" * 60)
print("MODELO RANDOM FOREST - ÁNGULO DE FRICCIÓN")
print("=" * 60)
print(f"R² Score: {r2_score(y_fric_test, y_fric_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_fric_test, y_fric_pred)):.4f}°")
print(f"MAE: {mean_absolute_error(y_fric_test, y_fric_pred):.4f}°")

# Modelo para Cohesión
rf_cohesion = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_cohesion.fit(X_train_scaled, y_coh_train)
y_coh_pred = rf_cohesion.predict(X_test_scaled)

print("\n" + "=" * 60)
print("MODELO RANDOM FOREST - COHESIÓN")
print("=" * 60)
print(f"R² Score: {r2_score(y_coh_test, y_coh_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_coh_test, y_coh_pred)):.4f} kPa")
print(f"MAE: {mean_absolute_error(y_coh_test, y_coh_pred):.4f} kPa")

# Importancia de características
importancias_fric = pd.DataFrame({
    'Feature': features,
    'Importance': rf_friccion.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nImportancia de Características (Ángulo de Fricción):")
print(importancias_fric)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Construir arquitectura de red neuronal
def crear_modelo_nn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001
)

# Modelo para Ángulo de Fricción
nn_friccion = crear_modelo_nn(X_train_scaled.shape[1])

print("\nArquitectura del Modelo Neural:")
nn_friccion.summary()

history_fric = nn_friccion.fit(
    X_train_scaled, y_fric_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Predicciones
y_fric_pred_nn = nn_friccion.predict(X_test_scaled, verbose=0).flatten()

print("\n" + "=" * 60)
print("MODELO NEURAL NETWORK - ÁNGULO DE FRICCIÓN")
print("=" * 60)
print(f"R² Score: {r2_score(y_fric_test, y_fric_pred_nn):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_fric_test, y_fric_pred_nn)):.4f}°")
print(f"MAE: {mean_absolute_error(y_fric_test, y_fric_pred_nn):.4f}°")

# Modelo para Cohesión
nn_cohesion = crear_modelo_nn(X_train_scaled.shape[1])

history_coh = nn_cohesion.fit(
    X_train_scaled, y_coh_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

y_coh_pred_nn = nn_cohesion.predict(X_test_scaled, verbose=0).flatten()

print("\n" + "=" * 60)
print("MODELO NEURAL NETWORK - COHESIÓN")
print("=" * 60)
print(f"R² Score: {r2_score(y_coh_test, y_coh_pred_nn):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_coh_test, y_coh_pred_nn)):.4f} kPa")
print(f"MAE: {mean_absolute_error(y_coh_test, y_coh_pred_nn):.4f} kPa")

#Visualizacion de datos

# Comparación de predicciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Random Forest - Fricción
axes[0, 0].scatter(y_fric_test, y_fric_pred, alpha=0.6)
axes[0, 0].plot([y_fric_test.min(), y_fric_test.max()], 
                [y_fric_test.min(), y_fric_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Ángulo de Fricción Real (°)')
axes[0, 0].set_ylabel('Predicción RF (°)')
axes[0, 0].set_title(f'Random Forest - Fricción (R²={r2_score(y_fric_test, y_fric_pred):.3f})')
axes[0, 0].grid(True, alpha=0.3)

# Neural Network - Fricción
axes[0, 1].scatter(y_fric_test, y_fric_pred_nn, alpha=0.6, color='orange')
axes[0, 1].plot([y_fric_test.min(), y_fric_test.max()], 
                [y_fric_test.min(), y_fric_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Ángulo de Fricción Real (°)')
axes[0, 1].set_ylabel('Predicción NN (°)')
axes[0, 1].set_title(f'Neural Network - Fricción (R²={r2_score(y_fric_test, y_fric_pred_nn):.3f})')
axes[0, 1].grid(True, alpha=0.3)

# Random Forest - Cohesión
axes[1, 0].scatter(y_coh_test, y_coh_pred, alpha=0.6, color='green')
axes[1, 0].plot([y_coh_test.min(), y_coh_test.max()], 
                [y_coh_test.min(), y_coh_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Cohesión Real (kPa)')
axes[1, 0].set_ylabel('Predicción RF (kPa)')
axes[1, 0].set_title(f'Random Forest - Cohesión (R²={r2_score(y_coh_test, y_coh_pred):.3f})')
axes[1, 0].grid(True, alpha=0.3)

# Neural Network - Cohesión
axes[1, 1].scatter(y_coh_test, y_coh_pred_nn, alpha=0.6, color='purple')
axes[1, 1].plot([y_coh_test.min(), y_coh_test.max()], 
                [y_coh_test.min(), y_coh_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Cohesión Real (kPa)')
axes[1, 1].set_ylabel('Predicción NN (kPa)')
axes[1, 1].set_title(f'Neural Network - Cohesión (R²={r2_score(y_coh_test, y_coh_pred_nn):.3f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=300)
plt.show()

# Curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_fric.history['loss'], label='Training Loss')
axes[0].plot(history_fric.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Curva de Aprendizaje - Ángulo de Fricción')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_coh.history['loss'], label='Training Loss')
axes[1].plot(history_coh.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Curva de Aprendizaje - Cohesión')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_aprendizaje.png', dpi=300)
plt.show()

#Aplicación Práctica: Predicción de Nuevas Muestras
def predecir_propiedades_suelo(limite_liq, limite_plas, humedad, 
                               peso_esp, finos, modelo_fric, modelo_coh, scaler):
    """
    Predice ángulo de fricción y cohesión para una nueva muestra de suelo
    """
    indice_plas = limite_liq - limite_plas
    
    muestra = np.array([[limite_liq, limite_plas, indice_plas, 
                        humedad, peso_esp, finos]])
    
    muestra_scaled = scaler.transform(muestra)
    
    friccion = modelo_fric.predict(muestra_scaled)[0]
    cohesion = modelo_coh.predict(muestra_scaled)[0]
    
    return friccion, cohesion

# Ejemplo de uso
print("\n" + "=" * 60)
print("PREDICCIÓN DE NUEVA MUESTRA")
print("=" * 60)

# Datos de una nueva muestra
nueva_muestra = {
    'LL': 45.0,
    'LP': 25.0,
    'w': 22.0,
    'γ': 18.5,
    'finos': 60.0
}

fric_pred, coh_pred = predecir_propiedades_suelo(
    nueva_muestra['LL'],
    nueva_muestra['LP'],
    nueva_muestra['w'],
    nueva_muestra['γ'],
    nueva_muestra['finos'],
    rf_friccion,
    rf_cohesion,
    scaler
)

print(f"\nCaracterísticas de entrada:")
print(f"  Límite Líquido: {nueva_muestra['LL']}%")
print(f"  Límite Plástico: {nueva_muestra['LP']}%")
print(f"  Índice de Plasticidad: {nueva_muestra['LL'] - nueva_muestra['LP']}%")
print(f"  Contenido de Humedad: {nueva_muestra['w']}%")
print(f"  Peso Específico: {nueva_muestra['γ']} kN/m³")
print(f"  Contenido de Finos: {nueva_muestra['finos']}%")

print(f"\nPredicciones:")
print(f"  Ángulo de Fricción: {fric_pred:.2f}°")
print(f"  Cohesión: {coh_pred:.2f} kPa")

#Guardando los Modelos

import joblib

# Guardar modelos
joblib.dump(rf_friccion, 'modelo_friccion_rf.pkl')
joblib.dump(rf_cohesion, 'modelo_cohesion_rf.pkl')
joblib.dump(scaler, 'scaler_geotecnico.pkl')

nn_friccion.save('modelo_friccion_nn.h5')
nn_cohesion.save('modelo_cohesion_nn.h5')

print("\n✓ Modelos guardados exitosamente")
