# Dataset: Generación de Casos de Estudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

def calcular_factor_seguridad_bishop(altura, angulo, cohesion, friccion, 
                                    peso_unitario, nivel_freatico_rel, 
                                    sobrecarga=0):
    """
    Implementación simplificada del método de Bishop para calcular FS
    Esta es una aproximación para generar datos de entrenamiento
    """
    # Convertir ángulo a radianes
    beta = np.radians(angulo)
    phi = np.radians(friccion)
    
    # Parámetros geométricos
    L = altura / np.sin(beta)  # Longitud de la superficie de falla
    
    # Peso de la cuña de falla
    W = 0.5 * peso_unitario * altura * L * np.cos(beta)
    
    # Efecto del agua
    if nivel_freatico_rel > 0:
        hw = altura * nivel_freatico_rel
        u = 0.5 * 9.81 * hw  # Presión de poros promedio
        peso_sumergido = peso_unitario - 9.81
        W_efectivo = W * (peso_sumergido / peso_unitario)
    else:
        u = 0
        W_efectivo = W
    
    # Agregar sobrecarga
    W_total = W_efectivo + (sobrecarga * L * np.cos(beta))
    
    # Fuerzas resistentes
    N = W_total * np.cos(beta)
    c_total = cohesion * L
    resistencia_friccion = (N - u * L) * np.tan(phi)
    
    # Fuerzas actuantes
    T = W_total * np.sin(beta)
    
    # Factor de seguridad
    FS = (c_total + resistencia_friccion) / max(T, 0.1)
    
    return max(min(FS, 5.0), 0.1)  # Limitar valores extremos

def generar_dataset_taludes(n_samples=2000):
    """
    Genera dataset de análisis de estabilidad de taludes
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Parámetros del talud
        altura = np.random.uniform(3, 30)  # metros
        angulo = np.random.uniform(20, 70)  # grados
        
        # Propiedades del suelo
        cohesion = np.random.uniform(0, 50)  # kPa
        friccion = np.random.uniform(15, 40)  # grados
        peso_unitario = np.random.uniform(16, 22)  # kN/m³
        
        # Condiciones hidrogeológicas
        nivel_freatico = np.random.uniform(0, 1)  # 0=seco, 1=saturado
        
        # Cargas
        sobrecarga = np.random.uniform(0, 50)  # kPa
        
        # Estratificación (1=homogéneo, 2=dos capas)
        capas = np.random.choice([1, 2], p=[0.7, 0.3])
        
        # Calcular factor de seguridad
        fs = calcular_factor_seguridad_bishop(
            altura, angulo, cohesion, friccion, 
            peso_unitario, nivel_freatico, sobrecarga
        )
        
        # Clasificación de estabilidad
        if fs < 1.0:
            estabilidad = 'Inestable'
        elif fs < 1.5:
            estabilidad = 'Marginal'
        else:
            estabilidad = 'Estable'
        
        data.append({
            'altura': altura,
            'angulo': angulo,
            'cohesion': cohesion,
            'friccion': friccion,
            'peso_unitario': peso_unitario,
            'nivel_freatico': nivel_freatico,
            'sobrecarga': sobrecarga,
            'num_capas': capas,
            'factor_seguridad': fs,
            'estabilidad': estabilidad
        })
    
    return pd.DataFrame(data)

# Generar dataset
df_taludes = generar_dataset_taludes(2000)

print("Dataset de Estabilidad de Taludes:")
print(df_taludes.head(10))
print(f"\nDimensiones: {df_taludes.shape}")
print(f"\nDistribución de estabilidad:")
print(df_taludes['estabilidad'].value_counts())
print(f"\nEstadísticas del Factor de Seguridad:")
print(df_taludes['factor_seguridad'].describe())

# Analisis de datos
# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

fig = plt.figure(figsize=(18, 12))

# 1. Distribución del Factor de Seguridad
ax1 = plt.subplot(3, 3, 1)
df_taludes['factor_seguridad'].hist(bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='FS=1.0 (Falla)')
plt.axvline(x=1.5, color='green', linestyle='--', linewidth=2, label='FS=1.5 (Seguro)')
plt.xlabel('Factor de Seguridad')
plt.ylabel('Frecuencia')
plt.title('Distribución del Factor de Seguridad')
plt.legend()

# 2. FS vs Ángulo del Talud
ax2 = plt.subplot(3, 3, 2)
scatter = plt.scatter(df_taludes['angulo'], df_taludes['factor_seguridad'], 
                     c=df_taludes['cohesion'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cohesión (kPa)')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=1.5, color='green', linestyle='--', alpha=0.5)
plt.xlabel('Ángulo del Talud (°)')
plt.ylabel('Factor de Seguridad')
plt.title('FS vs Ángulo del Talud')

# 3. FS vs Altura
ax3 = plt.subplot(3, 3, 3)
scatter = plt.scatter(df_taludes['altura'], df_taludes['factor_seguridad'], 
                     c=df_taludes['friccion'], cmap='plasma', alpha=0.6)
plt.colorbar(scatter, label='Fricción (°)')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=1.5, color='green', linestyle='--', alpha=0.5)
plt.xlabel('Altura del Talud (m)')
plt.ylabel('Factor de Seguridad')
plt.title('FS vs Altura del Talud')

# 4. Efecto del Nivel Freático
ax4 = plt.subplot(3, 3, 4)
df_taludes.boxplot(column='factor_seguridad', by='estabilidad', ax=ax4)
plt.xlabel('Estado de Estabilidad')
plt.ylabel('Factor de Seguridad')
plt.title('FS por Categoría de Estabilidad')
plt.suptitle('')

# 5. Cohesión vs Fricción coloreado por FS
ax5 = plt.subplot(3, 3, 5)
scatter = plt.scatter(df_taludes['cohesion'], df_taludes['friccion'], 
                     c=df_taludes['factor_seguridad'], cmap='RdYlGn', 
                     alpha=0.6, vmin=0.5, vmax=2.5)
plt.colorbar(scatter, label='Factor de Seguridad')
plt.xlabel('Cohesión (kPa)')
plt.ylabel('Ángulo de Fricción (°)')
plt.title('Resistencia del Suelo vs FS')

# 6. Matriz de correlación
ax6 = plt.subplot(3, 3, 6)
features_corr = ['altura', 'angulo', 'cohesion', 'friccion', 
                 'peso_unitario', 'nivel_freatico', 'factor_seguridad']
corr_matrix = df_taludes[features_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax6, cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación')

# 7. Nivel freático vs FS
ax7 = plt.subplot(3, 3, 7)
plt.scatter(df_taludes['nivel_freatico'], df_taludes['factor_seguridad'], 
           alpha=0.4)
plt.xlabel('Nivel Freático Relativo')
plt.ylabel('Factor de Seguridad')
plt.title('Efecto del Nivel Freático')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

# 8. Sobrecarga vs FS
ax8 = plt.subplot(3, 3, 8)
plt.scatter(df_taludes['sobrecarga'], df_taludes['factor_seguridad'], 
           alpha=0.4, color='orange')
plt.xlabel('Sobrecarga (kPa)')
plt.ylabel('Factor de Seguridad')
plt.title('Efecto de Sobrecarga')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

# 9. Distribución por categoría
ax9 = plt.subplot(3, 3, 9)
estabilidad_counts = df_taludes['estabilidad'].value_counts()
colors = {'Estable': 'green', 'Marginal': 'orange', 'Inestable': 'red'}
bars = plt.bar(estabilidad_counts.index, estabilidad_counts.values, 
               color=[colors[x] for x in estabilidad_counts.index])
plt.xlabel('Categoría')
plt.ylabel('Cantidad')
plt.title('Distribución de Casos por Estabilidad')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('analisis_exploratorio_taludes.png', dpi=300, bbox_inches='tight')
plt.show()

# Modelo 1: Red Neuronal Profunda para Regresión (FS)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Preparar datos
features = ['altura', 'angulo', 'cohesion', 'friccion', 
            'peso_unitario', 'nivel_freatico', 'sobrecarga', 'num_capas']

X = df_taludes[features].values
y = df_taludes['factor_seguridad'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Arquitectura de red profunda
def crear_modelo_dnn_regresion():
    model = keras.Sequential([
        layers.Input(shape=(8,)),
        
        # Bloque 1
        layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Bloque 2
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Bloque 3
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Bloque 4
        layers.Dense(16, activation='relu'),
        
        # Output
        layers.Dense(1, activation='linear')
    ])
    
    # Compilar con tasa de aprendizaje adaptativa
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# Crear modelo
model_dnn = crear_modelo_dnn_regresion()
print("Arquitectura del Modelo DNN:")
model_dnn.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'mejor_modelo_fs.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Entrenamiento
print("\nIniciando entrenamiento del modelo DNN...")
history = model_dnn.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# Evaluación
y_pred_dnn = model_dnn.predict(X_test_scaled, verbose=0).flatten()

print("\n" + "="*70)
print("RESULTADOS DEL MODELO DNN - PREDICCIÓN DE FACTOR DE SEGURIDAD")
print("="*70)
print(f"R² Score: {r2_score(y_test, y_pred_dnn):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_dnn)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_dnn):.4f}")

# Calcular error porcentual
error_porcentual = np.abs((y_test - y_pred_dnn) / y_test) * 100
print(f"Error Porcentual Promedio: {error_porcentual.mean():.2f}%")
print(f"Error Porcentual Mediano: {np.median(error_porcentual):.2f}%")

# Modelo 2: Clasificación Multiclase (Estable/Marginal/Inestable)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Preparar datos para clasificación
y_class = df_taludes['estabilidad'].values
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)
y_class_categorical = to_categorical(y_class_encoded)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class_categorical, test_size=0.2, random_state=42, stratify=y_class_encoded
)

# Normalizar
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Arquitectura para clasificación
def crear_modelo_clasificacion():
    model = keras.Sequential([
        layers.Input(shape=(8,)),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(3, activation='softmax')  # 3 clases
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

model_class = crear_modelo_clasificacion()
print("\nArquitectura del Modelo de Clasificación:")
model_class.summary()

# Entrenamiento
history_class = model_class.fit(
    X_train_c_scaled, y_train_c,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Predicciones
y_pred_class_prob = model_class.predict(X_test_c_scaled, verbose=0)
y_pred_class = np.argmax(y_pred_class_prob, axis=1)
y_test_class_labels = np.argmax(y_test_c, axis=1)

# Reporte de clasificación
print("\n" + "="*70)
print("RESULTADOS DEL MODELO DE CLASIFICACIÓN")
print("="*70)
print("\nReporte de Clasificación:")
print(classification_report(y_test_class_labels, y_pred_class, 
                          target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test_class_labels, y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Clasificación de Estabilidad')
plt.tight_layout()
plt.savefig('matriz_confusion_estabilidad.png', dpi=300)
plt.show()

# Visualización de Resultados

# Visualización completa de resultados
fig = plt.figure(figsize=(18, 12))

# 1. Curvas de aprendizaje - Regresión
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.title('Curvas de Aprendizaje - Modelo de Regresión')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. MAE durante entrenamiento
ax2 = plt.subplot(2, 3, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Época')
plt.ylabel('MAE')
plt.title('Error Absoluto Medio durante Entrenamiento')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Predicciones vs Valores Reales
ax3 = plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_dnn, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=3, label='Predicción Perfecta')
plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='FS=1.0')
plt.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='FS=1.5')
plt.xlabel('Factor de Seguridad Real')
plt.ylabel('Factor de Seguridad Predicho')
plt.title(f'Predicciones del Modelo DNN (R²={r2_score(y_test, y_pred_dnn):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Distribución de errores
ax4 = plt.subplot(2, 3, 4)
errores = y_test - y_pred_dnn
plt.hist(errores, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.title(f'Distribución de Errores (Media: {errores.mean():.4f})')
plt.grid(True, alpha=0.3)

# 5. Curvas de aprendizaje - Clasificación
ax5 = plt.subplot(2, 3, 5)
plt.plot(history_class.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history_class.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Accuracy durante Entrenamiento - Clasificación')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Análisis de error por rango de FS
ax6 = plt.subplot(2, 3, 6)
bins_fs = [0, 1.0, 1.5, 2.0, 5.0]
labels_fs = ['<1.0\n(Inestable)', '1.0-1.5\n(Marginal)', 
             '1.5-2.0\n(Estable)', '>2.0\n(Muy Estable)']
y_test_binned = pd.cut(y_test, bins=bins_fs, labels=labels_fs)

error_by_bin = []
for label in labels_fs:
    mask = y_test_binned == label
    if mask.sum() > 0:
        error_by_bin.append(np.abs(errores[mask]).mean())
    else:
        error_by_bin.append(0)

plt.bar(labels_fs, error_by_bin, color=['red', 'orange', 'lightgreen', 'green'])
plt.xlabel('Rango de Factor de Seguridad')
plt.ylabel('MAE Promedio')
plt.title('Error Promedio por Rango de FS')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('resultados_completos_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

# Aplicación Práctica: Sistema de Predicción Interactivo

class SistemaPredictionTaludes:
    """
    Sistema completo para predicción de estabilidad de taludes
    """
    def __init__(self, modelo_regresion, modelo_clasificacion, 
                 scaler, label_encoder):
        self.modelo_fs = modelo_regresion
        self.modelo_clase = modelo_clasificacion
        self.scaler = scaler
        self.label_encoder = label_encoder
    
    def predecir_estabilidad(self, altura, angulo, cohesion, friccion,
                           peso_unitario, nivel_freatico, sobrecarga, num_capas):
        """
        Predice el factor de seguridad y la clasificación de estabilidad
        """
        # Preparar entrada
        entrada = np.array([[altura, angulo, cohesion, friccion,
                           peso_unitario, nivel_freatico, sobrecarga, num_capas]])
        entrada_scaled = self.scaler.transform(entrada)
        
        # Predicción de FS
        fs_pred = self.modelo_fs.predict(entrada_scaled, verbose=0)[0][0]
        
        # Predicción de clasificación
        clase_prob = self.modelo_clase.predict(entrada_scaled, verbose=0)[0]
        clase_idx = np.argmax(clase_prob)
        clase_nombre = self.label_encoder.classes_[clase_idx]
        
        return {
            'factor_seguridad': fs_pred,
            'clasificacion': clase_nombre,
            'probabilidades': {
                self.label_encoder.classes_[i]: float(clase_prob[i])
                for i in range(len(self.label_encoder.classes_))
            }
        }
    
    def analizar_sensibilidad(self, caso_base, variable, rango_valores):
        """
        Análisis de sensibilidad variando una variable
        """
        resultados = []
        
        for valor in rango_valores:
            caso = caso_base.copy()
            caso[variable] = valor
            
            pred = self.predecir_estabilidad(**caso)
            resultados.append({
                variable: valor,
                'fs': pred['factor_seguridad'],
                'clase': pred['clasificacion']
            })
        
        return pd.DataFrame(resultados)
    
    def generar_reporte(self, caso):
        """
        Genera un reporte completo de estabilidad
        """
        pred = self.predecir_estabilidad(**caso)
        
        print("="*70)
        print("REPORTE DE ANÁLISIS DE ESTABILIDAD DE TALUD")
        print("="*70)
        print("\nPARÁMETROS DE ENTRADA:")
        print(f"  Altura del talud: {caso['altura']:.2f} m")
        print(f"  Ángulo del talud: {caso['angulo']:.2f}°")
        print(f"  Cohesión: {caso['cohesion']:.2f} kPa")
        print(f"  Ángulo de fricción: {caso['friccion']:.2f}°")
        print(f"  Peso unitario: {caso['peso_unitario']:.2f} kN/m³")
        print(f"  Nivel freático: {caso['nivel_freatico']*100:.1f}% de la altura")
        print(f"  Sobrecarga: {caso['sobrecarga']:.2f} kPa")
        print(f"  Número de capas: {caso['num_capas']}")
        
        print(f"\nRESULTADOS:")
        print(f"  Factor de Seguridad Predicho: {pred['factor_seguridad']:.3f}")
        print(f"  Clasificación: {pred['clasificacion']}")
        
        print(f"\nPROBABILIDADES POR CATEGORÍA:")
        for clase, prob in pred['probabilidades'].items():
            print(f"  {clase}: {prob*100:.1f}%")
        
        print(f"\nRECOMENDACIONES:")
        if pred['factor_seguridad'] < 1.0:
            print("  ⚠️  ALERTA: El talud es INESTABLE. Se requieren medidas correctivas urgentes.")
            print("  - Considerar reducción del ángulo del talud")
            print("  - Implementar sistema de drenaje")
            print("  - Evaluar refuerzo con geomallas o muros de contención")
        elif pred['factor_seguridad'] < 1.5:
            print("  ⚠️  PRECAUCIÓN: El talud está en condición MARGINAL.")
            print("  - Monitoreo continuo recomendado")
            print("  - Considerar mejoras en el drenaje")
            print("  - Evaluarmejoramiento del suelo")
        else:
            print("  ✓  El talud se encuentra en condición ESTABLE.")
            print("  - Mantener inspecciones periódicas")
            print("  - Monitorear cambios en condiciones hidrogeológicas")
        
        print("="*70)
        
        return pred

# Crear sistema
sistema = SistemaPredictionTaludes(
    model_dnn, model_class, scaler, label_encoder
)

# Ejemplo 1: Caso estable
print("\n" + "="*70)
print("EJEMPLO 1: ANÁLISIS DE TALUD TÍPICO")
print("="*70)

caso_ejemplo_1 = {
    'altura': 10.0,
    'angulo': 35.0,
    'cohesion': 25.0,
    'friccion': 30.0,
    'peso_unitario': 18.5,
    'nivel_freatico': 0.3,
    'sobrecarga': 15.0,
    'num_capas': 1
}

resultado_1 = sistema.generar_reporte(caso_ejemplo_1)

# Ejemplo 2: Caso crítico
print("\n" + "="*70)
print("EJEMPLO 2: TALUD EN CONDICIÓN CRÍTICA")
print("="*70)

caso_ejemplo_2 = {
    'altura': 15.0,
    'angulo': 55.0,
    'cohesion': 5.0,
    'friccion': 20.0,
    'peso_unitario': 19.0,
    'nivel_freatico': 0.8,
    'sobrecarga': 30.0,
    'num_capas': 2
}

resultado_2 = sistema.generar_reporte(caso_ejemplo_2)

# Análisis de Sensibilidad
# Análisis de sensibilidad para el ángulo
print("\n" + "="*70)
print("ANÁLISIS DE SENSIBILIDAD: VARIACIÓN DEL ÁNGULO")
print("="*70)

caso_base = caso_ejemplo_1.copy()
angulos = np.linspace(20, 60, 20)

df_sensibilidad_angulo = sistema.analizar_sensibilidad(
    caso_base, 'angulo', angulos
)

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de FS vs Ángulo
ax1.plot(df_sensibilidad_angulo['angulo'], 
         df_sensibilidad_angulo['fs'], 
         'b-', linewidth=2.5, marker='o', markersize=6)
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='FS=1.0 (Falla)')
ax1.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='FS=1.5 (Seguro)')
ax1.fill_between(df_sensibilidad_angulo['angulo'], 0, 1.0, alpha=0.2, color='red')
ax1.fill_between(df_sensibilidad_angulo['angulo'], 1.0, 1.5, alpha=0.2, color='orange')
ax1.fill_between(df_sensibilidad_angulo['angulo'], 1.5, 5, alpha=0.2, color='green')
ax1.set_xlabel('Ángulo del Talud (°)', fontsize=12)
ax1.set_ylabel('Factor de Seguridad', fontsize=12)
ax1.set_title('Sensibilidad del FS al Ángulo del Talud', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Análisis de sensibilidad para cohesión
cohesiones = np.linspace(0, 50, 20)
df_sensibilidad_cohesion = sistema.analizar_sensibilidad(
    caso_base, 'cohesion', cohesiones
)

ax2.plot(df_sensibilidad_cohesion['cohesion'], 
         df_sensibilidad_cohesion['fs'], 
         'g-', linewidth=2.5, marker='s', markersize=6)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='FS=1.0 (Falla)')
ax2.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='FS=1.5 (Seguro)')
ax2.fill_between(df_sensibilidad_cohesion['cohesion'], 0, 1.0, alpha=0.2, color='red')
ax2.fill_between(df_sensibilidad_cohesion['cohesion'], 1.0, 1.5, alpha=0.2, color='orange')
ax2.fill_between(df_sensibilidad_cohesion['cohesion'], 1.5, 5, alpha=0.2, color='green')
ax2.set_xlabel('Cohesión (kPa)', fontsize=12)
ax2.set_ylabel('Factor de Seguridad', fontsize=12)
ax2.set_title('Sensibilidad del FS a la Cohesión', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('analisis_sensibilidad.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nRango crítico de ángulo (FS < 1.5): {df_sensibilidad_angulo[df_sensibilidad_angulo['fs'] < 1.5]['angulo'].min():.1f}° - {df_sensibilidad_angulo[df_sensibilidad_angulo['fs'] < 1.5]['angulo'].max():.1f}°")
print(f"Cohesión mínima requerida para FS > 1.5: {df_sensibilidad_cohesion[df_sensibilidad_cohesion['fs'] >= 1.5]['cohesion'].min():.1f} kPa")

# Análisis Multivariable: Mapa de Contorno

# Crear mapa de contorno FS vs Ángulo y Cohesión
print("\nGenerando mapa de contorno...")

angulos_grid = np.linspace(20, 60, 30)
cohesiones_grid = np.linspace(0, 50, 30)
fs_grid = np.zeros((len(cohesiones_grid), len(angulos_grid)))

caso_grid = caso_base.copy()

for i, coh in enumerate(cohesiones_grid):
    for j, ang in enumerate(angulos_grid):
        caso_grid['cohesion'] = coh
        caso_grid['angulo'] = ang
        pred = sistema.predecir_estabilidad(**caso_grid)
        fs_grid[i, j] = pred['factor_seguridad']

# Visualización
plt.figure(figsize=(12, 9))
contour = plt.contourf(angulos_grid, cohesiones_grid, fs_grid, 
                       levels=20, cmap='RdYlGn', alpha=0.8)
plt.colorbar(contour, label='Factor de Seguridad')

# Líneas de contorno específicas
contour_lines = plt.contour(angulos_grid, cohesiones_grid, fs_grid,
                            levels=[1.0, 1.5, 2.0], colors='black', 
                            linewidths=2, linestyles=['--', '-', ':'])
plt.clabel(contour_lines, inline=True, fontsize=10, fmt='FS=%.1f')

plt.xlabel('Ángulo del Talud (°)', fontsize=12)
plt.ylabel('Cohesión (kPa)', fontsize=12)
plt.title('Mapa de Factor de Seguridad\n(Altura=10m, Fricción=30°, NF=30%)', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, color='white', linewidth=0.5)

plt.tight_layout()
plt.savefig('mapa_contorno_fs.png', dpi=300, bbox_inches='tight')
plt.show()

# Guardar Modelos y Escalador

# Guardar todos los componentes del sistema
model_dnn.save('modelo_fs_dnn.h5')
model_class.save('modelo_clasificacion_dnn.h5')

import joblib
joblib.dump(scaler, 'scaler_taludes.pkl')
joblib.dump(label_encoder, 'label_encoder_estabilidad.pkl')

print("\n✓ Modelos y componentes guardados exitosamente:")
print("  - modelo_fs_dnn.h5")
print("  - modelo_clasificacion_dnn.h5")
print("  - scaler_taludes.pkl")
print("  - label_encoder_estabilidad.pkl")

#5. Explicabilidad con SHAP
# Ejemplo de implementación futura con SHAP
"""
import shap

# Crear explicador
explainer = shap.DeepExplainer(model_dnn, X_train_scaled[:100])

# Calcular valores SHAP para caso específico
shap_values = explainer.shap_values(X_test_scaled[0:1])

# Visualizar importancia de características
shap.waterfall_plot(shap_values[0])
"""