# 5 Scripts de Python Útiles para la Exploración de Datos

Automatiza el análisis exploratorio de datos con estos 5 completos scripts de Python que te ahorran horas de trabajo manual.

## Scripts Incluidos

1. **Perfilador de Datos Integral** - Perfilado completo de conjuntos de datos con estadísticas y controles de calidad
2. **Analizador y Visualizador de Distribuciones** - Análisis de distribuciones con gráficos y pruebas de normalidad
3. **Explorador de Correlaciones y Relaciones** - Análisis de correlación multimétodo y detección de VIF
4. **Conjunto de Detección y Análisis de Valores Atípicos** - Múltiples métodos de detección de valores atípicos con consenso
5. **Analizador de Patrones de Datos Faltantes** - Detección de patrones faltantes y clasificación de mecanismos

---

## 📚 Dependencias

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

---

## Ejemplos de inicio rápido

### 1. Perfilador de datos completo

```python
from data_profiler import DataProfiler
import pandas as pd

df = pd.read_csv('data.csv')

# Crear perfilador
profiler = DataProfiler(df, high_cardinality_threshold=0.5)

# Imprimir resumen en la consola
profiler.print_summary()

# O obtenga un informe detallado
report = profiler.generate_full_profile()
print(report['overview'])
print(report['numeric_profiles'])
print(report['categorical_profiles'])
print(report['data_quality_issues'])

# Save reports
report['numeric_profiles'].to_csv('numeric_profile.csv', index=False)
report['data_quality_issues'].to_csv('issues.csv', index=False)
```

### 2. Analizador y visualizador de distribución

```python
from distribution_analyzer import DistributionAnalyzer

df = pd.read_csv('data.csv')
analyzer = DistributionAnalyzer(df)

# Generar informe de distribución
report = analyzer.generate_distribution_report()
print(report)

# Identificar columnas muy sesgadas
skewed = report[abs(report['skewness']) > 2]
print(f"Highly skewed columns: {skewed['column'].tolist()}")

# Visualizar distribuciones
analyzer.plot_numeric_distributions(max_cols=10)
analyzer.plot_boxplots()
analyzer.plot_qq_plots()
analyzer.plot_categorical_distributions()
```

### 3. Explorador de correlaciones y relaciones

```python
from correlation_explorer import CorrelationExplorer

df = pd.read_csv('data.csv')
explorer = CorrelationExplorer(df)

# Encuentra correlaciones altas
high_corr = explorer.find_high_correlations(threshold=0.7, method='pearson')
print(high_corr)

# Comprobación de multicolinealidad
vif = explorer.calculate_vif()
problematic = vif[vif['vif'] > 10]
print(f"Features with high multicollinearity:\n{problematic}")

# Información mutua con el objetivo
if 'target' in df.columns:
    mi_scores = explorer.mutual_information_analysis('target')
    print(f"Top features:\n{mi_scores.head(10)}")

# Visualizar
explorer.plot_correlation_heatmap(method='pearson')
explorer.plot_correlation_comparison()
explorer.plot_scatter_matrix(max_cols=5)
explorer.plot_top_correlations(n_pairs=10)
```

### 4. Suite de detección y análisis de valores atípicos

```python
from outlier_suite import OutlierSuite

df = pd.read_csv('data.csv')
suite = OutlierSuite(df)

# Comparar métodos en todas las columnas
summary = suite.compare_methods_all_columns()
print(summary)

# Analizar una columna específica
suite.plot_outlier_comparison('column_name')

# Detectar valores atípicos multivariados
iso_outliers = suite.detect_isolation_forest_outliers(contamination=0.1)
print(f"Found {iso_outliers.sum()} multivariate outliers")

suite.plot_multivariate_outliers(['feature1', 'feature2'])

# Analizar el impacto de los valores atípicos
impact = suite.analyze_outlier_impact('column_name')
print(impact)
```

### 5. Analizador de patrones de datos faltantes

```python
from missing_data_analyzer import MissingDataAnalyzer

df = pd.read_csv('data.csv')
analyzer = MissingDataAnalyzer(df)

# Generar informe completo
report = analyzer.generate_full_report()

print("Missing Value Summary:")
print(report['summary'])

print("\nMissingness Patterns (co-occurrence):")
print(report['patterns'])

print("\nMissingness Classifications:")
print(report['classifications'])

print("\nImputation Recommendations:")
print(report['recommendations'])

# Visualizar patrones
analyzer.plot_missing_bar()
analyzer.plot_missing_heatmap(max_cols=30)
analyzer.plot_missing_correlation()

# Clasificar columna específica
classification = analyzer.classify_missingness_type('column_name')
recommendation = analyzer.recommend_strategy('column_name')
print(f"Missingness type: {classification['missingness_type']}")
print(f"Recommendation: {recommendation}")
```

---
