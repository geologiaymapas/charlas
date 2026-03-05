# Introducción

Como científico o analista de datos, sabe que comprender sus datos es fundamental para el éxito de cualquier proyecto. Antes de poder crear modelos, crear paneles o generar información, necesita saber con qué está trabajando. Sin embargo, el análisis exploratorio de datos (EDA) es extremadamente repetitivo y lento.

Para cada nuevo conjunto de datos, probablemente escriba prácticamente el mismo código para comprobar los tipos de datos, calcular estadísticas, representar gráficamente las distribuciones y más. Necesita enfoques sistemáticos y automatizados para comprender sus datos de forma rápida y exhaustiva. Este artículo abarca cinco scripts de Python diseñados para automatizar los aspectos más importantes y que más tiempo requieren en la exploración de datos.

## 1. Perfilado de datos

- Identificación del punto crítico
Al abrir un conjunto de datos por primera vez, necesita comprender sus características básicas. Escriba código para comprobar los tipos de datos, contar valores únicos, identificar datos faltantes, calcular el uso de memoria y obtener estadísticas resumidas. Esto se hace para cada columna, generando el mismo código repetitivo para cada nuevo conjunto de datos. Este perfilado inicial por sí solo puede tardar una hora o más en conjuntos de datos complejos.

- Revisión de la función del script
Genera automáticamente un perfil completo de su conjunto de datos, incluyendo tipos de datos, patrones de valores faltantes, análisis de cardinalidad, uso de memoria y resúmenes estadísticos para todas las columnas. Detecta posibles problemas como variables categóricas de alta cardinalidad, columnas constantes y discrepancias en los tipos de datos. Genera un informe estructurado que le ofrece una visión completa de sus datos en segundos.

- Explicación de su funcionamiento
El script itera por cada columna, determina su tipo y calcula las estadísticas relevantes:

Para las columnas numéricas, calcula la media, la mediana, la desviación estándar, los cuartiles, la asimetría y la curtosis.
Para las columnas categóricas, identifica valores únicos, moda y distribuciones de frecuencia.
Marca posibles problemas de calidad de los datos, como columnas con más del 50 % de valores faltantes, columnas categóricas con demasiados valores únicos y columnas con varianza cero. Todos los resultados se compilan en un marco de datos fácil de leer.

[Script 1](script1.md)

## 2. Análisis y visualización de distribuciones

- Identificación del punto crítico
Comprender la distribución de los datos es fundamental para elegir las transformaciones y los modelos adecuados. Es necesario trazar histogramas, diagramas de caja y curvas de densidad para las características numéricas, y gráficos de barras para las categóricas. Generar estas visualizaciones manualmente implica escribir código para cada variable, ajustar los diseños y gestionar múltiples ventanas de figuras. Para conjuntos de datos con docenas de características, esto resulta engorroso.

- Revisión de la función del script
Genera visualizaciones de distribución completas para todas las características del conjunto de datos. Crea histogramas con estimaciones de densidad kernel para las características numéricas, diagramas de caja para mostrar valores atípicos, gráficos de barras para las características categóricas y diagramas Q-Q para evaluar la normalidad. Detecta y resalta distribuciones asimétricas, patrones multimodales y posibles valores atípicos. Organiza todos los gráficos en una cuadrícula clara con escalado automático.

- Explicación de su funcionamiento
El script separa las columnas numéricas y categóricas y genera visualizaciones adecuadas para cada tipo:

Para las características numéricas, crea subgráficos que muestran histogramas con curvas de estimación de densidad de kernel (KDE) superpuestas, anotadas con valores de asimetría y curtosis.
Para las características categóricas, genera gráficos de barras ordenados que muestran las frecuencias de los valores.
El script determina automáticamente los tamaños óptimos de los intervalos, gestiona los valores atípicos y utiliza pruebas estadísticas para identificar las distribuciones que se desvían significativamente de la normalidad. Todas las visualizaciones se generan con un estilo consistente y se pueden exportar según sea necesario.

[Script 2](script2.md)

## 3. Exploración de correlaciones y relaciones

- Identificación del punto crítico
Comprender las relaciones entre variables es esencial, pero tedioso. Es necesario calcular matrices de correlación, crear gráficos de dispersión para pares prometedores, identificar problemas de multicolinealidad y detectar relaciones no lineales. Hacer esto manualmente requiere generar docenas de gráficos, calcular diversos coeficientes de correlación como Pearson, Spearman y Kendall, e intentar detectar patrones en los mapas de calor de correlación. El proceso es lento y, a menudo, se pasan por alto relaciones importantes.

- Revisión de la función del script
Analiza las relaciones entre todas las variables del conjunto de datos. Genera matrices de correlación con múltiples métodos, crea gráficos de dispersión para pares altamente correlacionados, detecta problemas de multicolinealidad para el modelado de regresión e identifica relaciones no lineales que la correlación lineal podría pasar por alto. Crea visualizaciones que permiten profundizar en relaciones específicas y señala posibles problemas, como correlaciones perfectas o características redundantes.

- Explicación de su funcionamiento
El script calcula matrices de correlación utilizando correlaciones de Pearson, Spearman y Kendall para capturar diferentes tipos de relaciones. Genera un mapa de calor anotado que destaca las correlaciones fuertes.