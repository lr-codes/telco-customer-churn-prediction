# Predicción de Fuga de Clientes con Random Forest

Este proyecto implementa un famoso modelo de Machine Learning capaz de identificar a los clientes de una empresa de telecomunicaciones en riesgo de abandonarla (*Customer Churn*).

El objetivo principal es proporcionar una herramienta auditable que maximize la detección de bajas antes de que se consoliden, optimizando así las campañas de fidelización y presupuestos.

## Indicadores Clave
En este tipo de problemas, dejar escapar a un Falso Negativo es significativamente más costoso que lanzar una oferta preventiva a un cliente leal. Por ello el proyecto se centra en optimizar el **Recall** por encima de la precisión global.

* **Modelo Base:** Detección del **61%** de clientes en riesgo de fuga.
* **Modelo Optimizado:** Detección del **82%** de fugas.
* **Resultado:** Incremento neto del **21%** en la capacidadde captura de desertores. 

## Arquitectura del proyecto
El conjunto de datos original (*Telco Customer Churn* de IBM) presentaba ciertos desafíos:
1. **Tratamiento de errores ocultos:** Detección y corrección de espacios en blanco en la columna `TotalCharges`, forzando la conversión a tipo numérico.
2. **Codificación Categóriga:** Se aplica *One-Hot Encoding* (`pd.get_dummies`) para transformar variables de texto complejas en un sistema matricial numérico limpio de 31 variables predictoras.
3. **Gestión del Desbalanceo:** Dado que la clase de clientes leales superaba ampliamente a la de bajas, se implementó una estrategia de penalización mediante `class_weight='balanced'` para evitar que el algoritmo ignorara a la minoría. 

## Optimización de Hiperparámetros (GridSearchCV)

Para conseguir el máximo rendimiento del algoritmo `RandomForestClassifier`, se ejecutó una búsqueda en cuadrícula automatizada evaluando múltiples combinaciones mediante Validación Cruzada, en este caso de 5 cortes.

La combinación ganadora que logró el 82% de Recall fue:
* `max_depth`: 5 *(Evita el sobreajuste, forzando al modelo a aprender patrones generales)*
* `min_samples_leaf`: 4
* `n_estimators`: 200

## Factores de Riesgo Más Críticos (Feature Importance)

![Top 10 Características](importancias.png)

El modelo de Bosque Aleatorio nos permite auditar su comportamiento. Tras analizar el peso de las decisiones de los 200 árboles combinados, los 3 factores que acumulan más de la mitad de la relevancia absoluta para que un cliente se marche son:

1. **Tenure (Antigüedad del cliente):** El tiempo histórico de permanencia en la empresa es el indicador número uno de lealtad.
2. **Contract_Two year (Contrato de dos años):** El nivel de compromiso a largo plazo es una barrera clave contra la fuga.
3. **InternetService_Fiber optic (Servicio de Fibra Óptica):** El tipo de tecnología principal del usuario influye directamente en su probabilidad de abandonar la compañía.

*(Nota de Negocio: Gracias a la optimización del modelo, descubrimos que la fidelidad pura (antigüedad y blindaje de contratos) y la calidad del servicio técnico (Fibra) pesan incluso más que el precio total acumulado, indicando que el departamento de retención debe enfocarse en incentivar contratos largos y asegurar la estabilidad de la red de fibra óptica).*