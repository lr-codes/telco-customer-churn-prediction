import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
datos = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1º/ El ID es engañoso, pues son letras aleatorias, lo borramos
datos = datos.drop('customerID', axis=1)

# 2º/ TotalCharges es un número, pero al tener la posibilidad de ser "", se trata como string
# coerce convierte los valores "" en nulos (NaN), y rellenamos los nulos con 0
datos['TotalCharges'] = pd.to_numeric(datos['TotalCharges'], errors='coerce')
datos['TotalCharges'] = datos['TotalCharges'].fillna(0)

# 3º/ Transformamos el Churn
datos['Churn'] = datos['Churn'].map({'Yes': 1, 'No': 0})

# 4º/ Transformamos el resto de columnas
# Get dummies convierte el resto de columnas automáticamente en columnas con valores 0 y 1
datos = pd.get_dummies(datos, drop_first=True)


# Separación de variables 
X = datos.drop('Churn', axis=1)     
y = datos['Churn']

X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=25) 


# ==========================================
# 2. MODELO BASE (Referencia inicial)
# ==========================================


print("\n" + "="*50)
print(" ENTRENANDO MODELO BASE")
print("="*50)
# Creación del modelo, teniendo en cuenta el peso de las clases
modelo_rfc = RandomForestClassifier(n_estimators = 100, random_state=25, class_weight='balanced')
modelo_rfc.fit(X_training, y_training)
predicciones = modelo_rfc.predict(X_test)

# Clasificación
print("--- INFORME MODELO BASE ---")
print(classification_report(y_test, predicciones))


# ==========================================
# 3. OPTIMIZACIÓN DEL MODELO 
# ==========================================

print("\n" + "="*50)
print(" ENTRENANDO MODELO OPTIMIZADO")
print("="*50)

parametros = {
    'n_estimators':[100, 200, 300],     # Cantidad de arboles
    'max_depth':[5, 10, None],          # Profundidad máxima
    'min_samples_leaf':[1, 2, 4]        # Tam minimo para tomar una decisión
}


# Creación del buscador
# cv=5 (validacion cruzada de 5 cortes)
# n_jobs=1 (todos los núcleos del procesador)
# scoring='recall' (objetivo principal es max la detección de fugas)
buscador = GridSearchCV(estimator=modelo_rfc, param_grid=parametros, cv=5, n_jobs=-1, scoring='recall')
print("buscando el Bosque Aleatorio adecuado.")
buscador.fit(X_training, y_training)

# Sacamos el modelo ganador
modelo_perf = buscador.best_estimator_
print("Búsqueda finalizada, la mejor combinación de hiperparámetros es: ")
print(buscador.best_params_)

# --- EVALUACION MODELO MULTIPLE ---
predic_optimas = modelo_perf.predict(X_test)
print("\n--- INFORME DEL MODELO OPTIMIZADO ---")
print(classification_report(y_test, predic_optimas))


# ==========================================
# 4. EXTRACCIÓN DE IMPORTANCIAS Y GRÁFICA
# ==========================================

print("\n" + "="*50)
print(" GENERANDO GRÁFICA DE IMPORTANCIAS")
print("="*50)

# Se pegan ambos, con el indice el nombre de la columna y de contenido la importancia
importancia_emparejada = pd.Series(modelo_perf.feature_importances_, index=X_training.columns)
importancia_ordenada = importancia_emparejada.sort_values(ascending=False)

# Generación de la gráfica
plt.figure(figsize=(10,6))

# Dibujamos un gráfico de barras horizontales
sns.barplot(x=importancia_ordenada.head(10).values, 
            y=importancia_ordenada.head(10).index, 
            hue=importancia_ordenada.head(10).index,
            palette="viridis",
            legend=False)
plt.title("Top 10 Factores de Riesgo en la Fuga de Clientes", fontsize=14, pad=15)
plt.xlabel("Nivel de Importancia (0 a 1)", fontsize=12)
plt.ylabel("Variables", fontsize=12)
plt.tight_layout() # Ajusta los márgenes para que no se corte el texto
plt.show()
