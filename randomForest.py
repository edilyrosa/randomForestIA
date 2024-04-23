#Importar bibliotecas
from sklearn.ensemble import RandomForestClassifier #function
from sklearn.model_selection import train_test_split #function to split the data into -> train & test
from sklearn.metrics import accuracy_score #Assess model's accuracy
import numpy as np # Arrays.

#!Generacion aleatoria de datos de e.g, son 3 arrays[100] dentro de los rangos o secuencia dados
#?Para entrenar, valores basados en criterios no aleatorios.
#la aleatoriedad para seleccinar las muestras bootstrapping & caracteristicas.
np.random.seed(42)
horas_estudio = np.random.randint(1, 10, 100) # 1 a 9 
calificacion_previa = np.random.randint(60, 100, 100) # 60 a 99 
aprobado = np.random.choice([0, 1], 100) # 0 || 1 


#! Creacion la matriz dataset: de 3 columnas (las variables).
data = np.column_stack((horas_estudio, calificacion_previa, aprobado))

#! Dividir datos variables
X = data[:, :2] #características 
y = data[:, 2] #etiquetas

#! Dividir conjunto de entrenamiento y prueba para "X" &"y"
#?Evaluar el modelo en datos no vistos
#?Si solo se entrena, ocurriria sobreajuste, el modelo memoriza los datos en lugar de aprender patrones generales.
# test_size=0.2 -> 20% de los datos como set de prueba, el 80% de entrenamiento.
#random_state=42 -> la división sea reproducible, si vuelves a ejecutar el código con el mismo valor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#! Crear un clasificador de Random Forest
    #10 arboles y garantizar la reproducibilidad
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

#! Entrenar al clasificador con fit(dataset-entremamiento). 
rf_classifier.fit(X_train, y_train)

#!Entrenado el clasificador, hacemos predicciones -> predict(set de testing)
#la predicion nos devolvera "y" de cada instancia en el set de test.
#e.g -> predictions[0] = 1, el modelo predice que el primer estudiante en el set de test está "aprobado".
predictions = rf_classifier.predict(X_test)
#?comparemos predicciones con las etiquetas desconocidas (y_test) para evaluar el rendimiento del modelo.

# Evaluar precisión
accuracy = accuracy_score(y_test, predictions)
print(f'Precisión del modelo: {accuracy}')
print(f'Prdictions: {predictions}')
print(f'Test_y    : {y_test}')
