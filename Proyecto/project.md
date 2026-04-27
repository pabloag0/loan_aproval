# Descripcion del proyecto

Este proyecto tiene como objetivo analizar un conjunto de datos de prestamos y preparar un flujo basico de aprendizaje automatico para predecir el estado de un prestamo (`loan_status`).

El programa principal carga los datos, permite realizar un analisis exploratorio, aplica un preprocesado para convertir las variables categoricas en variables numericas y separa el conjunto de datos en entrenamiento y prueba.

La idea general del proyecto es comparar distintos modelos de clasificacion, como regresion logistica y redes neuronales, para estudiar que variables del cliente y del prestamo ayudan mejor a predecir si un prestamo saldra bien o no.

Ademas, el proyecto guarda resultados intermedios, como el dataset preprocesado y distintas figuras del analisis exploratorio, de forma que el flujo completo quede organizado y sea facil de reproducir desde `main.py`.
