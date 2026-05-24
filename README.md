# Proyecto de clasificacion de prestamos

Este proyecto aplica tecnicas de aprendizaje automatico a un problema de
clasificacion binaria sobre solicitudes de prestamos. El objetivo es predecir
la variable `loan_status` a partir de las caracteristicas del solicitante y del
prestamo.

El punto de entrada principal del proyecto es:

```bash
python Proyecto/main.py
```

## Estructura del proyecto

```text
Proyecto/
├── data/
│   └── loan_data.csv
├── main.py
├── src/
│   ├── eda.py
│   ├── evaluation.py
│   ├── learning_curves.py
│   ├── preprocess.py
│   ├── validation.py
│   └── models/
│       ├── logistic_regression.py
│       ├── neural_network.py
│       └── deep_neural_network.py
├── results/
│   └── figures/
└── report/
```

Los ficheros `README.md` y `requirements.txt` se encuentran en la raiz del
repositorio, fuera de la carpeta `Proyecto/`.

## Modelos implementados

El proyecto compara tres modelos:

- Regresion logistica implementada manualmente con NumPy.
- Red neuronal implementada manualmente con NumPy.
- Red neuronal profunda implementada con PyTorch.

Ademas, se incluyen:

- Analisis exploratorio de datos.
- Analisis sencillo de outliers.
- Preprocesado de variables categoricas y normalizacion.
- Limpieza conservadora de registros claramente sospechosos.
- Division train/test estratificada.
- Validacion cruzada estratificada.
- Evaluacion con accuracy, precision, recall y F1.
- Comparacion con y sin balanceo de clases.
- Curvas de entrenamiento, learning curves y validation curves.

## Instalacion del entorno

Se recomienda crear un entorno limpio de Python antes de ejecutar el proyecto.

Con conda:

```bash
conda create -n loan_project python=3.11
conda activate loan_project
```

O con venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

Despues, instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Dependencias principales

Las librerias necesarias para reproducir los resultados son:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `imbalanced-learn`
- `torch`

`torch` se utiliza para la red neuronal profunda. En Mac con Apple Silicon, el
codigo intenta usar aceleracion `mps` si esta disponible; si no, se ejecuta en
CPU.

## Ejecucion

Desde la carpeta del proyecto:

```bash
python Proyecto/main.py
```

Durante la ejecucion, el programa pregunta primero si se quiere ejecutar el
analisis exploratorio de datos:

```text
Quieres ejecutar el EDA? (s/n):
```

Si se responde `s`, se muestra el analisis exploratorio, el analisis de la
variable de defaults previos y el resumen de outliers. Si se responde `n`, el
programa continua directamente.

Despues del split, el programa pregunta si se quieren generar las curvas:

```text
Quieres generar las curvas? (s/n):
```

Si se responde `s`, se generan las curvas de entrenamiento, learning curves y
validation curve. Si se responde `n`, se omiten para que la ejecucion sea mas
rapida. Omitir las curvas no afecta a la validacion cruzada ni a la evaluacion
final en test.

El programa ejecuta el flujo principal en este orden:

1. Carga del dataset.
2. Analisis exploratorio opcional.
3. Limpieza conservadora de outliers y division en train/test.
4. Generacion opcional de curvas.
5. Validacion cruzada sin balanceo.
6. Evaluacion final en test sin balanceo.
7. Validacion cruzada con balanceo.
8. Evaluacion final en test con balanceo.

## Limpieza de outliers

El dataset original contiene 45.000 filas. Antes de separar train y test se
eliminan solo registros claramente sospechosos mediante reglas sencillas:

- `person_age <= 100`
- `person_emp_exp <= 60`
- `person_income <= 1_000_000`

Esta limpieza elimina 31 filas, quedando 44.969 ejemplos. La division
estratificada resultante es:

- Train: 35.975 ejemplos.
- Test: 8.994 ejemplos.

La limpieza es conservadora: no se eliminan todos los outliers estadisticos,
solo valores extremos poco plausibles.

## Reproducibilidad

El dataset necesario se incluye en:

```text
data/loan_data.csv
```

Para reproducir los resultados descritos en la memoria, ejecutar el programa
principal desde `Proyecto/main.py`. Para reproducir tambien las graficas,
responder `s` cuando el programa pregunte si se quieren generar las curvas.
Algunas partes, como el entrenamiento de redes neuronales y la generacion de
curvas, pueden tardar varios minutos dependiendo del ordenador.
