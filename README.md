# Sistemas de Recomendación: Implementación de MF, NCF y NGCF

## Descripción

Este proyecto implementa y extiende varios modelos de sistemas de recomendación, incluyendo:

- **MF** (Matrix Factorization)
- **NCF** (Neural Collaborative Filtering)
- **NGCF** (Neural Graph Collaborative Filtering)

Se ha tomado como base la implementación de NGCF publicada por Xiang Wang et al. en el paper:

> Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering. In SIGIR'19, Paris, France. [Paper en ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) o [Paper en arXiv](https://arxiv.org/abs/1905.08108).

La implementación original ha sido adaptada y ampliada para ajustarse a las necesidades del proyecto. Se han realizado mejoras en la configuración, la integración con datasets adicionales y la flexibilidad de los hiperparámetros para experimentar con diferentes arquitecturas y escenarios. Además se han implemnetado las arquitecturas de MF y NCF

## Autor

Mario González Monge

## Estructura del Proyecto

- `src/Data/` → Datos procesados, `movielens_1M`, incluyendo archivos como:
  - `train.txt`, `test.txt`: División de datos para entrenamiento y prueba.
  - `user_list.txt`, `item_list.txt`: Mapeo entre IDs originales y remapeados.
- `src/MF/` → Implementación del modelo **Matrix Factorization (MF)**.
- `src/NCF/` → Implementación de modelos basados en **Neural Collaborative Filtering (NCF)**:
  - **GMF**: Generalized Matrix Factorization.
  - **MLP**: Multi-Layer Perceptron.
  - **NeuMF**: Neural Matrix Factorization (combinación de GMF y MLP).
- `src/NGCF/` → Implementación de **Neural Graph Collaborative Filtering (NGCF)**.
- `src/utility/` → Utilidades comunes, como carga de datos y preprocesamiento.
- `evaluation/` → Evaluación y comparación de los resultados de los modelos.
- `output/` → Resultados generados durante la ejecución de los modelos.
- `README.md` → Este archivo.

## Uso y Comentarios

### Requisitos

El archivo `requirements.txt` contiene las librerías necesarias para ejecutar los modelos. Puedes instalar las dependencias con:

pip install -r requirements.txt


### Ejecución de los Modelos

Se incluyen configuraciones predefinidas en `launch.json` para ejecutar y depurar los distintos modelos desde **Visual Studio Code**. Esto permite una gestión flexible de los hiperparámetros y facilita el desarrollo.

#### Abrir el proyecto en VSCode

- Asegúrate de tener configurado `launch.json` dentro de `.vscode/`.

#### Seleccionar el entorno virtual de Python

- Activa tu entorno virtual y asegúrate de que VSCode lo reconozca.

#### Configurar hiperparámetros

- En el archivo `launch.json`, puedes modificar los hiperparámetros, capas, tasas de aprendizaje y otras opciones para cada modelo.

#### Ejecutar los modelos

1. Dirígete a la pestaña de **Run and Debug** en VSCode.
2. Selecciona la configuración deseada, como `Debug NGCF` o `Debug MF`.

#### Configuraciones Disponibles en `launch.json`

- **Debug NGCF**: Ejecuta el modelo NGCF con hiperparámetros personalizables.
- **Debug MF**: Ejecuta el modelo de Matrix Factorization.
- **Debug GMF**: Ejecuta el modelo Generalized Matrix Factorization.
- **Debug MLP**: Ejecuta el modelo Multi-Layer Perceptron.
- **Debug NeuMF**: Ejecuta la combinación de GMF y MLP (NeuMF).
