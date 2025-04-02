# Dashboard Interactivo para el Análisis de Meteoritos

Este proyecto presenta un **dashboard interactivo** desarrollado en Jupyter Notebook para analizar el dataset de meteoritos. El dashboard permite la visualización de tendencias temporales, distribución de masas (con escala logarítmica), clasificaciones principales, un mapa interactivo global y diversas métricas descriptivas y de correlación.

El dataset utilizado fue obtenido desde [Kaggle](https://www.kaggle.com/) y se espera que se encuentre en la carpeta `data/` bajo el nombre `Meteorite_Landings.csv`.

---

## Información del Proyecto

- **Presentado por:**  
  - Agustín Villarreal Carrillo  
  - Víctor Manuel Mariscal Cervantes

- **Asignatura:** Tópicos de Industria 1  
- **Maestría en Cómputo Aplicado**  
- **Lenguajes Utilizados:**  
  - [Jupyter Notebook](https://jupyter.org/)  
  - [Python 3.9.12](https://www.python.org/)

---

## Requisitos Previos

- **Python:** Versión 3.9.12  
- **Jupyter Notebook o JupyterLab**

---

## Dependencias

El proyecto utiliza las siguientes librerías:
- `pandas`
- `numpy`
- `panel`
- `hvplot`
- `holoviews`
- `colorcet`
- `folium`
- `plotly`

*Asegúrate de tener estas dependencias instaladas para que todas las celdas del notebook se ejecuten correctamente.*

---

## Instalación y Configuración del Entorno Virtual

### 1. Clonar el Repositorio

Abre tu terminal y ejecuta:

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Crear e Iniciar el Entorno Virtual (venv)

Utiliza el siguiente comando para crear un entorno virtual con Python 3.9.12:

```bash
python3.9 -m venv venv
```

Para activarlo, usa:

- **En Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```
- **En Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Instalar las Dependencias

Asegúrate de contar con un archivo `requirements.txt` que contenga todas las librerías necesarias. Luego, ejecuta:

```bash
pip install -r requirements.txt
```

---

## Ejecución del Notebook

Existen dos opciones para ejecutar el dashboard:

### Opción 1: Ejecutar Directamente en Jupyter Notebook

1. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Abre el archivo `dashboard_meteoritos.ipynb`.
3. Ejecuta las celdas secuencialmente para cargar datos, aplicar filtros y generar las visualizaciones interactivas.

### Opción 2: Ejecutar en un Servidor Interactivo (Recomendado)

Para obtener una experiencia visual más atractiva y estética, se recomienda ejecutar el dashboard en un servidor interactivo con Panel. Para ello, utiliza:

```bash
panel serve dashboard_meteoritos.ipynb --show
```

Este comando lanzará un servidor interactivo en el que se podrá explorar el dashboard en tiempo real, con todos los controles y visualizaciones actualizados dinámicamente.

---

## Decoradores de los Lenguajes Utilizados

- **Jupyter Notebook**  
  ![Jupyter](https://jupyter.org/assets/main-logo.svg)

- **Python 3.9.12**  
  ![Python](https://www.python.org/static/community_logos/python-logo.png)

---

## Notas Adicionales

- **Dataset:**  
  El archivo `Meteorite_Landings.csv` debe estar ubicado en la carpeta `data/`.

- **Código Fuente:**  
  Los módulos auxiliares y funciones personalizadas se encuentran en la carpeta `src/`.

- **Contacto:**  
  Si tienes dudas o encuentras inconvenientes, por favor contacta a los responsables del proyecto.

---

*Proyecto presentado para la asignatura de Tópicos de Industria 1 en la Maestría en Cómputo Aplicado por Agustín Villarreal Carrillo y Víctor Manuel Mariscal Cervantes.*
