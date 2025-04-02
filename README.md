# Interactive Dashboard for Meteorite Analysis

This project presents an **interactive dashboard** developed in a Jupyter Notebook to analyze a meteorite dataset. The dashboard allows you to explore temporal trends, mass distribution (with logarithmic scaling), primary classifications, an interactive global map, and various descriptive statistics and correlation metrics.

The dataset was obtained from [Kaggle](https://www.kaggle.com/) and is expected to be located in the `data/` folder under the filename `Meteorite_Landings.csv`.

---

## Project Information

- **Presented by:**  
  - Agustín Villarreal Carrillo  
  - Víctor Manuel Mariscal Cervantes

- **Course:** Topics in Industry 1  
- **Master's in Applied Computing**  
- **Languages and Tools Used:**  
  - [Jupyter Notebook](https://jupyter.org/)  
  - [Python 3.9.12](https://www.python.org/)

---

## Prerequisites

- **Python:** Version 3.9.12  
- **Jupyter Notebook or JupyterLab**

---

## Dependencies

The project relies on the following Python libraries:
- `pandas`
- `numpy`
- `panel`
- `hvplot`
- `holoviews`
- `colorcet`
- `folium`
- `plotly`

Make sure these dependencies are installed so that every cell in the notebook runs correctly.

---

## Installation and Virtual Environment Setup

### 1. Clone the Repository

Open your terminal and execute:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

### 2. Create and Activate the Virtual Environment

Create a virtual environment using Python 3.9.12 with the following command:

```bash
python3.9 -m venv venv
```

Activate the virtual environment with:

- **On Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies

Ensure you have a `requirements.txt` file containing all required libraries. Then install them with:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

There are two options to run the interactive dashboard:

### Option 1: Run Directly in Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the notebook file (e.g., `dashboard_meteorites.ipynb`).
3. Execute the cells sequentially to load the data, apply filters, and generate interactive visualizations.

### Option 2: Run on an Interactive Server (Recommended)

For a more visually appealing and refined experience, it is recommended to run the dashboard on an interactive server using Panel. To do so, execute:

```bash
panel serve dashboard_meteorites.ipynb --show
```

This command will launch an interactive server where you can explore the dashboard in real-time, with all controls and visualizations updating dynamically.

---

## Language and Technology Badges

- **Jupyter Notebook**  
  ![Jupyter](https://jupyter.org/assets/main-logo.svg)

- **Python 3.9.12**  
  ![Python](https://www.python.org/static/community_logos/python-logo.png)

---

## Additional Notes

- **Dataset:**  
  The `Meteorite_Landings.csv` file must be located in the `data/` folder.

- **Source Code:**  
  Auxiliary modules and custom functions are located in the `src/` folder.

- **Contact:**  
  If you have any questions or encounter issues, please reach out to the project presenters.

---

*This project was developed for the Topics in Industry 1 course in the Master's in Applied Computing program by Agustín Villarreal Carrillo and Víctor Manuel Mariscal Cervantes.*
