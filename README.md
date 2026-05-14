# palomares-datacleaner

Script personalizado hecho para facilitar el tratamiento y la limpieza de DataFrames de Pandas. Permite:

- Visualizar estado general del DataFrame
- Tratamiento de datos nulos
- Tratamiento de datos duplicados
- Modificación de tipos de datos
- Tratamiento de valores atípicos

## Estructura del directorio

```
│   # script principal
├── datacleaner
│   ├── __init__.py
│   └── cleaner.py
│
│   # notebook con ejemplo de uso
├── notebook
│   ├── notebook.ipynb
│   └── ventas.csv
│
│   # directorio para pruebas con pytest
├── tests
│   ├── __init__.py
│   ├── test_cast_columns.py
│   ├── test_clean_dups.py
│   ├── test_clean_nulls.py
│   ├── test_clean_outliers.py
│   ├── test_display_outliers.py
│   ├── test_evaluate_null_strategy.py
│   ├── test_get_bounds.py
│   ├── test_handle_outliers_col.py
│   └── test_iter_outlier_columns.py
│
├── .gitignore            # git-ignore
├── .python-version       # versión de python utilizada
├── pyptoject.toml        # control de versiones y dependencias
├── README.md             # este archivo
└── requirements-dev.txt  # dependencias para desarrollo
```

## Instalación

**Requisitos:**

- Python 3.12.3 o superior

**Instalación desde GitHub:**

```bash
pip install git+https://github.com/NicolasPalomaresDS/palomares-datacleaner.git
```

**Para instalar una versión específica:**

```bash
pip install git+https://github.com/NicolasPalomaresDS/palomares-datacleaner.git@v1.0.0
```

**Uso rápido:**

```python
import pandas as pd
from datacleaner import DataCleaner

df = pd.read_csv("datos.csv")
cleaner = DataCleaner(df)

# Resumen del DataFrame
cleaner.get_summary()

# Limpieza de nulos
cleaner.clean_nulls(columns=["precio", "edad"], strategy="median") # Imputar valor
cleaner.clean_nulls(columns=["nombre"], delete=True) # Eliminarlos

# Compara filas e índices duplicados
cleaner.clean_dups_by_id()

# Considera solo las columnas especificadas
cleaner.clean_dups_by_subset(subset=["nombre", "fecha", "precio"])

# Considera todas las columnas
cleaner.clean_dups_by_subset()

# Conversión de tipos
cleaner.cast_columns(columns=["fecha"], to="datetime")
cleaner.cast_columns(columns=["precio", "edad"], to="float64")

# Detección y limpieza de outliers
cleaner.display_outliers(columns=["precio", "edad"], method="iqr")
cleaner.clean_outliers(columns=["precio", "edad"], method="iqr", delete=False)
```

## Desarrollo

Para clonar el repositorio y modificar el código:

**1. Clonar el repositorio**

```bash
git clone https://github.com/NicolasPalomaresDS/palomares-datacleaner.git
cd datacleaner
```

**2. Crear y activar el entorno virtual**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Instalar en modo editable**

```bash
pip install -e .
```

Las dependencias declaradas en `pyproject.toml` se instalan automáticamente.

**4. Instalar dependencias de desarrollo**

```bash
pip install -r requirements-dev.txt
```

**5. Correr los tests**

```bash
pytest tests/ -v
```