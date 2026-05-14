import pandas as pd
from IPython.display import display, HTML

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    #------------------------------------------------------------------------
    # SUMARIO DEL DATAFRAME
    #------------------------------------------------------------------------
    def get_summary(self, name: str="DataFrame") -> None:
        """Imprime un sumario con información del DataFrame

        Args:
            name (str, optional): nombre del DataFrame. Defaults to "DataFrame".
        """
        nrows, ncols = self.df.shape
        total_nulls = self.df.isna().sum().sum()
        total_dups = self.df.duplicated().sum()

        # Encabezado
        print(f"\n{'━' * 60}")
        print(f"  📋  {name}")
        print(f"{'━' * 60}")

        # Métricas rápidas
        metrics = {
            "Filas":         f"{nrows:,}",
            "Columnas":      f"{ncols}",
            "Nulos totales": f"{total_nulls:,}",
            "Duplicados":    f"{total_dups:,}",
        }
        col_w = 20
        for label, value in metrics.items():
            print(f"  {label:<{col_w}}{value}")

        # Tipos de datos + nulos + cardinalidad por columna
        print(f"\n{'─' * 70}")
        print(f"  {'Columna':<25} {'Tipo':<12} {'Únicos':>8} {'Nulos':>8} {'% nulos':>10}")
        print(f"{'─' * 70}")

        null_counts   = self.df.isna().sum()
        unique_counts = self.df.nunique()

        for col in self.df.columns:
            nulls   = null_counts[col]
            uniques = unique_counts[col]
            pct     = nulls / nrows * 100
            flag    = " ⚠️" if pct > 0 else ""

            print(
                f"  {col:<25}",
                f"{str(self.df[col].dtype):<12}",
                f"{uniques:>8,}",
                f"{nulls:>8,}",
                f"{pct:>9.2f}%{flag}"
            )

        # Duplicados
        print(f"\n{'─' * 60}")
        dup_flag = " ⚠️" if total_dups > 0 else " ✅"
        print(f"  Filas duplicadas: {total_dups:,}{dup_flag}")

        # Primeras filas
        print(f"\n{'─' * 60}")
        print(f"  Primeras 5 filas:\n")
        display(self.df.head())
        print(f"{'━' * 60}\n")

    #------------------------------------------------------------------------
    # TRATAMIENTO DE NULLS
    #------------------------------------------------------------------------

    def display_nulls(self) -> None:
        """Muestra un primer vistazo de las filas del DataFrame con valores nulos"""
        display(self.df[self.df.isna().any(axis=1)])

    def _evaluate_null_strategy(self, columns: list, strategy: str = "mean") -> None:
        """Imputa los datos nulos con la estadística especificada

        Args:
            columns (list): lista con los nombres de las columnas a imputar.
            strategy (str, optional): Estrategia para imputar valores nulos. Defaults to "mean".
        """
        for col in columns:
            if col not in self.df.columns:
                print(f"  ✗ '{col}' no existe en el DataFrame — omitida")
                continue
            if strategy == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                raise ValueError(
                    f"Estrategia '{strategy}' no reconocida. Use 'mean', 'median' o 'mode'."
                )

    def clean_nulls(
        self,
        columns:  list,
        delete:   bool = False,
        strategy: str  = "mean"
    ) -> None:
        """Limpia los datos nulos del DataFrame

        Args:
            columns (list): lista con los nombres de las columnas a limpiar.
            delete (bool, optional): decide si eliminar o reemplazar los datos nulos. Defaults to False.
            strategy (str, optional): Estrategia para imputar valores nulos. Defaults to "mean".
        """
        if delete:
            self.df = self.df.dropna(subset=columns)
            print("Datos nulos correctamente eliminados")
        else:
            self._evaluate_null_strategy(columns, strategy)
            print(f"Datos nulos correctamente reemplazados con: {strategy}")


    #------------------------------------------------------------------------
    # TRATAMIENTO DE DUPLICADOS
    #------------------------------------------------------------------------

    def _check_dups(self) -> None:
        """Imprime un checkeo general de los datos duplicados"""
        fully_duplicated = self.df[self.df.duplicated(keep=False)]
        duplicated_index = self.df[self.df.index.duplicated(keep=False)]

        is_match = len(fully_duplicated) == len(duplicated_index)

        print(
            f"Filas completamente duplicadas: {len(fully_duplicated)}",
            f"\nFilas con index duplicado: {len(duplicated_index)}",
            f"\n¿Coinciden? {is_match}\n",
            "-" * 80
        )

    def _delete_dups(self) -> None:
        """Elimina las filas duplicadas del DataFrame"""
        self.df = self.df[~self.df.duplicated(keep="first")]

        print(
            f"Filas restantes: {len(self.df)}",
            f"\nIndex duplicados restantes: {self.df.index.duplicated().sum()}"
        )

    def clean_dups(self) -> None:
        """Limpia por completo los datos duplicados del DataFrame"""
        self._check_dups()
        self._delete_dups()

    #------------------------------------------------------------------------
    # TRATAMIENTO DE DTYPES
    #------------------------------------------------------------------------

    def _datetime_condition(self, col: str) -> None:
        """Convierte las fechas a tipo 'datetime'

        Args:
            col (str): nombre de la columna a convertir.
        """
        self.df[col] = pd.to_datetime(
            self.df[col],
            dayfirst=True
        )

    def _numeric_condition(self, col: str, to: str) -> None:
        """Convierte los datos numéricos a int o float y elimina caracteres innecesarios

        Args:
            col (str): nombre de la columna a convertir
            to (str): nombre del tipo de dato objetivo (int64/float64)
        """
        cleaned = (
            self.df[col]
            .astype(str)
            .str.replace(
                r"[^0-9.]",
                "",
                regex=True
            )
        )

        if to == "int64":
            self.df[col] = (
                cleaned
                .astype(float)
                .astype(int)
            )

        else:
            self.df[col] = cleaned.astype(float)

    def _convert(self, col: str, to: str) -> None:
        """Convierte los datos de la columna al tipo especificado

        Args:
            col (str): nombre de la columna a convertir
            to (str): nombre del tipo de dato objetivo
        """
        try:
            if to == "int64" and self.df[col].isna().any():
                null_count = self.df[col].isna().sum()
                print(
                    f"✗ No se pudo convertir '{col}' a int64: "
                    f"la columna tiene {null_count} valor/es nulo/s. "
                    f"Se recomienda limpiar los nulos primero con clean_nulls()."
                )
                return

            if to == "datetime":
                self._datetime_condition(col=col)

            elif to in {"int64", "float64"}:
                self._numeric_condition(col=col, to=to)

            else:
                self.df[col] = self.df[col].astype(to)

            print(
                f"✓ '{col}' convertida correctamente → "
                f"{self.df[col].dtype}"
            )

        except (ValueError, TypeError) as e:
            print(
                f"✗ No se pudo convertir '{col}' "
                f"a {to}: {e}"
            )


    def cast_columns(self, columns: list, to: str) -> None:
        """Corrige los tipos de datos de las columnas del DataFrame

        Args:
            columns (list): lista con los nombres de las columnas a convertir
            to (str): nombre del tipo de dato objetivo
        """
        for col in columns:
            self._convert(col=col, to=to)

    #------------------------------------------------------------------------
    # TRATAMIENTO DE OUTLIERS
    #------------------------------------------------------------------------

    def _validate_method(self, method: str) -> None:
        """Valida que el método de tratamiento de outliers sea reconocido

        Args:
            method (str): método a validar

        Raises:
            ValueError: si el método no es 'iqr' ni 'zscore'
        """
        if method not in ("iqr", "zscore"):
            raise ValueError(
                f"Método '{method}' no reconocido. Use 'iqr' o 'zscore'."
            )

    def _get_iqr_bounds(self, col: str) -> tuple:
        """Calcula límite inferior y superior calculando IQR

        Args:
            col (str): nombre de la columna

        Returns:
            tuple: (límite inferior, límite superior)
        """
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (lower_bound, upper_bound)

    def _get_zscore_bounds(self, col: str, threshold: float) -> tuple:
        """Calcula límite inferior y superior calculando Z-score

        Args:
            col (str): nombre de la columna
            threshold (float): umbral a considerar

        Returns:
            tuple: (límite inferior, límite superior)
        """
        mean = self.df[col].mean()
        std = self.df[col].std()

        lower_bound = mean - threshold * std
        higher_bound = mean + threshold * std

        return (lower_bound, upper_bound)

    def _get_bounds(
        self,
        col: str,
        method: str,
        threshold: float
    ) -> tuple:
        """Calcula los límites del método correspondiente

        Args:
            col (str): nombre de la columna
            method (str): 'iqr' o 'zscore'
            threshold (float): umbral para Z-score

        Returns:
            tuple: (límite inferior, límite superior)
        """
        if method == "iqr":
            return self._get_iqr_bounds(col)

        return self._get_zscore_bounds(col, threshold)

    def _iter_outlier_columns(
        self,
        columns: list,
        method: str,
        threshold: float,
    ):
        """Itera sobre las columnas y devuelve los límites de outliers

        Args:
            columns (list): lista con los nombres de las columnas
            method (str): 'iqr' o 'zscore'
            threshold (float): umbral para Z-score

        Yields:
            tuple: (col, lower, upper)
        """
        for col in columns:
            if col not in self.df.columns:
                print(f"  ✗ '{col}' no existe en el DataFrame — omitida")
                continue

            if not pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"  ✗ '{col}' no es numérica — omitida")
                continue

            lower, upper = self._get_bounds(col, method, threshold)
            yield col, lower, upper

    def _handle_outliers(
        self,
        col: str,
        lower: float,
        upper: float,
        delete: bool
    ) -> int:
        """Maneja outliers con la acción elegida (elimina/cappea)

        Args:
            col (str): nombre de la columna
            lower (float): límite inferior
            upper (float): límite superior
            delete (bool): True elimina filas, False aplica capping

        Returns:
            int: cantidad de outliers detectados
        """
        mask = (self.df[col] < lower) | (self.df[col] > upper)
        n_out = mask.sum()

        if delete:
            self.df = self.df[~mask]
        else:
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)

        return n_out

    def display_outliers(
        self,
        columns: list,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> None:
        """Muestra los outliers detectados en las columnas especificadas

        Args:
            columns (list): lista con los nombres de las columnas
            method (str, optional): 'iqr' o 'zscore'. Defaults to "iqr".
            threshold (float, optional): umbral para Z-score. Defaults to 3.0.
        """
        self._validate_method(method)
        combined_mask = pd.Series(False, index=self.df.index)

        for col, lower, upper in self._iter_outlier_columns(
            columns = columns,
            method = method,
            threshold = threshold
        ):
            combined_mask |= (self.df[col] < lower) | (self.df[col] > upper)

        if combined_mask.any():
            display(self.df[combined_mask])
        else:
            print("✅ No se detectaron outliers en las columnas.")


    def clean_outliers(
        self,
        columns: list,
        method: str = "iqr",
        delete: bool = False,
        threshold: float = 3.0
    ) -> None:
        """Detecta y trata outliers de las columnas especificadas

        Args:
            columns (list): lista con los nombres de las columnas
            method (str, optional): IQR o Z-score. Defaults to "iqr".
            delete (bool, optional): True elimina filas, False aplica capping. Defaults to False.
            threshold (float, optional): umbral de detección para Z-score. Defaults to 3.0.
        """
        self._validate_method(method)

        action = "eliminados" if delete else "cappeados"
        print(f"  Método: {method.upper()}  |  Acción: {action}\n")
        print(f"  {'Columna':<25} {'Outliers':>10} {'Límite inf.':>14} {'Límite sup.':>14}")
        print(f"  {'─' * 65}")

        for col, lower, upper in self._iter_outlier_columns(columns, method, threshold):

            n_out = self._handle_outliers_col(col, lower, upper, delete)
            flag  = " ⚠️" if n_out > 0 else " ✅"
            print(
                f"  {col:<25} {n_out:>10,}{flag}"
                f"  {lower:>12.2f}  {upper:>12.2f}"
            )

