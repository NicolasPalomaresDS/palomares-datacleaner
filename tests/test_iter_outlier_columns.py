import pytest
from unittest.mock import MagicMock
import pandas as pd


class DummyCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df         = df
        self._get_bounds = MagicMock()

    def _iter_outlier_columns(self, columns: list, method: str, threshold: float):
        """Copia exacta de la implementación bajo prueba."""
        for col in columns:
            if col not in self.df.columns:
                print(f"  ✗ '{col}' no existe en el DataFrame — omitida")
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"  ✗ '{col}' no es numérica — omitida")
                continue
            lower, upper = self._get_bounds(col, method, threshold)
            yield col, lower, upper


@pytest.mark.parametrize(
    "df, columns, method, threshold, bounds_side_effect, expected_yields",
    [
        pytest.param(
            pd.DataFrame({"precio": [1.0, 2.0], "edad": [3.0, 4.0]}),
            ["precio", "edad"],
            "iqr",
            3.0,
            [(0.0, 100.0), (10.0, 80.0)],
            [("precio", 0.0, 100.0), ("edad", 10.0, 80.0)],
            id="happy-dos-columnas-validas",
        ),
        pytest.param(
            pd.DataFrame({"precio": [1.0, 2.0]}),
            ["precio"],
            "zscore",
            2.5,
            [(5.0, 50.0)],
            [("precio", 5.0, 50.0)],
            id="happy-una-columna-valida",
        ),
    ],
)
def test_iter_outlier_columns_happy_paths(capsys, df, columns, method, threshold, bounds_side_effect, expected_yields):

    # Arrange
    cleaner = DummyCleaner(df=df)
    cleaner._get_bounds.side_effect = bounds_side_effect

    # Act
    result = list(cleaner._iter_outlier_columns(columns, method, threshold))

    # Assert
    assert result == expected_yields
    assert cleaner._get_bounds.call_count == len(expected_yields)


@pytest.mark.parametrize(
    "df, columns, method, threshold, expected_yields, expected_print",
    [
        pytest.param(
            pd.DataFrame({"precio": [1.0, 2.0]}),
            [],
            "iqr",
            3.0,
            [],
            None,
            id="edge-lista-vacia-no-yields",
        ),
        pytest.param(
            pd.DataFrame({"precio": [1.0, 2.0]}),
            ["inexistente"],
            "iqr",
            3.0,
            [],
            "no existe en el DataFrame",
            id="edge-columna-inexistente-omitida",
        ),
        pytest.param(
            pd.DataFrame({"nombre": ["Ana", "Luis"]}),
            ["nombre"],
            "iqr",
            3.0,
            [],
            "no es numérica",
            id="edge-columna-no-numerica-omitida",
        ),
    ],
)
def test_iter_outlier_columns_edge_cases(capsys, df, columns, method, threshold, expected_yields, expected_print):

    # Arrange
    cleaner = DummyCleaner(df=df)

    # Act
    result = list(cleaner._iter_outlier_columns(columns, method, threshold))
    captured = capsys.readouterr().out

    # Assert
    assert result == expected_yields
    # sourcery skip: no-conditionals-in-tests
    if expected_print:
        assert expected_print in captured


@pytest.mark.parametrize(
    "df, columns, method, threshold, expected_error",
    [
        pytest.param(
            pd.DataFrame({"precio": [1.0, 2.0]}),
            ["precio"],
            "iqr",
            3.0,
            "bounds error",
            id="error-get-bounds-raises",
        ),
    ],
)
def test_iter_outlier_columns_error_paths(df, columns, method, threshold, expected_error):

    # Arrange
    cleaner = DummyCleaner(df=df)
    cleaner._get_bounds.side_effect = RuntimeError("bounds error")

    # Act / Assert
    with pytest.raises(RuntimeError) as excinfo:
        list(cleaner._iter_outlier_columns(columns, method, threshold))

    assert expected_error in str(excinfo.value)