import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


class DummyCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df              = df
        self._validate_method        = MagicMock()
        self._iter_outlier_columns   = MagicMock()

    def display_outliers(self, columns: list, method: str = "iqr", threshold: float = 3.0) -> None:
        """Copia exacta de la implementación bajo prueba."""
        self._validate_method(method)

        combined_mask = pd.Series(False, index=self.df.index)

        for col, lower, upper in self._iter_outlier_columns(columns, method, threshold):
            combined_mask |= (self.df[col] < lower) | (self.df[col] > upper)

        if combined_mask.any():
            from IPython.display import display
            display(self.df[combined_mask])
        else:
            print("✅ No se detectaron outliers en las columnas especificadas.")


@pytest.mark.parametrize(
    "df, columns, method, threshold, iter_rows, expected_print",
    [
        pytest.param(
            pd.DataFrame({"precio": [10, 200, 15, 12]}),
            ["precio"],
            "iqr",
            3.0,
            [("precio", 0.0, 100.0)],
            None,  # hay outliers → display, no print
            id="happy-con-outliers-llama-display",
        ),
        pytest.param(
            pd.DataFrame({"precio": [10, 20, 15, 12]}),
            ["precio"],
            "iqr",
            3.0,
            [("precio", 0.0, 100.0)],
            "✅ No se detectaron outliers en las columnas especificadas.",
            id="happy-sin-outliers-imprime-mensaje",
        ),
    ],
)
def test_display_outliers_happy_paths(capsys, df, columns, method, threshold, iter_rows, expected_print):

    # Arrange
    cleaner = DummyCleaner(df=df)
    cleaner._iter_outlier_columns.return_value = iter_rows

    # Act
    with patch("IPython.display.display"):
        cleaner.display_outliers(columns=columns, method=method, threshold=threshold)
    captured = capsys.readouterr().out

    # Assert
    cleaner._validate_method.assert_called_once_with(method)
    cleaner._iter_outlier_columns.assert_called_once_with(columns, method, threshold)
    # sourcery skip: no-conditionals-in-tests
    if expected_print:
        assert expected_print in captured


@pytest.mark.parametrize(
    "df, columns, method, threshold, iter_rows, expected_print",
    [
        pytest.param(
            pd.DataFrame({"precio": [10, 20]}),
            [],
            "iqr",
            3.0,
            [],
            "✅ No se detectaron outliers en las columnas especificadas.",
            id="edge-lista-vacia-mensaje-sin-outliers",
        ),
    ],
)
def test_display_outliers_edge_cases(capsys, df, columns, method, threshold, iter_rows, expected_print):

    # Arrange
    cleaner = DummyCleaner(df=df)
    cleaner._iter_outlier_columns.return_value = iter_rows

    # Act
    cleaner.display_outliers(columns=columns, method=method, threshold=threshold)
    captured = capsys.readouterr().out

    # Assert
    cleaner._validate_method.assert_called_once_with(method)
    assert expected_print in captured


@pytest.mark.parametrize(
    "iter_side_effect, expected_error",
    [
        pytest.param(
            RuntimeError("iter error"),
            "iter error",
            id="error-iter-outlier-columns-raises",
        ),
    ],
)
def test_display_outliers_error_paths(iter_side_effect, expected_error):

    # Arrange
    df = pd.DataFrame({"precio": [10, 200]})
    cleaner = DummyCleaner(df=df)
    cleaner._iter_outlier_columns.side_effect = iter_side_effect

    # Act / Assert
    with pytest.raises(RuntimeError) as excinfo:
        cleaner.display_outliers(columns=["precio"])

    assert expected_error in str(excinfo.value)