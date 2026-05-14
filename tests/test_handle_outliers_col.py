import pytest
import pandas as pd


class DummyCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _handle_outliers_col(self, col: str, lower: float, upper: float, delete: bool) -> int:
        """Copia exacta de la implementación bajo prueba."""
        mask  = (self.df[col] < lower) | (self.df[col] > upper)
        n_out = mask.sum()

        if delete:
            self.df = self.df[~mask]
        else:
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)

        return n_out


@pytest.mark.parametrize(
    "data, col, lower, upper, delete, expected_n_out, expected_col_values, expected_len",
    [
        pytest.param(
            {"precio": [10.0, 200.0, 15.0, 12.0]},
            "precio", 0.0, 100.0, False,
            1,
            [10.0, 100.0, 15.0, 12.0],
            4,
            id="happy-capping-un-outlier",
        ),
        pytest.param(
            {"precio": [10.0, 200.0, 15.0, 12.0]},
            "precio", 0.0, 100.0, True,
            1,
            [10.0, 15.0, 12.0],
            3,
            id="happy-delete-un-outlier",
        ),
        pytest.param(
            {"precio": [10.0, 20.0, 15.0]},
            "precio", 0.0, 100.0, False,
            0,
            [10.0, 20.0, 15.0],
            3,
            id="happy-sin-outliers-sin-cambios",
        ),
    ],
)
def test_handle_outliers_col_happy_paths(data, col, lower, upper, delete, expected_n_out, expected_col_values, expected_len):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    n_out = cleaner._handle_outliers_col(col, lower, upper, delete)

    # Assert
    assert n_out == expected_n_out
    assert len(cleaner.df) == expected_len
    # sourcery skip: no-conditionals-in-tests
    if not delete:
        assert list(cleaner.df[col]) == expected_col_values


@pytest.mark.parametrize(
    "data, col, lower, upper, delete, expected_n_out, expected_len",
    [
        pytest.param(
            {"precio": [200.0, 300.0]},
            "precio", 0.0, 100.0, True,
            2,
            0,
            id="edge-delete-todos-outliers-df-vacio",
        ),
        pytest.param(
            {"precio": [10.0, -5.0, 200.0]},
            "precio", 0.0, 100.0, False,
            2,
            3,
            id="edge-capping-outliers-en-ambos-extremos",
        ),
    ],
)
def test_handle_outliers_col_edge_cases(data, col, lower, upper, delete, expected_n_out, expected_len):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    n_out = cleaner._handle_outliers_col(col, lower, upper, delete)

    # Assert
    assert n_out == expected_n_out
    assert len(cleaner.df) == expected_len


@pytest.mark.parametrize(
    "data, col, lower, upper, delete",
    [
        pytest.param(
            {"precio": []},
            "precio", 0.0, 100.0, False,
            id="error-df-vacio-no-lanza-excepcion",
        ),
    ],
)
def test_handle_outliers_col_error_paths(data, col, lower, upper, delete):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    n_out = cleaner._handle_outliers_col(col, lower, upper, delete)

    # Assert — un df vacío no lanza error, simplemente devuelve 0
    assert n_out == 0
    assert len(cleaner.df) == 0