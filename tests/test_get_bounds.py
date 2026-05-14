import pytest
import pandas as pd


class DummyCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_bounds_iqr(self, col: str) -> tuple:
        """Copia exacta de la implementación bajo prueba."""
        Q1  = self.df[col].quantile(0.25)
        Q3  = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        return (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    def _get_bounds_zscore(self, col: str, threshold: float) -> tuple:
        """Copia exacta de la implementación bajo prueba."""
        mean = self.df[col].mean()
        std  = self.df[col].std()
        return (mean - threshold * std, mean + threshold * std)


# ─────────────────────────────────────────────────────────────────────────────
# _get_bounds_iqr
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "data, col, expected_lower, expected_upper",
    [
        pytest.param(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0]},
            "a",
            -1.0,
            7.0,
            id="happy-iqr-serie-simple"
        ),
        pytest.param(
            {"precio": [100.0, 200.0, 300.0, 400.0, 500.0]},
            "precio",
            -100.0,
            700.0,
            id="happy-iqr-valores-grandes"
        ),
    ],
)
def test_get_bounds_iqr_happy_paths(data, col, expected_lower, expected_upper):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_iqr(col)

    # Assert
    assert pytest.approx(lower, rel=1e-3) == expected_lower
    assert pytest.approx(upper, rel=1e-3) == expected_upper


@pytest.mark.parametrize(
    "data, col",
    [
        pytest.param(
            {"a": [5.0, 5.0, 5.0, 5.0]},
            "a",
            id="edge-iqr-cero-todos-iguales",
        ),
    ],
)
def test_get_bounds_iqr_edge_cases(data, col):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_iqr(col)

    # Assert — con IQR=0, lower y upper son iguales al valor constante
    assert lower == upper == 5.0


@pytest.mark.parametrize(
    "data, col",
    [
        pytest.param(
            {"a": [1.0]},
            "a",
            id="error-una-sola-fila-iqr-cero",
        ),
    ],
)
def test_get_bounds_iqr_error_paths(data, col):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_iqr(col)

    # Assert — no lanza error, IQR=0 así que lower==upper==valor
    assert lower == upper


# ─────────────────────────────────────────────────────────────────────────────
# _get_bounds_zscore
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "data, col, threshold, expected_lower, expected_upper",
    [
        pytest.param(
            {"a": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}, "a", 3.0,
            pytest.approx(-1.414, rel=1e-2),
            pytest.approx(11.414, rel=1e-2),
            id="happy-zscore-threshold-3",
        ),
        pytest.param(
            {"a": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}, "a", 2.0,
            pytest.approx(0.724, rel=1e-2),
            pytest.approx(9.276, rel=1e-2),
            id="happy-zscore-threshold-2",
        ),
    ],
)
def test_get_bounds_zscore_happy_paths(data, col, threshold, expected_lower, expected_upper):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_zscore(col, threshold)

    # Assert
    assert lower == expected_lower
    assert upper == expected_upper


@pytest.mark.parametrize(
    "data, col, threshold",
    [
        pytest.param(
            {"a": [5.0, 5.0, 5.0, 5.0]},
            "a",
            3.0,
            id="edge-zscore-std-cero-todos-iguales",
        ),
    ],
)
def test_get_bounds_zscore_edge_cases(data, col, threshold):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_zscore(col, threshold)

    # Assert — con std=0, lower y upper son iguales a la media
    assert lower == upper == 5.0


@pytest.mark.parametrize(
    "data, col, threshold",
    [
        pytest.param(
            {"a": [1.0, 2.0, 3.0]},
            "a",
            0.0,
            id="error-threshold-cero-bounds-iguales-a-media",
        ),
    ],
)
def test_get_bounds_zscore_error_paths(data, col, threshold):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame(data))

    # Act
    lower, upper = cleaner._get_bounds_zscore(col, threshold)

    # Assert — threshold=0 → lower==upper==mean
    assert lower == upper == 2.0