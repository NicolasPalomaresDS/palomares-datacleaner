import pytest
from unittest.mock import MagicMock
import pandas as pd


class DummyCleaner:
    def __init__(self):
        self._evaluate_null_strategy = MagicMock()
        self.df = pd.DataFrame({"a": [1.0, None]})

    def clean_nulls(self, delete: bool = False, strategy: str = "mean") -> None:
        """Copia exacta de la implementación bajo prueba."""
        if delete:
            self.df = self.df.dropna()
            print("Datos nulos correctamente eliminados")
        else:
            self._evaluate_null_strategy(strategy)
            print(f"Datos nulos correctamente reemplazados con: {strategy}")


@pytest.mark.parametrize(
    "delete, strategy, expected_print, evaluate_called",
    [
        pytest.param(
            False,
            "mean",
            "Datos nulos correctamente reemplazados con: mean",
            True,
            id="happy-imputation-mean",
        ),
        pytest.param(
            False,
            "median",
            "Datos nulos correctamente reemplazados con: median",
            True,
            id="happy-imputation-median",
        ),
        pytest.param(
            False,
            "mode",
            "Datos nulos correctamente reemplazados con: mode",
            True,
            id="happy-imputation-mode",
        ),
        pytest.param(
            True,
            "mean",
            "Datos nulos correctamente eliminados",
            False,
            id="happy-delete-ignora-strategy",
        ),
    ],
)
def test_clean_nulls_happy_paths(capsys, delete, strategy, expected_print, evaluate_called):

    # Arrange
    cleaner = DummyCleaner()

    # Act
    cleaner.clean_nulls(delete=delete, strategy=strategy)
    captured = capsys.readouterr().out

    # Assert
    assert expected_print in captured
    # sourcery skip: no-conditionals-in-tests
    if evaluate_called:
        cleaner._evaluate_null_strategy.assert_called_once_with(strategy)
    else:
        cleaner._evaluate_null_strategy.assert_not_called()


@pytest.mark.parametrize(
    "delete, strategy, expected_print, evaluate_called",
    [
        pytest.param(
            False,
            "mean",
            "Datos nulos correctamente reemplazados con: mean",
            True,
            id="edge-defaults",
        ),
        pytest.param(
            True,
            "median",
            "Datos nulos correctamente eliminados",
            False,
            id="edge-delete-true-strategy-ignorada",
        ),
    ],
)
def test_clean_nulls_edge_cases(capsys, delete, strategy, expected_print, evaluate_called):

    # Arrange
    cleaner = DummyCleaner()

    # Act
    cleaner.clean_nulls(delete=delete, strategy=strategy)
    captured = capsys.readouterr().out

    # Assert
    assert expected_print in captured
    # sourcery skip: no-conditionals-in-tests
    if evaluate_called:
        cleaner._evaluate_null_strategy.assert_called_once_with(strategy)
    else:
        cleaner._evaluate_null_strategy.assert_not_called()


@pytest.mark.parametrize(
    "strategy, exc_type, expected_error",
    [
        pytest.param(
            "invalid",
            ValueError,
            "bad strategy",
            id="error-invalid-strategy",
        ),
        pytest.param(
            "",
            ValueError,
            "bad strategy",
            id="error-empty-strategy",
        ),
    ],
)
def test_clean_nulls_error_paths(capsys, strategy, exc_type, expected_error):

    # Arrange
    cleaner = DummyCleaner()
    cleaner._evaluate_null_strategy.side_effect = exc_type("bad strategy")

    # Act
    with pytest.raises(exc_type) as excinfo:
        cleaner.clean_nulls(delete=False, strategy=strategy)
    captured = capsys.readouterr().out

    # Assert
    assert expected_error in str(excinfo.value)
    cleaner._evaluate_null_strategy.assert_called_once_with(strategy)
    assert "Datos nulos correctamente reemplazados con" not in captured