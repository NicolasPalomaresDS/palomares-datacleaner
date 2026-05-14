import pytest
from unittest.mock import MagicMock, call


class DummyCleaner:
    def __init__(self):
        self._convert = MagicMock()

    def cast_columns(self, columns: list, to: str) -> None:
        """Copia exacta de la implementación bajo prueba."""
        for col in columns:
            self._convert(col=col, to=to)


@pytest.mark.parametrize(
    "columns, to, expected_calls",
    [
        pytest.param(
            ["precio", "edad"],
            "float64",
            [call(col="precio", to="float64"), call(col="edad", to="float64")],
            id="happy-dos-columnas-float64",
        ),
        pytest.param(
            ["fecha"],
            "datetime",
            [call(col="fecha", to="datetime")],
            id="happy-una-columna-datetime",
        ),
        pytest.param(
            ["a", "b", "c"],
            "int64",
            [call(col="a", to="int64"), call(col="b", to="int64"), call(col="c", to="int64")],
            id="happy-tres-columnas-int64",
        ),
    ],
)
def test_cast_columns_happy_paths(columns, to, expected_calls):

    # Arrange
    cleaner = DummyCleaner()

    # Act
    cleaner.cast_columns(columns=columns, to=to)

    # Assert
    assert cleaner._convert.call_args_list == expected_calls


@pytest.mark.parametrize(
    "columns, to, expected_call_count",
    [
        pytest.param(
            [],
            "float64",
            0,
            id="edge-lista-vacia-no-llama-convert",
        ),
        pytest.param(
            ["col"],
            "float64",
            1,
            id="edge-una-sola-columna",
        ),
    ],
)
def test_cast_columns_edge_cases(columns, to, expected_call_count):

    # Arrange
    cleaner = DummyCleaner()

    # Act
    cleaner.cast_columns(columns=columns, to=to)

    # Assert
    assert cleaner._convert.call_count == expected_call_count


@pytest.mark.parametrize(
    "columns, to, failing_col, expected_error",
    [
        pytest.param(
            ["precio"],
            "int64",
            "precio",
            "convert error",
            id="error-convert-raises-en-primera-columna",
        ),
    ],
)
def test_cast_columns_error_paths(columns, to, failing_col, expected_error):

    # Arrange
    cleaner = DummyCleaner()
    cleaner._convert.side_effect = RuntimeError("convert error")

    # Act / Assert
    with pytest.raises(RuntimeError) as excinfo:
        cleaner.cast_columns(columns=columns, to=to)

    assert expected_error in str(excinfo.value)