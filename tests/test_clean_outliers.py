import pytest
from unittest.mock import MagicMock, call


class DummyCleaner:
    def __init__(self):
        self._validate_method = MagicMock()
        self._iter_outlier_columns = MagicMock()
        self._handle_outliers_col = MagicMock()

    def clean_outliers(
        self,
        columns: list,
        method: str = "iqr",
        delete: bool = False,
        threshold: float = 3.0,
    ) -> None:
        """Copia exacta de la implementación bajo prueba."""
        self._validate_method(method)

        action = "eliminados" if delete else "cappeados"
        print(f"  Método: {method.upper()}  |  Acción: {action}\n")
        print(f"  {'Columna':<25} {'Outliers':>10} {'Límite inf.':>14} {'Límite sup.':>14}")
        print(f"  {'─' * 65}")

        for col, lower, upper in self._iter_outlier_columns(columns, method, threshold):

            n_out = self._handle_outliers_col(col, lower, upper, delete)
            flag = " ⚠️" if n_out > 0 else " ✅"
            print(
                f"  {col:<25} {n_out:>10,}{flag}"
                f"  {lower:>12.2f}  {upper:>12.2f}"
            )


@pytest.mark.parametrize(
    "columns, method, delete, threshold, iter_rows, handle_side_effects, expected_print_fragments",
    [
        pytest.param(
            ["col1", "col2"],
            "iqr",
            False,
            3.0,
            # _iter_outlier_columns yields two columns
            [("col1", 0.0, 10.0), ("col2", -5.5, 20.25)],
            # _handle_outliers_col side effects for each column
            [2, 0],
            [
                "Método: IQR  |  Acción: cappeados",
                "Columna",
                "col1",
                "2 ⚠️",
                "0.00",
                "10.00",
                "col2",
                "0 ✅",
                "-5.50",
                "20.25",
            ],
            id="happy-iqr-capping-two-cols",
        ),
        pytest.param(
            ["price"],
            "z-score",
            True,
            2.5,
            [("price", 100.0, 200.0)],
            [5],
            [
                "Método: Z-SCORE  |  Acción: eliminados",
                "price",
                "5 ⚠️",
                "100.00",
                "200.00",
            ],
            id="happy-zscore-delete-single-col",
        ),
        pytest.param(
            ["x"],
            "iqr",
            True,
            3.0,
            [("x", -1.234, 9.876)],
            [0],
            [
                "Método: IQR  |  Acción: eliminados",
                "x",
                "0 ✅",
                "-1.23",
                "9.88",
            ],
            id="happy-iqr-delete-no-outliers",
        ),
    ],
)
def test_clean_outliers_happy_paths(
    capsys,
    columns,
    method,
    delete,
    threshold,
    iter_rows,
    handle_side_effects,
    expected_print_fragments,
):

    # Arrange

    cleaner = DummyCleaner()
    cleaner._iter_outlier_columns.return_value = iter_rows
    cleaner._handle_outliers_col.side_effect = handle_side_effects

    # Act

    cleaner.clean_outliers(columns=columns, method=method, delete=delete, threshold=threshold)
    captured = capsys.readouterr().out

    # Assert

    cleaner._validate_method.assert_called_once_with(method)
    cleaner._iter_outlier_columns.assert_called_once_with(columns, method, threshold)

    expected_calls = [
        call(col, lower, upper, delete) for (col, lower, upper) in iter_rows
    ]
    assert cleaner._handle_outliers_col.call_args_list == expected_calls

# sourcery skip: no-loop-in-tests
    for fragment in expected_print_fragments:
        assert fragment in captured


@pytest.mark.parametrize(
    "columns, method, delete, threshold, iter_rows, handle_side_effects, expected_calls",
    [
        pytest.param(
            [],
            "iqr",
            False,
            3.0,
            [],
            [],
            [],
            id="edge-empty-columns-no-iteration",
        ),
        pytest.param(
            ["a"],
            "z-score",
            False,
            0.0,
            [("a", 1.0, 2.0)],
            [1],
            [call("a", 1.0, 2.0, False)],
            id="edge-threshold-zero",
        ),
        pytest.param(
            ["a"],
            "iqr",
            True,
            1e9,
            [("a", -1e3, 1e3)],
            [3],
            [call("a", -1e3, 1e3, True)],
            id="edge-very-large-threshold",
        ),
    ],
)
def test_clean_outliers_edge_cases(
    capsys,
    columns,
    method,
    delete,
    threshold,
    iter_rows,
    handle_side_effects,
    expected_calls,
):

    # Arrange

    cleaner = DummyCleaner()
    cleaner._iter_outlier_columns.return_value = iter_rows
    cleaner._handle_outliers_col.side_effect = handle_side_effects

    # Act

    cleaner.clean_outliers(columns=columns, method=method, delete=delete, threshold=threshold)
    captured = capsys.readouterr().out

    # Assert

    cleaner._validate_method.assert_called_once_with(method)
    cleaner._iter_outlier_columns.assert_called_once_with(columns, method, threshold)

    assert cleaner._handle_outliers_col.call_args_list == expected_calls

    assert "Columna" in captured
    assert "─" * 10 in captured  # Parte del separador presente


@pytest.mark.parametrize(
    "method, iter_side_effect, handle_side_effect, expected_error, expected_handle_calls",
    [
        pytest.param(
            "iqr",
            RuntimeError("iterator error"),
            None,
            "iterator error",
            [],
            id="error-iter-outlier-columns-raises",
        ),
        pytest.param(
            "z-score",
            None,
            RuntimeError("handle error"),
            "handle error",
            [call("a", 0.0, 1.0, False)],
            id="error-handle-outliers-col-raises",
        ),
    ],
)
def test_clean_outliers_error_paths(capsys, method, iter_side_effect, handle_side_effect, expected_error, expected_handle_calls):

    # Arrange
    cleaner = DummyCleaner()
    cleaner._iter_outlier_columns.side_effect = iter_side_effect
    cleaner._iter_outlier_columns.return_value = [("a", 0.0, 1.0)]
    cleaner._handle_outliers_col.side_effect = handle_side_effect

    # Act
    with pytest.raises(RuntimeError) as excinfo:
        cleaner.clean_outliers(columns=["a"], method=method, delete=False, threshold=3.0)
    captured = capsys.readouterr().out

    # Assert
    assert expected_error in str(excinfo.value)
    cleaner._validate_method.assert_called_once_with(method)
    cleaner._iter_outlier_columns.assert_called_once_with(["a"], method, 3.0)
    assert cleaner._handle_outliers_col.call_args_list == expected_handle_calls
    assert "Método" in captured