import pytest
import pandas as pd


class DummyCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _evaluate_null_strategy(self, strategy: str = "mean") -> None:
        """Copia exacta de la implementación bajo prueba."""
        for col in self.df.columns[self.df.isna().any()]:
            if strategy == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                raise ValueError(
                    f"Estrategia '{strategy}' no reconocida.",
                    "Use 'mean', 'median' o 'mode'."
                )


@pytest.mark.parametrize(
    "initial_df, strategy, expected_df",
    [
        pytest.param(
            pd.DataFrame({"a": [1.0, None, 3.0]}),
            "mean",
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            id="happy-mean-imputes-with-mean",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.0, None, 3.0, 4.0]}),
            "median",
            pd.DataFrame({"a": [1.0, 3.0, 3.0, 4.0]}),
            id="happy-median-imputes-with-median",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.0, None, 1.0, 2.0]}),
            "mode",
            pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0]}),
            id="happy-mode-imputes-with-mode",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            "mean",
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            id="happy-no-nulls-df-unchanged",
        ),
    ],
)
def test_evaluate_null_strategy_happy_paths(initial_df, strategy, expected_df):

    # Arrange
    cleaner = DummyCleaner(df=initial_df)

    # Act
    cleaner._evaluate_null_strategy(strategy)

    # Assert
    pd.testing.assert_frame_equal(cleaner.df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    "initial_df, strategy, expected_df",
    [
        pytest.param(
            pd.DataFrame({"a": pd.array([None, None, None], dtype="float64")}),
            "mean",
            pd.DataFrame({"a": pd.array([float("nan"), float("nan"), float("nan")], dtype="float64")}),
            id="edge-all-nulls-mean-stays-nan",
        ),
        pytest.param(
            pd.DataFrame({"a": [1.0, None], "b": [3.0, None]}),
            "median",
            pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 3.0]}),
            id="edge-multiple-cols-imputed",
        ),
    ],
)
def test_evaluate_null_strategy_edge_cases(initial_df, strategy, expected_df):

    # Arrange
    cleaner = DummyCleaner(df=initial_df)

    # Act
    cleaner._evaluate_null_strategy(strategy)

    # Assert
    pd.testing.assert_frame_equal(cleaner.df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param("invalid", id="error-estrategia-invalida"),
        pytest.param("",        id="error-estrategia-vacia"),
    ],
)
def test_evaluate_null_strategy_error_paths(strategy):

    # Arrange
    cleaner = DummyCleaner(df=pd.DataFrame({"a": [1.0, None]}))

    # Act / Assert
    with pytest.raises(ValueError):
        cleaner._evaluate_null_strategy(strategy)