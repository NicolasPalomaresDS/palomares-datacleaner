import pytest
from unittest.mock import MagicMock, call


class DummyCleaner:
    def __init__(self):
        self._check_dups  = MagicMock()
        self._delete_dups = MagicMock()

    def clean_dups(self) -> None:
        """Copia exacta de la implementación bajo prueba."""
        self._check_dups()
        self._delete_dups()


@pytest.mark.parametrize(
    "check_effect, delete_effect",
    [
        pytest.param(None, None, id="happy-ambos-colaboradores-llamados"),
    ],
)
def test_clean_dups_happy_paths(check_effect, delete_effect):

    # Arrange
    cleaner = DummyCleaner()
    cleaner._check_dups.side_effect  = check_effect
    cleaner._delete_dups.side_effect = delete_effect

    # Act
    cleaner.clean_dups()

    # Assert
    cleaner._check_dups.assert_called_once()
    cleaner._delete_dups.assert_called_once()


@pytest.mark.parametrize(
    "check_effect, delete_effect",
    [
        pytest.param(None, None, id="edge-llamadas-en-orden-correcto"),
    ],
)
def test_clean_dups_edge_cases(check_effect, delete_effect):

    # Arrange
    cleaner = DummyCleaner()
    call_order = []
    cleaner._check_dups.side_effect  = lambda: call_order.append("check")
    cleaner._delete_dups.side_effect = lambda: call_order.append("delete")

    # Act
    cleaner.clean_dups()

    # Assert
    assert call_order == ["check", "delete"]


@pytest.mark.parametrize(
    "check_raises, delete_raises, expected_error",
    [
        pytest.param(
            RuntimeError("check error"),
            None,
            "check error",
            id="error-check-dups-raises",
        ),
        pytest.param(
            None,
            RuntimeError("delete error"),
            "delete error",
            id="error-delete-dups-raises",
        ),
    ],
)
def test_clean_dups_error_paths(check_raises, delete_raises, expected_error):

    # Arrange
    cleaner = DummyCleaner()
    cleaner._check_dups.side_effect  = check_raises
    cleaner._delete_dups.side_effect = delete_raises

    # Act / Assert
    with pytest.raises(RuntimeError) as excinfo:
        cleaner.clean_dups()

    assert expected_error in str(excinfo.value)