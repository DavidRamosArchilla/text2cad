import pytest

from t2c.cad.sandbox import CadExecutor


@pytest.fixture(scope="session")
def executor():
    with CadExecutor(workers=4, timeout=15) as ex:
        yield ex
