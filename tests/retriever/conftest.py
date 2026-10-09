import pytest

from tests.retriever.helpers import Env


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture
def make_env(tmp_path):
    counter = {"n": 0}

    def make(**kw):
        counter["n"] += 1
        return Env(tmp_path / f"env{counter['n']}", **kw)

    return make
