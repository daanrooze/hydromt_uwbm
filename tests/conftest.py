"""Global fixtures for pytest"""

import pytest


@pytest.fixture()
def random_number():
    return 4  # chosen by a fair dice roll.


@pytest.fixture()
def one():
    return 1
