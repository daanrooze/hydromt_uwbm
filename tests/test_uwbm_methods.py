"""Unit tests for hydromt_plugin_uwbm specific methods eg setup methods or workflows"""

import pytest


def test_random_number_add(random_number, one):
    assert random_number + one == 5  # should work for all random numbers right?
