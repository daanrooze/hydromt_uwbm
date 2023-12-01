"""Testing UWBM high level methods"""

from hydromt_plugin_uwbm import UWBM


def test_model_has_name():
    model = UWBM()
    assert hasattr(model, "_NAME")


def test_model_class():
    model = UWBM()
    non_compliant = model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
