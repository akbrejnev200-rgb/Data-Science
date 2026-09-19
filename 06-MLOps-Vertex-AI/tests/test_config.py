from aml_detection import config


def test_config_defaults_are_sane():
    assert config.PROJECT_ID
    assert isinstance(config.REGION, str) and config.REGION
    assert 0 < config.CONTAMINATION_PRIOR < 1
    assert 0 < config.TARGET_RECALL <= 1
    assert config.MODELS_DIR.name == "models"
