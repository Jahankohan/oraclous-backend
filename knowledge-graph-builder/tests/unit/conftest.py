"""
Unit test configuration.

Ensures pytest-asyncio is in auto mode so async test functions run without
requiring explicit @pytest.mark.asyncio decorators.
"""
import pytest


def pytest_configure(config):
    """Force asyncio_mode=auto for unit tests."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    # Set asyncio mode to auto if not already set
    try:
        if hasattr(config, "_inicache"):
            config._inicache.pop("asyncio_mode", None)
        config.option.asyncio_mode = "auto"
    except AttributeError:
        pass
