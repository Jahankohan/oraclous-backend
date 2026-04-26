"""
Unit test configuration.

Ensures pytest-asyncio is in auto mode so async test functions run without
requiring explicit @pytest.mark.asyncio decorators.

Also stubs missing TASK-003 symbols (CommunitySummaryRetriever,
RetrieverType.COMMUNITY_SUMMARY) so TASK-007/TASK-008 chat_service tests
can be collected and run before the TASK-003 branch is merged.
"""
from unittest.mock import MagicMock


def pytest_configure(config):
    """Force asyncio_mode=auto for unit tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    # Set asyncio mode to auto if not already set
    try:
        if hasattr(config, "_inicache"):
            config._inicache.pop("asyncio_mode", None)
        config.option.asyncio_mode = "auto"
    except AttributeError:
        pass


def pytest_sessionstart(session):
    """
    Inject stubs for symbols that live on the TASK-003 branch (not yet merged)
    but are imported by chat_service.py on the TASK-007 branch.

    These stubs are injected early enough (before collection) so that
    test_chat_service.py and test_bitemporal_story002.py can be collected
    without ImportError.  The stubs do not affect production code — they are
    only present in the test process.
    """
    try:
        import app.services.retriever_factory as _rf
        from app.services.retriever_factory import RetrieverType

        # Stub CommunitySummaryRetriever if absent (TASK-003 not merged)
        if not hasattr(_rf, "CommunitySummaryRetriever"):
            _rf.CommunitySummaryRetriever = MagicMock(name="CommunitySummaryRetriever")

        # Extend RetrieverType with COMMUNITY_SUMMARY if absent (TASK-003 not merged)
        if not hasattr(RetrieverType, "COMMUNITY_SUMMARY"):
            import enum as _enum

            _new_vals = {m.name: m.value for m in RetrieverType}
            _new_vals["COMMUNITY_SUMMARY"] = "community_summary"
            extended = _enum.StrEnum("RetrieverType", _new_vals)
            _rf.RetrieverType = extended

    except Exception:
        # If retriever_factory itself can't be imported yet, skip the stub.
        # The tests that need it will fail gracefully with a clear error.
        pass
