from translator.core.state import AppState

def test_app_state_defaults() -> None:
    state = AppState()
    assert state.is_running is False
    assert state.is_monitoring is False
    assert state.monitoring_region is None
    assert state.errors == []

def test_record_error() -> None:
    state = AppState()
    state.record_error("Error 1")
    assert len(state.errors) == 1
    assert state.errors[0] == "Error 1"

def test_error_limit() -> None:
    state = AppState()
    for i in range(105):
        state.record_error(f"Error {i}")
    
    assert len(state.errors) == 100
    assert state.errors[0] == "Error 5"
    assert state.errors[-1] == "Error 104"
