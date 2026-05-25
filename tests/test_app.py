"""Phase 6 Streamlit app smoke — uses streamlit AppTest. TDD."""

from __future__ import annotations

from pathlib import Path


def test_streamlit_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)
    assert not at.exception, f"App raised: {at.exception}"


def test_streamlit_app_renders_all_five_chart_tabs_and_metrics():
    """Plotly chart is not a typed element in AppTest v1, so we assert the
    structure that proves the app body executed past the headline metrics."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)
    assert len(at.tabs) == 5
    tab_labels = {t.label for t in at.tabs}
    assert "Params vs d_model" in tab_labels
    assert "Measured vs estimated" in tab_labels
    assert len(at.metric) == 4  # Params, Activation, KV cache, Decode step
