"""Tests for the archive-view data/plot export."""

import asyncio
import datetime as dt
from uuid import NAMESPACE_URL, uuid5

import pytest

# The export machinery lives on BaseDataView, which imports panel/panelini.
pytest.importorskip("panel")
pytest.importorskip("panelini")
pytest.importorskip("pandas")
pytest.importorskip("pint_pandas")

from opensemantic import compute_scoped_uuid  # noqa: E402
from opensemantic.base.v1 import (  # noqa: E402
    Database,
    DataChannel,
    DataTool,
    DataToolController,
)
from opensemantic.base.view import DataToolView  # noqa: E402
from opensemantic.base.view import (  # noqa: E402
    DataToolPlotControlsConfig,
    DataToolViewConfig,
)
from opensemantic.base.view._base_view import (  # noqa: E402
    BaseDataView,
    _series_to_dataframe,
)
from opensemantic.characteristics.quantitative.v1 import (  # noqa: E402
    Temperature,
    TemperatureUnit,
)
from opensemantic.core.v1 import Label  # noqa: E402


class _Stub(BaseDataView):
    """Minimal BaseDataView to exercise the pure export path."""

    EXPORT_MAX_ROWS = 1_000_000

    def __init__(self, series):
        self._series = series

    def export_series(self):
        return self._series


def _records():
    t0 = dt.datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "label": "tool/temp",
            "x": [t0, t0 + dt.timedelta(seconds=1)],
            "y": [300.0, 301.0],
            "x_kind": "datetime",
            "unit": "kelvin",
        },
        {
            "label": "tool/volt",
            "x": [t0],
            "y": [1.5],
            "x_kind": "datetime",
            "unit": "volt",
        },
    ]


def test_series_to_dataframe_units_and_alignment():
    df = _series_to_dataframe(_records(), 1_000_000)
    assert list(df.columns) == ["tool/temp", "tool/volt"]
    assert df.shape[0] == 2  # outer join over the two timestamps
    assert str(df["tool/temp"].pint.units) == "kelvin"
    assert str(df["tool/volt"].pint.units) == "volt"


def test_build_data_export_csv_has_unit_header():
    text = _Stub(_records())._build_data_export("csv").getvalue().decode()
    # dequantify() writes a unit header row and keeps column keys unit-free.
    assert "kelvin" in text and "volt" in text
    assert "tool/temp" in text and "tool/volt" in text


def test_build_data_export_row_cap():
    n = 100
    t0 = dt.datetime(2024, 1, 1)
    series = [
        {
            "label": "A",
            "x": [t0 + dt.timedelta(seconds=i) for i in range(n)],
            "y": [float(i) for i in range(n)],
            "x_kind": "datetime",
            "unit": "kelvin",
        }
    ]
    assert len(_series_to_dataframe(series, 10)) == 10


def test_empty_series_exports_empty():
    assert _Stub([])._build_data_export("csv").getvalue() == b""
    assert _series_to_dataframe([], 10) is None


def test_duplicate_labels_are_disambiguated():
    # Composite sub-fields can share a label; each must stay its own column so
    # dequantify sees Series (not a DataFrame) per column.
    t0 = dt.datetime(2024, 1, 1)
    series = [
        {"label": "A/AQ", "x": [t0], "y": [1.0], "x_kind": "datetime", "unit": "K"},
        {"label": "A/AQ", "x": [t0], "y": [2.0], "x_kind": "datetime", "unit": "volt"},
    ]
    df = _series_to_dataframe(series, 1_000_000)
    assert list(df.columns) == ["A/AQ", "A/AQ (1)"]
    text = _Stub(series)._build_data_export("csv").getvalue().decode()
    assert "kelvin" in text and "volt" in text


def test_text_channel_exports_as_object_column():
    # A checked text-log channel (unit=None) exports alongside numeric ones.
    t0 = dt.datetime(2024, 1, 1)
    series = [
        {
            "label": "tool/temp",
            "x": [t0],
            "y": [300.0],
            "x_kind": "datetime",
            "unit": "kelvin",
        },
        {
            "label": "tool/status",
            "x": [t0],
            "y": ["OK"],
            "x_kind": "datetime",
            "unit": None,
        },
    ]
    df = _series_to_dataframe(series, 1_000_000)
    assert list(df.columns) == ["tool/temp", "tool/status"]
    assert str(df["tool/temp"].pint.units) == "kelvin"
    assert df["tool/status"].dtype == object
    text = _Stub(series)._build_data_export("csv").getvalue().decode()
    assert "tool/status" in text and "OK" in text and "kelvin" in text


# -- View integration: export_series / figures / plot HTML --


def _loaded_view():
    parent = uuid5(NAMESPACE_URL, "ExportSensor")
    tool = DataTool(
        uuid=parent,
        name="ExportSensor",
        label=[Label(text="Export Sensor")],
        data_channels=[
            DataChannel(
                uuid=str(compute_scoped_uuid(parent, "temp")),
                osw_id="placeholder",
                name="temperature",
                label=[Label(text="Temperature")],
                characteristic=Temperature.get_cls_iri(),
            ),
        ],
        storage_locations=[Database(name="export_test_db", label=[Label(text="DB")])],
    )
    ctrl = DataToolController(tool, auto_archive=True)
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

    async def store():
        for i in range(5):
            await ctrl.store_channel_data(
                DataToolController.StoreChannelDataParams(
                    channel="temperature",
                    value=Temperature(value=300.0 + i, unit=TemperatureUnit.kelvin),
                    timestamp=base + dt.timedelta(seconds=i),
                )
            )

    asyncio.run(store())

    view = DataToolView(
        controllers=[ctrl],
        config=DataToolViewConfig(plot=DataToolPlotControlsConfig(auto_fetch=False)),
        title="Export Test",
        embeddable=True,
    )
    view.set_time_range(
        base - dt.timedelta(seconds=1), base + dt.timedelta(seconds=10), fetch=False
    )
    for root in view._tree.source:
        for child in root.get("children", []):
            child["selected"] = True
    view._update_selection()
    view._update_unit_controls()
    asyncio.run(view._load_and_plot())
    return view, ctrl


def test_datatool_export_series_and_figures():
    view, ctrl = _loaded_view()
    try:
        records = view.export_series()
        assert len(records) == 1
        rec = records[0]
        assert set(rec) == {"label", "x", "y", "x_kind", "unit"}
        assert rec["x_kind"] == "datetime"
        assert rec["unit"] == "kelvin"
        assert rec["y"] == [300.0, 301.0, 302.0, 303.0, 304.0]
        assert len(view.figures) == 1
        html = view._build_plot_html().getvalue().decode()
        assert "<html" in html.lower() and "bokeh" in html.lower()
        csv = view._build_data_export("csv").getvalue().decode()
        assert "kelvin" in csv and "300.0" in csv
    finally:
        db = ctrl.archive_database
        drv = getattr(db, "_driver", None)
        path = getattr(drv, "db_path", None) if drv else None
        if path:
            import os

            try:
                os.remove(path)
            except OSError:
                pass


def test_inplace_update_reuses_sources_on_unit_change():
    """A unit switch updates the existing ColumnDataSources in place.

    Same trace set -> _build_figure takes the in-place path: the same CDS
    objects are reused (no pane rebuild) and their y values are re-converted to
    the newly selected display unit.
    """
    from opensemantic.base.view._channel_utils import get_unit_enum

    view, ctrl = _loaded_view()
    try:
        sources_before = dict(view._trace_sources)
        assert sources_before  # the initial full build populated them
        sig_before = view._plot_signature
        ids_before = {k: id(v) for k, v in sources_before.items()}

        key = next(iter(sources_before))
        y_kelvin = list(sources_before[key].data["y"])

        group_key = next(iter(view._groups))
        sample_ch = view._groups[group_key][0][1]
        enum = get_unit_enum(sample_ch)
        alt = next(m.name for m in enum if m.name != "kelvin")

        view._unit_selections[group_key] = alt
        view._refresh_plot()

        # In-place: signature unchanged and the same CDS objects reused.
        assert view._plot_signature == sig_before
        assert {k: id(v) for k, v in view._trace_sources.items()} == ids_before

        # Values were re-converted to the new unit (no live doc -> applied now).
        y_alt = list(view._trace_sources[key].data["y"])
        assert len(y_alt) == len(y_kelvin)
        assert y_alt != y_kelvin
    finally:
        db = ctrl.archive_database
        drv = getattr(db, "_driver", None)
        path = getattr(drv, "db_path", None) if drv else None
        if path:
            import os

            try:
                os.remove(path)
            except OSError:
                pass


def _cleanup(ctrl):
    db = ctrl.archive_database
    drv = getattr(db, "_driver", None)
    path = getattr(drv, "db_path", None) if drv else None
    if path:
        import os

        try:
            os.remove(path)
        except OSError:
            pass


def test_hover_tool_present_with_time_and_value():
    """Each plot figure has a hover tool reporting timestamp and value."""
    from bokeh.models import HoverTool

    view, ctrl = _loaded_view()
    try:
        assert view.figures
        hovers = view.figures[0].select(HoverTool)
        assert hovers, "no HoverTool on the figure"
        tips = dict(hovers[0].tooltips)
        assert "time" in tips and "value" in tips
        assert tips["value"].startswith("@y")
        assert hovers[0].formatters.get("@x") == "datetime"
    finally:
        _cleanup(ctrl)


def test_hover_unit_updates_on_unit_switch():
    """The hover value template carries the display unit and follows a switch."""
    from bokeh.models import HoverTool

    from opensemantic.base.view._channel_utils import (
        get_available_units,
        get_unit_enum,
    )

    view, ctrl = _loaded_view()
    try:
        group_key = next(iter(view._groups))
        sample_ch = view._groups[group_key][0][1]
        by_name = {u["name"]: u["symbol"] for u in get_available_units(sample_ch)}
        enum = get_unit_enum(sample_ch)
        alt = next(m.name for m in enum if m.name != "kelvin" and m.name in by_name)

        view._unit_selections[group_key] = alt
        view._refresh_plot()

        tips = dict(view.figures[0].select(HoverTool)[0].tooltips)
        assert by_name[alt] in tips["value"]
    finally:
        _cleanup(ctrl)


def test_set_plot_loading_toggles_spinner():
    view, ctrl = _loaded_view()
    try:
        assert view._plot_col.loading is False  # cleared after the initial load
        view._set_plot_loading(True)
        assert view._plot_col.loading is True
        view._set_plot_loading(False)
        assert view._plot_col.loading is False
    finally:
        _cleanup(ctrl)


def test_load_and_plot_wraps_with_loading():
    """_load_and_plot turns the spinner on for the fetch and off at the end."""
    view, ctrl = _loaded_view()
    try:
        calls = []
        view._set_plot_loading = lambda flag: calls.append(bool(flag))
        asyncio.run(view._load_and_plot())
        assert calls, "loading was never toggled"
        assert calls[0] is True
        assert calls[-1] is False
    finally:
        _cleanup(ctrl)


def test_hover_tooltip_renders_in_browser(tmp_path):
    """Playwright: the exported plot renders with a wired hover tool.

    Loads the standalone HTML export (same figures + HoverTool + data as the
    live plot) in a real browser and inspects the rendered Bokeh document for a
    HoverTool whose tooltip reports time + value with the display unit. This is
    deterministic; a pixel-perfect mouse hover over a short line is not. Skips if
    Playwright or a browser is unavailable.
    """
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    view, ctrl = _loaded_view()
    try:
        html = view._build_plot_html().getvalue().decode()
        page_file = tmp_path / "plot.html"
        page_file.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"no browser: {exc}")
            page = browser.new_page()
            page.goto(page_file.as_uri(), wait_until="load")
            page.wait_for_selector("canvas", timeout=30000)
            page.wait_for_function(
                "() => window.Bokeh && Bokeh.documents && Bokeh.documents.length > 0",
                timeout=30000,
            )
            # Pull every HoverTool's tooltip spec out of the live Bokeh document.
            tooltips = page.evaluate(
                """() => {
                    const out = [];
                    for (const doc of Bokeh.documents) {
                        for (const m of doc._all_models.values()) {
                            if (m.type === 'HoverTool') out.push(m.tooltips);
                        }
                    }
                    return out;
                }"""
            )
            browser.close()

        assert tooltips, "no HoverTool in the rendered Bokeh document"
        flat = str(tooltips)
        assert "time" in flat and "value" in flat and "@y" in flat
        # kelvin symbol carried into the value template.
        assert "K" in flat
    finally:
        _cleanup(ctrl)
