"""DataTool View - archive time series visualization.

Provides a Panelini-based view with:
- Wunderbaum TreeGrid sidebar showing DataTools and their channels
- Bokeh time series plot for selected (checked) channels
- Time range selection and unit switching controls

Usage:
    from opensemantic.base.view import DataToolView

    view = DataToolView(controllers=[ctrl1, ctrl2])
    view.servable()
"""

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import panel as pn
from bokeh.events import Reset
from bokeh.models import (
    BoxZoomTool,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    DatetimeTickFormatter,
)
from bokeh.plotting import figure as bk_figure
from panelini import Panelini
from panelini.panels.wunderbaum import Wunderbaum
from pydantic import ConfigDict, Field

from opensemantic.base.view._base_view import COLORS, BaseDataView
from opensemantic.base.view._channel_utils import (
    _t,
    build_tree_source,
    flatten_composite_channels,
    get_display_label,
    get_selected_channels,
    get_unit_enum,
    get_unit_enum_from_value,
    group_channels_by_characteristic,
    resolve_characteristic_class,
    resolve_downsample_method,
    resolve_value_type,
    to_display_magnitudes,
)
from opensemantic.base.view._config import (
    BaseViewConfig,
    PlotControlsConfig,
    TimeRange,
    TreeConfig,
)
from opensemantic.base.view._data_cache import ChannelDataCache


class DataToolPlotControlsConfig(PlotControlsConfig):
    """Plot controls for the channel-centered view - adds an absolute window."""

    model_config = ConfigDict(
        json_schema_extra={
            "title": "DataToolPlotControlsConfig",
            "defaultProperties": [
                "grouping",
                "auto_fetch",
                "row_limit",
                "cache_enabled",
                "downsample",
                "unit_selections",
                "time_range",
            ],
        }
    )

    time_range: Optional[TimeRange] = Field(
        None, title="Time range", json_schema_extra={"title*": {"de": "Zeitbereich"}}
    )


class DataToolViewConfig(BaseViewConfig):
    """Config for :class:`DataToolView` (channel-centered).

    Composes one channel ``tree`` and a plot-controls config with a
    ``time_range``. ``tree.source`` carries the checked-channel selection in
    Wunderbaum's native format.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "title": "DataToolViewConfig",
            "defaultProperties": ["controllers", "lang", "tree", "plot"],
        }
    )

    tree: TreeConfig = Field(default_factory=TreeConfig, title="Channel tree")
    plot: DataToolPlotControlsConfig = Field(
        default_factory=DataToolPlotControlsConfig, title="Plot controls"
    )


_logger = logging.getLogger(__name__)

# Wunderbaum's grid sizes its container to a fixed 800px (inline width), which
# overflows the sidebar and shows a horizontal scrollbar. Force it to fill the
# available width instead. !important beats the inline style (which has none).
_TREE_GRID_CSS = (
    ".tree-container.wunderbaum { width: 100% !important; }\n"
    ".wunderbaum-wrapper { overflow-x: hidden !important; }"
)


class DataToolView(BaseDataView):
    """Archive-mode DataTool view.

    Displays multiple DataToolControllers in a TreeGrid sidebar.
    Users check channels to plot archived time series data.
    Channels sharing the same characteristic are grouped on shared y-axes.

    Parameters
    ----------
    controllers
        List of DataToolController instances to display.
    config
        Dashboard configuration. If None, uses defaults.
    title
        Dashboard title shown in the Panelini header.
    """

    config_cls = DataToolViewConfig

    def __init__(
        self,
        controllers: Optional[List[Any]] = None,
        config: Optional[BaseViewConfig] = None,
        title: str = "DataTool Dashboard",
        embeddable: bool = False,
        url_sync: bool = False,
        url_mode=None,
    ):
        self._controllers = controllers or []
        self._config = type(self)._coerce_config(config)
        # The concrete config class is the single source of truth for
        # validation / round-tripping (a subclass such as LiveDataToolViewConfig).
        self._config_cls = type(self._config)
        self._url_sync = url_sync
        self._url_mode = url_mode
        self._title = title
        # When embeddable, skip building the internal Panelini app so the cards
        # can be placed into a host app via sidebar_cards / main_cards (Panel
        # rejects the same model living in two layouts/documents).
        self._embeddable = embeddable

        # Build lookup maps
        self._channel_map: Dict[str, Any] = {}
        self._controller_map: Dict[str, Any] = {}
        self._build_lookup_maps()

        # State
        self._selected: List[Tuple[Any, Any]] = []
        self._groups: Dict[str, List[Tuple[Any, Any]]] = {}
        self._unit_selections: Dict[str, str] = dict(self._config.plot.unit_selections)
        self._cached_data: Dict[str, List] = {}
        self._composite_parents: Dict[str, Tuple[Any, str]] = {}

        # Zoom-driven downsampling state. _zoom_window overrides the picker
        # window when "Load current range" re-fetches the box-zoomed range;
        # _shared_x_range is the linked x-range the button reads.
        self._zoom_window: Optional[Tuple[dt.datetime, dt.datetime]] = None
        self._shared_x_range = None
        # Reset bridge. figure.on_event(Reset) does not propagate through
        # Panel's Bokeh pane, but model property changes sync reliably. A
        # CustomJS on the Reset event bumps this source's data; its
        # server-side on_change (_on_reset_bridge) reloads the full window
        # after a "Load current range" zoom-reload.
        self._reset_bridge = ColumnDataSource(data={"n": [0]})
        self._reset_bridge.on_change("data", self._on_reset_bridge)

        # Cache
        self._cache = ChannelDataCache(enabled=self._config.plot.cache_enabled)

        # Build UI
        self._build_tree()
        self._build_controls()
        self._build_plot()
        self._build_log_console()
        self._build_layout()

        # Reflect any pre-set state (selection / time / units) from the config,
        # then optionally bind the config to the URL.
        self._apply_initial_config()
        if self._url_sync:
            self._enable_url_sync(mode=self._url_mode)

    def _build_lookup_maps(self):
        for ctrl in self._controllers:
            for ch in ctrl.get_all_channels():
                self._channel_map[ch.uuid] = ch
                self._controller_map[ch.uuid] = ctrl

    # -- TreeGrid --

    def _build_tree(self):
        source = build_tree_source(self._controllers, self.lang)
        self._tree = Wunderbaum(
            source=source,
            columns=[
                {"id": "*", "title": _t("channel", self.lang), "width": "190px"},
                {
                    "id": "characteristic",
                    "title": _t("characteristic", self.lang),
                    "width": "130px",
                },
            ],
            options={"checkbox": True, "selectMode": "hier"},
            stylesheets=[_TREE_GRID_CSS],
        )
        # Watch the source param for checkbox changes (emitSource syncs selected state)
        self._tree.param.watch(self._on_source_change, ["source"])
        self._tree_card = pn.Card(
            self._tree,
            title=_t("data_channels", self.lang),
            collapsed=False,
        )

    def _on_source_change(self, *args):
        """Handle tree source changes (checkbox toggling)."""
        if getattr(self, "_applying_config", False):
            return
        try:
            # A selection change loads the full picker window, not a zoom slice.
            self._zoom_window = None
            self._update_selection()
            self._update_unit_controls()
            self._write_selection_to_config()
            if self._config.plot.auto_fetch:
                self._trigger_load()
            self._emit_config_change()
        except Exception as e:
            _logger.error("Error in _on_source_change: %s", e)

    def _write_selection_to_config(self):
        """Mirror the checked channel keys into ``config.tree.selected``."""
        self._config.tree.selected = [
            child.get("key")
            for root in self._tree.source
            for child in root.get("children", [])
            if child.get("selected")
        ]

    def _update_selection(self):
        raw_selected = get_selected_channels(self._tree.source, self._controllers)
        # Expand composite channels into leaf sub-channels for grouping
        self._selected = []
        self._composite_parents = {}  # sub_ch.uuid -> (parent_ch, field_name)
        for ctrl, ch in raw_selected:
            vtype = resolve_value_type(ch)
            if vtype == "composite":
                for sub_ctrl, sub_ch, path in flatten_composite_channels(ctrl, ch):
                    self._selected.append((ctrl, sub_ch))
                    self._composite_parents[sub_ch.uuid] = (ch, sub_ch.name)
            else:
                self._selected.append((ctrl, ch))
        self._groups = group_channels_by_characteristic(
            self._selected, self._config.plot.grouping
        )

    # -- Plot Controls --

    def _build_controls(self):
        now = dt.datetime.now()
        self._start_picker = pn.widgets.DatetimePicker(
            name=_t("start", self.lang),
            value=now - dt.timedelta(hours=1),
        )
        self._end_picker = pn.widgets.DatetimePicker(
            name=_t("end", self.lang),
            value=now,
        )
        self._load_button = pn.widgets.Button(
            name=_t("load_data", self.lang),
            button_type="primary",
        )
        self._load_button.on_click(self._on_load_click)

        # Re-fetch the currently visible (box-zoomed) x-range at finer
        # resolution, keeping the zoom. Box-zoom itself is purely visual.
        self._load_range_button = pn.widgets.Button(
            name=_t("load_range", self.lang),
            button_type="default",
        )
        self._load_range_button.on_click(self._on_load_range)

        self._auto_fetch_cb = pn.widgets.Checkbox(
            name=_t("auto_fetch", self.lang),
            value=self._config.plot.auto_fetch,
        )
        self._auto_fetch_cb.param.watch(self._on_auto_fetch_change, ["value"])

        self._row_limit_input = pn.widgets.IntInput(
            name=_t("row_limit", self.lang),
            value=self._config.plot.row_limit,
            start=1,
            step=1000,
        )
        self._row_limit_input.param.watch(self._on_row_limit_change, ["value"])

        self._clear_cache_button = pn.widgets.Button(
            name=_t("clear_cache", self.lang),
            button_type="warning",
        )
        self._clear_cache_button.on_click(self._on_clear_cache)

        self._unit_controls = pn.Column()

        self._start_picker.param.watch(self._on_time_change, ["value"])
        self._end_picker.param.watch(self._on_time_change, ["value"])

        self._controls_card = pn.Card(
            self._start_picker,
            self._end_picker,
            self._load_button,
            self._load_range_button,
            self._auto_fetch_cb,
            self._build_grouping_control(),
            self._row_limit_input,
            self._build_cache_control(),
            self._clear_cache_button,
            self._build_downsample_controls(),
            self._unit_controls,
            self._build_export_toolbar(),
            title=_t("plot_controls", self.lang),
        )

    def _regroup(self):
        self._update_selection()

    def _has_active_selection(self) -> bool:
        return bool(self._selected)

    def _on_load_click(self, event):
        self._trigger_load()

    def _on_time_change(self, event):
        if getattr(self, "_applying_config", False):
            return
        # A manual time-range edit overrides any active zoom slice.
        self._zoom_window = None
        self._write_time_range_to_config()
        if self._config.plot.auto_fetch and self._selected:
            self._trigger_load()
        self._emit_config_change()

    def _write_time_range_to_config(self):
        """Mirror the pickers (naive local) into ``config.plot.time_range``."""
        self._config.plot.time_range = TimeRange(
            start=self._start_picker.value, end=self._end_picker.value
        )

    def _on_auto_fetch_change(self, event):
        if getattr(self, "_applying_config", False):
            return
        self._config.plot.auto_fetch = event.new
        self._emit_config_change()

    def _on_row_limit_change(self, event):
        if getattr(self, "_applying_config", False):
            return
        self._config.plot.row_limit = event.new
        self._emit_config_change()

    def _on_clear_cache(self, event):
        self._cache.clear_cache()
        self._cached_data.clear()

    # -- Config apply (config -> view); base handles the JsonEditor wiring --

    def _apply_config(self, old, new):
        """Apply a config to the widgets/state (see BaseDataView.set_config)."""
        super()._apply_config(old, new)  # units + plot-control widgets
        if old is None or old.controllers != getattr(new, "controllers", []):
            self._on_controllers_changed()
        if old is None or old.lang != new.lang:
            self._rebuild_ui_labels()
        self._apply_selection(new)
        self._apply_time_range(new)
        # Rebuild grouping/units from the applied selection, then (re)plot.
        self._update_selection()
        self._update_unit_controls()
        self._refresh_plot()
        if new.plot.auto_fetch and self._selected:
            self._trigger_load()

    def _apply_selection(self, config):
        """Rebuild the tree from the data with the config's channels checked."""
        keys = set(config.tree.selected or [])
        source = build_tree_source(self._controllers, self.lang)
        for root in source:
            for child in root.get("children", []):
                child["selected"] = child.get("key") in keys
        self._tree.set_source(source)

    def _config_has_state(self, config):
        return (
            bool(config.tree.selected)
            or bool(config.plot.unit_selections)
            or bool(config.plot.time_range)
        )

    def _apply_time_range(self, config):
        """Apply the config's time range to the pickers (verbatim if naive)."""
        tr = getattr(config.plot, "time_range", None)
        if not tr:
            return
        if tr.start is not None:
            self._start_picker.value = (
                tr.start if tr.start.tzinfo is None else self._to_picker_value(tr.start)
            )
        if tr.end is not None:
            self._end_picker.value = (
                tr.end if tr.end.tzinfo is None else self._to_picker_value(tr.end)
            )

    def _on_controllers_changed(self):
        """Handle controllers list change from config editor.

        Subclasses (LiveDashboard) can override to resolve IRIs
        and create new controller instances.
        """
        self._cache.clear_cache()
        self._cached_data.clear()
        self._build_lookup_maps()
        # Rebuild tree
        source = build_tree_source(self._controllers, self.lang)
        self._tree.set_source(source)

    def _rebuild_ui_labels(self):
        """Rebuild all UI labels after language change."""
        self._tree_card.title = _t("data_channels", self.lang)
        self._controls_card.title = _t("plot_controls", self.lang)
        self._plot_card.title = _t("time_series", self.lang)
        self._log_card.title = _t("log_console", self.lang)
        self._load_button.name = _t("load_data", self.lang)
        self._auto_fetch_cb.name = _t("auto_fetch", self.lang)
        self._row_limit_input.name = _t("row_limit", self.lang)
        self._clear_cache_button.name = _t("clear_cache", self.lang)
        self._start_picker.name = _t("start", self.lang)
        self._end_picker.name = _t("end", self.lang)
        # Rebuild tree source with new lang
        source = build_tree_source(self._controllers, self.lang)
        self._tree.set_source(source)
        # Rebuild unit controls
        self._update_unit_controls()

    # -- Data Loading (_trigger_load comes from BaseDataView) --

    async def _load_and_plot(self):
        """Load data for all selected channels and update plots."""
        if self._zoom_window is not None:
            start, end = self._zoom_window
        else:
            # Respect the pickers. auto_fetch only controls whether a change
            # (selection/time) auto-triggers a load, not the window itself;
            # live "follow now" is the LiveDataToolView's job, not this view's.
            start = self._start_picker.value
            end = self._end_picker.value
        if start is None or end is None:
            return

        # DatetimePicker returns naive local time; convert to UTC
        if start.tzinfo is None:
            start = start.astimezone(dt.timezone.utc)
        if end.tzinfo is None:
            end = end.astimezone(dt.timezone.utc)

        limit = self._config.plot.row_limit
        self._cached_data.clear()

        # Load data for each channel. For composite sub-channels, load the parent.
        loaded_parents = set()
        for ctrl, ch in self._selected:
            try:
                if ch.uuid in self._composite_parents:
                    parent_ch, _ = self._composite_parents[ch.uuid]
                    if parent_ch.uuid in loaded_parents:
                        continue
                    loaded_parents.add(parent_ch.uuid)
                    mp, method, edge = self._downsample_for(parent_ch)
                    rows = await self._cache.get_data(
                        ctrl,
                        parent_ch,
                        start,
                        end,
                        limit,
                        typed=False,
                        max_points=mp,
                        method=method,
                        edge_anchors=edge,
                    )
                    self._cached_data[parent_ch.uuid] = rows
                else:
                    mp, method, edge = self._downsample_for(ch)
                    rows = await self._cache.get_data(
                        ctrl,
                        ch,
                        start,
                        end,
                        limit,
                        typed=False,
                        max_points=mp,
                        method=method,
                        edge_anchors=edge,
                    )
                    self._cached_data[ch.uuid] = rows
            except Exception as e:
                _logger.error("Error loading %s/%s: %s", ctrl.name, ch.name, e)

        self._refresh_plot()

    def _downsample_for(self, channel):
        """Return ``(max_points, method, edge_anchors)`` for a channel.

        Default applies the dashboard config, auto-resolving the method per
        channel type. Override to vary the strategy per channel (e.g. force
        a different method, or raw/no downsampling, for specific channels).
        Return ``(None, None, None)`` to read the channel at full resolution.
        """
        ds = self._config.plot.downsample
        if not ds.enabled:
            return None, None, None
        method = resolve_downsample_method(channel, ds.method.value)
        return ds.max_points, method, ds.edge_anchors

    def _collect_traces(self):
        """Extract per-group plot traces from the cache in one pass.

        Returns an ordered list of ``(group_key, axis_label, [(trace_name,
        timestamps, values), ...])`` for every non-text group that has data.
        Shared by the live build, the in-place update and HTML export so the
        data is extracted (and unit-converted) once.
        """
        traces = []
        for group_key, channels in self._groups.items():
            if not channels:
                continue
            if resolve_value_type(channels[0][1]) == "text":
                continue
            series = []
            for ctrl, ch in channels:
                timestamps, values = self._extract_trace_data(ch, group_key)
                if len(timestamps):
                    trace_name = (
                        f"{get_display_label(ctrl, self.lang)}/"
                        f"{get_display_label(ch, self.lang)}"
                    )
                    series.append((trace_name, timestamps, values))
            if series:
                traces.append((group_key, self._get_axis_label(group_key), series))
        return traces

    def _make_figures(self, traces=None, source_map=None, group_map=None):
        """Build one Bokeh figure per group from collected traces.

        Returns ``(figs, shared_x_range)``. ``traces`` defaults to a fresh
        extraction, so this doubles as the detached builder for HTML export.
        When given, ``source_map``/``group_map`` are populated with the
        per-trace ColumnDataSource and per-group figure so a later refresh can
        update them in place instead of rebuilding.
        """
        if traces is None:
            traces = self._collect_traces()
        if not traces:
            return [], None

        # One shared x-range for every figure, so panning, zooming or resetting
        # any plot moves all of them together (the y-ranges stay independent).
        # Sharing it at figure creation - rather than reassigning f.x_range
        # after the panes render - is what guarantees the link: a deferred
        # reassignment races the render and leaves some plots unlinked.
        shared_x = DataRange1d()

        figs = []
        color_idx = 0
        for group_key, axis_label, series in traces:
            fig = bk_figure(
                height=250,
                sizing_mode="stretch_width",
                x_axis_type="datetime",
                x_range=shared_x,
                y_axis_label=axis_label,
                tools="pan,wheel_zoom,reset,save",
            )
            # x-only box zoom as the active drag tool: dragging zooms the time
            # axis only (y stays auto so a downsampled spike isn't clipped).
            # The zoom is purely visual; use the "Load current range" button to
            # re-fetch that window at a finer resolution.
            box_zoom = BoxZoomTool(dimensions="width")
            fig.add_tools(box_zoom)
            fig.toolbar.active_drag = box_zoom
            fig.xaxis.formatter = DatetimeTickFormatter(
                seconds="%H:%M:%S",
                minutes="%H:%M",
                hours="%H:%M",
            )

            for trace_name, timestamps, values in series:
                src = ColumnDataSource(data={"x": timestamps, "y": values})
                fig.line(
                    "x",
                    "y",
                    source=src,
                    legend_label=trace_name,
                    color=COLORS[color_idx % len(COLORS)],
                    line_width=2,
                )
                color_idx += 1
                if source_map is not None:
                    source_map[(group_key, trace_name)] = src

            fig.legend.click_policy = "hide"
            if group_map is not None:
                group_map[group_key] = fig
            figs.append(fig)
        return figs, shared_x

    def _build_figure(self):
        """Build or update the plot: one figure per characteristic group.

        Rebuilds the panes only when the trace set (groups x channels) changes;
        otherwise updates the existing ColumnDataSources (and axis labels) in
        place, which avoids re-serializing whole Bokeh models to the browser on
        every reload or unit switch. The in-place mutation is scheduled on the
        Bokeh document (see _schedule_doc) to avoid the async load's doc lock.
        """
        traces = self._collect_traces()
        signature = tuple(
            (gk, tn) for gk, _al, series in traces for (tn, _ts, _v) in series
        )
        sources = getattr(self, "_trace_sources", None)
        if (
            signature
            and signature == getattr(self, "_plot_signature", None)
            and sources
            and all((gk, tn) in sources for gk, tn in signature)
        ):
            self._update_plot_in_place(traces)
            return

        self._plot_col.clear()
        source_map: dict = {}
        group_map: dict = {}
        figs, shared_x = self._make_figures(traces, source_map, group_map)
        self._figures = figs
        self._trace_sources = source_map
        self._group_figures = group_map
        self._plot_signature = signature if figs else None
        if not figs:
            return

        # Keep the shared range for "Load current range" and bridge the Reset
        # event: figure.on_event(Reset) does not propagate through Panel's Bokeh
        # pane, so a CustomJS on the Reset event bumps _reset_bridge, whose
        # server-side on_change reloads.
        self._shared_x_range = shared_x
        reset_cb = CustomJS(
            args={"bridge": self._reset_bridge},
            code="bridge.data = {n: [(bridge.data.n[0] || 0) + 1]};",
        )
        for fig in figs:
            fig.js_on_event(Reset, reset_cb)

        for fig in figs:
            self._plot_col.append(pn.pane.Bokeh(fig, sizing_mode="stretch_width"))

    def _update_plot_in_place(self, traces):
        """Update existing figures' data and axis labels without new panes."""
        sources = self._trace_sources
        group_figs = getattr(self, "_group_figures", {})
        updates = []
        for group_key, _axis_label, series in traces:
            for trace_name, timestamps, values in series:
                src = sources.get((group_key, trace_name))
                if src is not None:
                    updates.append((src, {"x": timestamps, "y": values}))

        def _apply():
            for src, data in updates:
                src.data = data
            for group_key, axis_label, _series in traces:
                fig = group_figs.get(group_key)
                if fig is not None:
                    fig.yaxis.axis_label = axis_label
            # The shared x-range is left untouched: an auto range (never zoomed)
            # re-follows the new data on the CDS update, and a prior box-zoom is
            # intentionally preserved so "Load current range" keeps the window.

        self._schedule_doc(_apply)

    def _schedule_doc(self, fn):
        """Run a Bokeh-model mutation on the document thread when live.

        Mutating live models directly from the async load task raises a
        document-lock error, so schedule on the session document's next tick.
        Falls back to a direct call when there is no live document (tests,
        initial synchronous build).
        """
        doc = getattr(pn.state, "curdoc", None)
        if doc is not None and getattr(doc, "session_context", None) is not None:
            try:
                doc.add_next_tick_callback(fn)
                return
            except Exception:
                pass
        fn()

    def _export_figures(self):
        """Fresh, unattached figures for HTML export (see _make_figures)."""
        figs, _ = self._make_figures()
        return figs

    def _current_xrange_window(self):
        """Return (start, end) *naive* datetimes of the current plot x-range.

        The shared x-range start/end (epoch ms) reflect the user's latest
        pan/zoom, synced from the browser. The data is plotted in local wall
        time (see _extract_trace_data), so these epochs are the local
        wall-clock values - returned as naive datetimes so _load_and_plot
        converts them naive->UTC exactly like the date pickers do (treating
        them as local). Returning UTC-aware here would shift the loaded window
        by the local/UTC offset. Returns None if no zoom range is set.
        """
        xr = self._shared_x_range
        if xr is None or xr.start is None or xr.end is None:
            return None
        try:
            start = dt.datetime.fromtimestamp(
                xr.start / 1000.0, dt.timezone.utc
            ).replace(tzinfo=None)
            end = dt.datetime.fromtimestamp(xr.end / 1000.0, dt.timezone.utc).replace(
                tzinfo=None
            )
        except Exception:
            return None
        return (start, end) if end > start else None

    def _on_load_range(self, event):
        """Re-fetch the currently visible (zoomed) x-range at finer resolution."""
        window = self._current_xrange_window()
        if window is None:
            return
        self._zoom_window = window
        self._trigger_load()

    def _on_reset_bridge(self, attr, old, new):
        """Toolbar reset (bridged via CustomJS -> source data change).

        figure.on_event(Reset) does not reach the server through Panel's Bokeh
        pane, so the Reset event is bridged through this source's data change
        (see _build_figure). After "Load current range" the figures hold only
        the
        zoomed window, so Bokeh's own reset would only return to that window;
        clear the zoom and reload the full (picker) window. For a plain visual
        box-zoom (nothing was reloaded) there is no zoom window, and Bokeh's
        reset already restores the full view, so we do nothing.
        """
        if self._zoom_window is None:
            return
        self._zoom_window = None
        self._trigger_load()

    @staticmethod
    def _naive_local(ts: Any) -> Any:
        """Normalize a timestamp to a naive local datetime for plotting."""
        if isinstance(ts, str):
            ts = dt.datetime.fromisoformat(ts)
        if ts.tzinfo is not None:
            ts = ts.astimezone(tz=None).replace(tzinfo=None)
        return ts

    @staticmethod
    def _leaf_number(v: Any) -> Any:
        """Extract the bare numeric leaf from a cached value (dict/typed/scalar)."""
        if isinstance(v, dict):
            return v.get("value")
        if hasattr(v, "value"):
            return v.value
        if isinstance(v, (int, float)):
            return v
        return None

    def _series_units(
        self, ch: Any, group_key: str, raw0: Any, field: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve ``(source_unit, target_unit)`` pint strings for a series.

        Builds one representative typed value from the first raw row to read the
        stored (source) unit and the display (target) unit, so the per-point
        unit conversion can be replaced by a single vectorized pass. The display
        unit is the group's selected unit, else the channel's declared unit
        (scalar), else the stored unit. Returns ``(None, None)`` when units
        cannot be resolved (untyped / dimensionless), which skips conversion.
        """
        cls = resolve_characteristic_class(ch)
        if cls is None or not hasattr(cls, "from_json"):
            return None, None
        try:
            rep = cls.from_json(raw0)
        except Exception:
            return None, None
        if field is not None:
            rep = getattr(rep, field, None)
        if rep is None or not hasattr(rep, "to_pint"):
            return None, None
        try:
            source = str(rep.to_pint().units)
        except Exception:
            return None, None

        sel = self._unit_selections.get(group_key)
        enum = get_unit_enum(ch) if field is None else get_unit_enum_from_value(rep)
        disp = rep
        try:
            if sel and enum is not None and sel in enum.__members__:
                disp = rep.to_unit(enum[sel])
            elif field is None:
                ch_unit = getattr(ch, "unit", None)
                if ch_unit is not None:
                    disp = rep.to_unit(ch_unit)
            target = str(disp.to_pint().units)
        except Exception:
            target = source
        return source, target

    def _extract_trace_data(self, ch: Any, group_key: str) -> Tuple[List, Any]:
        """Extract timestamps and display-unit values from cached points.

        Values are converted to the display unit in a single vectorized
        pint-pandas pass (see ``_series_units``/``to_display_magnitudes``)
        rather than per point. Handles regular and composite sub-channels.
        """
        if ch.uuid in self._composite_parents:
            parent_ch, field_name = self._composite_parents[ch.uuid]
            return self._extract_composite_field(parent_ch, field_name, group_key)

        points = self._cached_data.get(ch.uuid, [])
        if not points:
            return [], []

        timestamps: List = []
        raw: List = []
        for pt in points:
            v = self._leaf_number(pt.value)
            if v is None:
                continue
            timestamps.append(self._naive_local(pt.timestamp))
            raw.append(v)
        if not raw:
            return [], []

        # Text / non-numeric series (log channels, exported alongside plots)
        # pass through unconverted; only numeric series get the vectorized
        # pint-pandas unit conversion.
        if not all(isinstance(v, (int, float)) for v in raw):
            return timestamps, raw

        source, target = self._series_units(ch, group_key, points[0].value)
        return timestamps, to_display_magnitudes(raw, source, target)

    def _extract_composite_field(
        self, parent_ch: Any, field_name: str, group_key: str
    ) -> Tuple[List, Any]:
        """Extract a sub-field from composite points, vectorizing the conversion.

        Reads from the parent channel's cache and extracts the named field.
        Works with both raw dicts (typed=False loading) and typed objects.
        """
        points = self._cached_data.get(parent_ch.uuid, [])
        if not points:
            return [], []

        timestamps: List = []
        raw: List = []
        for pt in points:
            value = pt.value
            if isinstance(value, dict):
                sub = value.get(field_name)
            elif hasattr(value, field_name):
                sub = getattr(value, field_name)
            else:
                continue
            v = self._leaf_number(sub) if sub is not None else None
            if v is None:
                continue
            timestamps.append(self._naive_local(pt.timestamp))
            raw.append(v)
        if not raw:
            return [], []

        if not all(isinstance(v, (int, float)) for v in raw):
            return timestamps, raw

        source, target = self._series_units(
            parent_ch, group_key, points[0].value, field=field_name
        )
        return timestamps, to_display_magnitudes(raw, source, target)

    # -- Export --

    def _representative_value(self, ch: Any) -> Any:
        """A typed (Characteristic) value for the channel, or None.

        The plot cache holds raw dicts (typed=False), so a raw value is typed on
        demand via the characteristic class - one build per channel, only for
        export/unit-label needs, not per point.
        """
        if ch.uuid in self._composite_parents:
            parent_ch, field_name = self._composite_parents[ch.uuid]
            cls = resolve_characteristic_class(parent_ch)
            for pt in self._cached_data.get(parent_ch.uuid, []):
                val = pt.value
                if isinstance(val, dict):
                    if cls is None or not hasattr(cls, "from_json"):
                        continue
                    try:
                        val = cls.from_json(val)
                    except Exception:
                        continue
                sub = getattr(val, field_name, None)
                if sub is not None and hasattr(sub, "to_pint"):
                    return sub
            return None
        cls = resolve_characteristic_class(ch)
        for pt in self._cached_data.get(ch.uuid, []):
            val = pt.value
            if isinstance(val, dict):
                if cls is None or not hasattr(cls, "from_json"):
                    continue
                try:
                    val = cls.from_json(val)
                except Exception:
                    continue
            if hasattr(val, "to_pint"):
                return val
        return None

    def _series_unit(self, ch: Any, group_key: str) -> str:
        """Pint-parseable unit string of the plotted (display-unit) values."""
        return self._pint_unit(
            self._representative_value(ch), self._unit_selections.get(group_key)
        )

    def export_series(self) -> List[dict]:
        """Every checked channel as tidy records (datetime x).

        Numeric channels carry their display unit and match _build_figure's
        traces; text-log channels (a different display group) are included too,
        with ``unit=None`` so they export as plain text columns.
        """
        records: List[dict] = []
        for group_key, channels in self._groups.items():
            if not channels:
                continue
            is_text = resolve_value_type(channels[0][1]) == "text"
            for ctrl, ch in channels:
                x, y = self._extract_trace_data(ch, group_key)
                if not x:
                    continue
                # Composite sub-fields share the parent channel label, so add
                # the sub-field name to keep one column per series.
                if ch.uuid in self._composite_parents:
                    parent_ch, field_name = self._composite_parents[ch.uuid]
                    label = (
                        f"{get_display_label(ctrl, self.lang)}/"
                        f"{get_display_label(parent_ch, self.lang)}/{field_name}"
                    )
                else:
                    label = (
                        f"{get_display_label(ctrl, self.lang)}/"
                        f"{get_display_label(ch, self.lang)}"
                    )
                records.append(
                    {
                        "label": label,
                        "x": x,
                        # y is a numpy array for numeric series (vectorized
                        # conversion); export records keep the list contract.
                        "y": list(y),
                        "x_kind": "datetime",
                        "unit": None if is_text else self._series_unit(ch, group_key),
                    }
                )
        return records

    def _update_log_console(self):
        """Update the log console with text-type channel data."""
        log_entries = []
        has_text = False

        for group_key, channels in self._groups.items():
            for ctrl, ch in channels:
                vtype = resolve_value_type(ch)
                if vtype != "text":
                    continue
                has_text = True
                rows = self._cached_data.get(ch.uuid, [])
                ch_label = get_display_label(ch, self.lang)
                for pt in rows:
                    val = pt.value
                    if hasattr(val, "value"):
                        text = str(val.value)
                    elif isinstance(val, dict):
                        text = str(val.get("value", val))
                    else:
                        text = str(val)
                    log_entries.append((pt.timestamp, ch_label, text))

        self._log_card.visible = has_text
        if not log_entries:
            return

        log_entries.sort(key=lambda x: x[0])
        html_lines = []
        for ts, ch_label, text in log_entries:
            if isinstance(ts, str):
                ts_str = ts
            elif hasattr(ts, "strftime"):
                local_ts = ts.astimezone(tz=None) if ts.tzinfo else ts
                ts_str = local_ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            html_lines.append(
                f"<div style='font-family:monospace; font-size:12px;'>"
                f"<span style='color:#888'>{ts_str}</span> "
                f"<span style='color:#1f77b4'>[{ch_label}]</span> {text}</div>"
            )
        self._log_pane.object = "\n".join(html_lines)

    # -- Layout --

    def _build_layout(self):
        if self._embeddable:
            # Host app owns the layout; expose cards via sidebar_cards/main_cards.
            self._app = None
            return
        self._app = Panelini(
            title=self._title,
            sidebar_enabled=True,
            sidebars_max_width=400,
        )
        self._app.sidebar_set(
            [
                self._tree_card,
                self._controls_card,
            ]
        )
        self._build_main_area()

    def _build_main_area(self):
        """Set main area content. Override in subclasses for tabs."""
        self._app.main_set([self._plot_card, self._log_card])

    @property
    def sidebar_cards(self):
        """Sidebar cards (channel tree, plot controls) for embedding."""
        return [self._tree_card, self._controls_card]

    @property
    def main_cards(self):
        """Main-area cards (time series plot, log console) for embedding."""
        return [self._plot_card, self._log_card]

    def set_time_range(self, start, end, fetch: bool = True):
        """Set an explicit time range and (optionally) reload the data.

        Accepts tz-aware or naive datetimes (naive is treated as UTC). The
        window is honored as-is: auto-fetch no longer pins the end to ``now``
        (it only controls whether changes auto-trigger a load), so an arbitrary
        historical window can be shown without disabling auto-fetch.
        """
        self._start_picker.value = self._to_picker_value(start)
        self._end_picker.value = self._to_picker_value(end)
        if fetch:
            self._trigger_load()

    @staticmethod
    def _to_picker_value(value):
        """Convert a datetime to the naive local value the picker expects.

        The DatetimePicker holds naive local time which _load_and_plot converts
        back to UTC, so hand it the local-naive representation of ``value``.
        """
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone().replace(tzinfo=None)

    # servable() / panel() come from BaseDataView.
