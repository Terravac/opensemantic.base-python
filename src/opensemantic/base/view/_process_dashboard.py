"""Process/object-centered DataTool view.

Two Wunderbaum trees:
- Objects (Item instances tracked as process inputs)
- Process types -> aggregated channels (grouped by datatool type, channel name,
  characteristic)

Selected object x aggregated-channel pairs resolve to concrete channels, are
loaded over each process's [start, end] window, time-normalized so the first
data point of each process is t=0, and plotted (seconds on the x-axis) grouped
by characteristic - reusing the plot/group/unit machinery of DataToolView.

Usage:
    from opensemantic.base.view import ProcessObjectView

    view = ProcessObjectView(objects, processes, controllers)
    view.servable()
"""

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import panel as pn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure as bk_figure
from panelini import Panelini
from panelini.panels.wunderbaum import Wunderbaum
from pydantic import ConfigDict, Field

from opensemantic.base.view._base_view import COLORS, BaseDataView
from opensemantic.base.view._channel_utils import (
    _t,
    get_display_label,
    group_channels_by_characteristic,
    resolve_downsample_method,
    resolve_value_type,
    to_utc,
)
from opensemantic.base.view._config import BaseViewConfig, TreeConfig
from opensemantic.base.view._data_cache import ChannelDataCache
from opensemantic.base.view._process_utils import (
    build_concrete_tree,
    build_object_tree_source,
    build_process_tree_source,
    derive_aggregated_channels,
    entity_iri,
    get_selected_keys,
    resolve_aggregated_channel,
)

_logger = logging.getLogger(__name__)

# Local labels not in the shared _channel_utils translation table.
_PT_STRINGS = {
    "objects": {"en": "Objects", "de": "Objekte"},
    "process_channels": {
        "en": "Process Types / Channels",
        "de": "Prozesstypen / Kanaele",
    },
    "rel_time": {"en": "t [s] (relative)", "de": "t [s] (relativ)"},
    "n_processes": {"en": "#", "de": "#"},
}


def _pt(key: str, lang: str = "en") -> str:
    entry = _PT_STRINGS.get(key, {})
    return entry.get(lang, entry.get("en", key))


class ProcessObjectViewConfig(BaseViewConfig):
    """Config for :class:`ProcessObjectView` (process/object-centered).

    Composes two trees (object + process); their ``source`` fields carry the
    checked selection in Wunderbaum's native format. Process traces use relative
    time, so the plot has no absolute ``time_range``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "title": "ProcessObjectViewConfig",
            "defaultProperties": [
                "controllers",
                "lang",
                "object_tree",
                "process_tree",
                "plot",
            ],
        }
    )

    object_tree: TreeConfig = Field(default_factory=TreeConfig, title="Object tree")
    process_tree: TreeConfig = Field(default_factory=TreeConfig, title="Process tree")


class ProcessObjectView(BaseDataView):
    """Process/object-centered archive view.

    Parameters
    ----------
    objects
        Item instances (the entities tracked as process inputs) shown in tree 1.
    processes
        Process instances to scan. A process qualifies when one of ``objects``
        is among its inputs, it has start+end times, and >=1 DataTool attached.
    controllers
        DataToolController instances used to load channel data; matched to
        process tools by IRI.
    config
        Dashboard configuration (uses ``lang`` and ``plot.*``). If None, defaults.
    title
        Dashboard title shown in the Panelini header.
    embeddable
        If True, skip the internal Panelini app (expose ``sidebar_cards`` /
        ``main_cards`` for a host app instead).
    """

    config_cls = ProcessObjectViewConfig

    def __init__(
        self,
        objects: Optional[List[Any]] = None,
        processes: Optional[List[Any]] = None,
        controllers: Optional[List[Any]] = None,
        config: Optional[BaseViewConfig] = None,
        title: str = "Process Dashboard",
        embeddable: bool = False,
        url_sync: bool = False,
        url_mode=None,
    ):
        self._objects = objects or []
        self._processes = processes or []
        self._controllers = controllers or []
        self._config = type(self)._coerce_config(config)
        self._config_cls = type(self._config)
        self._url_sync = url_sync
        self._url_mode = url_mode
        self._title = title
        self._embeddable = embeddable

        # Virtual structure
        self._concrete = build_concrete_tree(
            self._objects, self._processes, self._controllers, self.lang
        )
        self._aggregated = derive_aggregated_channels(self._concrete, self.lang)
        self._agg_by_key: Dict[str, Dict[str, Any]] = {}
        for grp in self._aggregated.values():
            for agg in grp["channels"].values():
                self._agg_by_key[agg["key"]] = agg

        # State
        self._cache = ChannelDataCache(enabled=self._config.plot.cache_enabled)
        self._selected_objects: List[Dict[str, Any]] = []
        self._selected_aggs: List[Dict[str, Any]] = []
        self._groups: Dict[str, List[Tuple[Any, Any]]] = {}
        self._group_of: Dict[str, str] = {}
        self._unit_selections: Dict[str, str] = dict(self._config.plot.unit_selections)
        self._traces: List[Dict[str, Any]] = []
        self._t0: Dict[Tuple[Any, Any], dt.datetime] = {}

        # UI
        self._build_object_tree()
        self._build_process_tree()
        self._build_controls()
        self._build_plot()
        self._build_log_console()
        self._build_layout()

        self._apply_initial_config()
        if self._url_sync:
            self._enable_url_sync(mode=self._url_mode)

    # -- Trees --

    def _build_object_tree(self):
        source = build_object_tree_source(self._concrete, self.lang)
        self._obj_tree = Wunderbaum(
            source=source,
            height=220,
            columns=[
                {"id": "*", "title": _pt("objects", self.lang), "width": "200px"},
                {
                    "id": "processes",
                    "title": _pt("n_processes", self.lang),
                    "width": "60px",
                },
            ],
            options={"checkbox": True, "selectMode": "hier"},
        )
        self._obj_tree.param.watch(self._on_change, ["source"])
        self._obj_tree_card = pn.Card(
            self._obj_tree,
            title=_pt("objects", self.lang),
            collapsed=False,
        )

    def _build_process_tree(self):
        source = build_process_tree_source(self._aggregated, self.lang)
        self._proc_tree = Wunderbaum(
            source=source,
            height=220,
            columns=[
                {
                    "id": "*",
                    "title": _pt("process_channels", self.lang),
                    "width": "220px",
                },
                {
                    "id": "characteristic",
                    "title": _t("characteristic", self.lang),
                    "width": "150px",
                },
            ],
            options={"checkbox": True, "selectMode": "hier"},
        )
        self._proc_tree.param.watch(self._on_change, ["source"])
        self._proc_tree_card = pn.Card(
            self._proc_tree,
            title=_pt("process_channels", self.lang),
            collapsed=False,
        )

    def _on_change(self, *args):
        if getattr(self, "_applying_config", False):
            return
        try:
            self._recompute_selection()
            self._update_unit_controls()
            self._write_selection_to_config()
            if self._config.plot.auto_fetch:
                self._trigger_load()
            self._emit_config_change()
        except Exception as e:
            _logger.error("Error in _on_change: %s", e)

    def _write_selection_to_config(self):
        """Mirror the checked object / aggregate keys into the config."""
        self._config.object_tree.selected = get_selected_keys(self._obj_tree.source)
        self._config.process_tree.selected = get_selected_keys(self._proc_tree.source)

    def _recompute_selection(self):
        obj_keys = get_selected_keys(self._obj_tree.source)
        agg_keys = get_selected_keys(self._proc_tree.source)
        self._selected_objects = [
            self._concrete[k] for k in obj_keys if k in self._concrete
        ]
        self._selected_aggs = [
            self._agg_by_key[k] for k in agg_keys if k in self._agg_by_key
        ]
        # Unique concrete (controller, channel) pairs for y-axis grouping
        pairs: Dict[str, Tuple[Any, Any]] = {}
        for obj_entry in self._selected_objects:
            for agg in self._selected_aggs:
                for ctrl, ch, _proc in resolve_aggregated_channel(obj_entry, agg):
                    pairs[ch.uuid] = (ctrl, ch)
        self._groups = group_channels_by_characteristic(
            list(pairs.values()), self._config.plot.grouping
        )
        self._group_of = {}
        for gkey, chans in self._groups.items():
            for _ctrl, ch in chans:
                self._group_of[ch.uuid] = gkey

    # -- Controls --

    def _build_controls(self):
        self._load_button = pn.widgets.Button(
            name=_t("load_data", self.lang), button_type="primary"
        )
        self._load_button.on_click(lambda e: self._trigger_load())

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
            name=_t("clear_cache", self.lang), button_type="warning"
        )
        self._clear_cache_button.on_click(self._on_clear_cache)

        self._unit_controls = pn.Column()

        self._controls_card = pn.Card(
            self._load_button,
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
        self._recompute_selection()

    def _has_active_selection(self) -> bool:
        return bool(self._selected_objects and self._selected_aggs)

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

    # -- Config apply (config -> view); base handles the JsonEditor wiring --

    def _apply_config(self, old, new):
        """Apply a config to the widgets/state (see BaseDataView.set_config)."""
        super()._apply_config(old, new)  # units + plot-control widgets
        self._apply_selection(new)
        self._recompute_selection()
        self._update_unit_controls()
        self._refresh_plot()
        if new.plot.auto_fetch and self._selected_objects and self._selected_aggs:
            self._trigger_load()

    def _apply_selection(self, config):
        """Rebuild the object/process trees with the config's keys checked."""
        osrc = build_object_tree_source(self._concrete, self.lang)
        self._mark_selected(osrc, set(config.object_tree.selected or []))
        self._obj_tree.set_source(osrc)
        psrc = build_process_tree_source(self._aggregated, self.lang)
        self._mark_selected(psrc, set(config.process_tree.selected or []))
        self._proc_tree.set_source(psrc)

    def _mark_selected(self, nodes, keys):
        for n in nodes:
            n["selected"] = n.get("key") in keys
            self._mark_selected(n.get("children", []), keys)

    def _config_has_state(self, config):
        return (
            bool(config.object_tree.selected)
            or bool(config.process_tree.selected)
            or bool(config.plot.unit_selections)
        )

    # -- Data loading (_trigger_load comes from BaseDataView) --

    async def _load_and_plot(self):
        limit = self._config.plot.row_limit
        ds = self._config.plot.downsample
        max_points = ds.max_points if ds.enabled else None
        edge_anchors = ds.edge_anchors if ds.enabled else None
        method_cfg = ds.method.value if ds.enabled else None
        self._traces = []

        for obj_entry in self._selected_objects:
            for agg in self._selected_aggs:
                for ctrl, ch, proc in resolve_aggregated_channel(obj_entry, agg):
                    start = to_utc(getattr(proc, "start_date_time", None))
                    end = to_utc(getattr(proc, "end_date_time", None))
                    if start is None:
                        continue
                    method = (
                        resolve_downsample_method(ch, method_cfg)
                        if method_cfg
                        else None
                    )
                    try:
                        points = await self._cache.get_data(
                            ctrl,
                            ch,
                            start,
                            end,
                            limit,
                            max_points=max_points,
                            method=method,
                            edge_anchors=edge_anchors,
                        )
                    except Exception as e:
                        _logger.error(
                            "Error loading %s/%s: %s",
                            getattr(ctrl, "name", "?"),
                            getattr(ch, "name", "?"),
                            e,
                        )
                        points = []
                    self._traces.append(
                        {
                            "object": obj_entry["object"],
                            "object_label": obj_entry["label"],
                            "process": proc,
                            "process_label": get_display_label(proc, self.lang),
                            "controller": ctrl,
                            "channel": ch,
                            "points": points,
                        }
                    )

        # t=0 per (object, process) = earliest loaded point across its channels
        self._t0 = {}
        for tr in self._traces:
            if not tr["points"]:
                continue
            key = (entity_iri(tr["object"]), entity_iri(tr["process"]))
            mn = min(to_utc(p.timestamp) for p in tr["points"])
            cur = self._t0.get(key)
            if cur is None or mn < cur:
                self._t0[key] = mn

        self._refresh_plot()

    def _make_figures(self) -> List[Any]:
        """Build a fresh list of Bokeh figures (one per group) from the traces.

        The figures are not attached to any pane/document, so this is reused
        both for live rendering and for a detached copy for HTML export (a model
        may live in only one document).
        """
        # Group traces by y-axis group, skipping text channels.
        plot_groups: Dict[str, List[Dict[str, Any]]] = {}
        for tr in self._traces:
            ch = tr["channel"]
            if resolve_value_type(ch) == "text":
                continue
            gkey = self._group_of.get(ch.uuid)
            if gkey is None:
                continue
            plot_groups.setdefault(gkey, []).append(tr)

        if not plot_groups:
            return []

        figs: List[Any] = []
        color_idx = 0
        for gkey, traces in plot_groups.items():
            axis_label = self._get_axis_label(gkey)
            fig = bk_figure(
                height=250,
                sizing_mode="stretch_width",
                x_axis_label=_pt("rel_time", self.lang),
                y_axis_label=axis_label,
            )
            lines, sources, colors, names = [], [], [], []
            for tr in traces:
                xs, ys = self._extract_trace(tr, gkey)
                if not xs:
                    continue
                # One aggregated channel fans out to every real channel each
                # sample has data on (across process runs and across datatools
                # of the same type). Label each line distinctly by
                # object · process / datatool / channel so Bokeh keeps them as
                # separate, individually-toggleable legend entries.
                label = self._trace_label(tr)
                src = ColumnDataSource(data={"x": xs, "y": ys})
                color = COLORS[color_idx % len(COLORS)]
                line = fig.line(
                    "x",
                    "y",
                    source=src,
                    name=label,
                    legend_label=label,
                    color=color,
                    line_width=2,
                )
                lines.append(line)
                sources.append(src)
                colors.append(color)
                names.append(label)
                color_idx += 1
            # Shared combined tooltip + per-series markers. The x axis is
            # relative time (numeric), so format it as a number, not a date.
            if lines:
                self._attach_combined_hover(
                    fig,
                    lines,
                    sources,
                    colors,
                    names,
                    self._group_unit_symbol(gkey),
                    x_kind="number",
                )
            fig.legend.click_policy = "hide"
            fig.legend.label_text_font_size = "8pt"
            figs.append(fig)
        return figs

    def _build_figure(self):
        self._plot_col.clear()
        self._figures = self._make_figures()
        for fig in self._figures:
            self._plot_col.append(pn.pane.Bokeh(fig, sizing_mode="stretch_width"))

    def _export_figures(self) -> List[Any]:
        """Fresh, unattached figures for HTML export (see _make_figures)."""
        return self._make_figures()

    def _trace_label(self, tr: Dict[str, Any]) -> str:
        return (
            f"{tr['object_label']} · {tr['process_label']} / "
            f"{get_display_label(tr['controller'], self.lang)} / "
            f"{get_display_label(tr['channel'], self.lang)}"
        )

    def export_series(self) -> List[Dict[str, Any]]:
        """Plotted traces as tidy records (relative-seconds x)."""
        records: List[Dict[str, Any]] = []
        for tr in self._traces:
            ch = tr["channel"]
            if resolve_value_type(ch) == "text":
                continue
            gkey = self._group_of.get(ch.uuid)
            if gkey is None:
                continue
            xs, ys = self._extract_trace(tr, gkey)
            if not xs:
                continue
            rep = next(
                (p.value for p in tr["points"] if hasattr(p.value, "to_pint")),
                None,
            )
            records.append(
                {
                    "label": self._trace_label(tr),
                    "x": xs,
                    "y": ys,
                    "x_kind": "seconds",
                    "unit": self._pint_unit(rep, self._unit_selections.get(gkey)),
                }
            )
        return records

    def _extract_trace(self, tr: Dict[str, Any], group_key: str) -> Tuple[List, List]:
        """Relative-seconds x and (unit-converted) numeric y for a trace."""
        points = tr["points"]
        if not points:
            return [], []
        key = (entity_iri(tr["object"]), entity_iri(tr["process"]))
        t0 = self._t0.get(key)
        if t0 is None:
            return [], []

        ch = tr["channel"]
        target_unit_name = self._unit_selections.get(group_key)
        xs: List[float] = []
        ys: List[Any] = []

        for pt in points:
            ts = to_utc(pt.timestamp)
            secs = (ts - t0).total_seconds()
            v = self._numeric(pt.value, ch, target_unit_name)
            if v is None:
                continue
            xs.append(secs)
            ys.append(v)

        return xs, ys

    def _update_log_console(self):
        log_entries = []
        has_text = False
        for tr in self._traces:
            ch = tr["channel"]
            if resolve_value_type(ch) != "text":
                continue
            has_text = True
            key = (entity_iri(tr["object"]), entity_iri(tr["process"]))
            t0 = self._t0.get(key)
            prefix = f"{tr['object_label']} · {tr['process_label']}"
            for pt in tr["points"]:
                ts = to_utc(pt.timestamp)
                secs = (ts - t0).total_seconds() if t0 is not None else 0.0
                val = pt.value
                if hasattr(val, "value"):
                    text = str(val.value)
                elif isinstance(val, dict):
                    text = str(val.get("value", val))
                else:
                    text = str(val)
                log_entries.append((secs, prefix, text))

        self._log_card.visible = has_text
        if not log_entries:
            self._log_pane.object = ""
            return

        log_entries.sort(key=lambda x: x[0])
        html_lines = []
        for secs, prefix, text in log_entries:
            html_lines.append(
                "<div style='font-family:monospace; font-size:12px;'>"
                f"<span style='color:#888'>+{secs:.1f}s</span> "
                f"<span style='color:#1f77b4'>[{prefix}]</span> {text}</div>"
            )
        self._log_pane.object = "\n".join(html_lines)

    # -- Layout --

    def _build_layout(self):
        if self._embeddable:
            self._app = None
            return
        self._app = Panelini(
            title=self._title,
            sidebar_enabled=True,
            sidebars_max_width=400,
        )
        self._app.sidebar_set(
            [
                self._obj_tree_card,
                self._proc_tree_card,
                self._controls_card,
            ]
        )
        self._app.main_set([self._plot_card, self._log_card])

    @property
    def sidebar_cards(self):
        """Sidebar cards (object tree, process tree, controls)."""
        return [
            self._obj_tree_card,
            self._proc_tree_card,
            self._controls_card,
        ]

    @property
    def main_cards(self):
        """Main-area cards (time series plot, log console)."""
        return [self._plot_card, self._log_card]

    # servable() / panel() come from BaseDataView.
