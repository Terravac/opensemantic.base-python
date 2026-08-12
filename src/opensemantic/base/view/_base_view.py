"""Shared building blocks for archive views.

``BaseDataView`` is a mixin (no ``__init__``) holding the parts that are
identical across the channel-centered :class:`DataToolView` and the
process/object-centered :class:`ProcessObjectView`: unit-switch controls, the
plot/log/config cards, the value→numeric unit conversion, and the
servable/panel helpers. Subclasses provide the data model and the
view-specific bits (trees, controls, ``_build_figure`` /
``_update_log_console`` / ``_load_and_plot``).

The mixin only reads attributes the subclass sets in its own ``__init__`` /
build steps (``_config``, ``_groups``, ``_unit_selections``,
``_unit_controls``, ``_plot_col``/``_plot_card``, ``_app`` ...), so it composes
cleanly without participating in ``__init__`` and without disturbing existing
subclasses (e.g. ``LiveDataToolView``).
"""

import asyncio
import io
import logging
from typing import Any, List, Tuple

import panel as pn
from bokeh.events import MouseLeave
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from bokeh.palettes import Category10_10

from opensemantic.base.view._channel_utils import (
    _get_unit_symbol_map,
    _t,
    get_available_units,
    get_unit_enum,
    resolve_characteristic_class,
    resolve_characteristic_label,
    resolve_value_type,
)
from opensemantic.base.view._config import (
    BaseViewConfig,
    DownsampleMethod,
    GroupingMode,
)

_logger = logging.getLogger(__name__)

COLORS = Category10_10

# Row cap for data export (uniform np.linspace downsampling above this).
EXPORT_MAX_ROWS = 1_000_000

# Shared-crosshair hover (see _attach_combined_hover): a HoverTool (vline, no
# native tooltip) runs this on every move. It finds the nearest point per
# series, marks them in ``hl`` (one row per series) and fills a single floating
# tooltip with one row per series - a native multi-series hover renders a
# separate box per series, which overlap into "one" when same-group channels
# share similar values. The unit comes from ``meta`` so a unit switch refreshes
# it in place. Timestamps are formatted in UTC to match the datetime axis.
_HOVER_JS = r"""
if (!window.__osHoverInit) {
  window.__osHoverMouse = {x: 0, y: 0};
  window.addEventListener('mousemove', function (e) {
    window.__osHoverMouse = {x: e.clientX, y: e.clientY};
  }, true);
  window.__osHoverInit = true;
}
let tip = document.getElementById('os-hover-tip');
const g = cb_data.geometry;
if (g == null || g.x == null) { if (tip) { tip.style.display = 'none'; } return; }
const hx = g.x;
const unit = (meta.data.unit && meta.data.unit[0]) || '';
const xkind = (meta.data.xkind && meta.data.xkind[0]) || 'datetime';
function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
  });
}
function pad(n) { return String(n).padStart(2, '0'); }
function fmtx(v) {
  if (xkind === 'datetime') {
    const d = new Date(v);
    const day = d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) +
                '-' + pad(d.getUTCDate());
    const tod = pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) +
                ':' + pad(d.getUTCSeconds());
    return day + ' ' + tod;
  }
  return Number(v).toLocaleString(undefined, {maximumFractionDigits: 3});
}
const xs = [], ys = [], cs = [];
let rows = '';
for (let s = 0; s < sources.length; s++) {
  const data = sources[s].data;
  const X = data['x'], Y = data['y'];
  if (!X || X.length === 0) { continue; }
  let best = 0, bestd = Infinity;
  for (let i = 0; i < X.length; i++) {
    const d = Math.abs(X[i] - hx);
    if (d < bestd) { bestd = d; best = i; }
  }
  xs.push(X[best]); ys.push(Y[best]); cs.push(colors[s]);
  const val = Number(Y[best]).toLocaleString(undefined, {maximumFractionDigits: 6});
  rows += '<div class="os-tip-row" style="white-space:nowrap">' +
          '<span style="color:' + colors[s] + '">●</span> ' +
          esc(names[s]) + ': <b>' + val + (unit ? ' ' + esc(unit) : '') + '</b>' +
          ' <span style="opacity:.6">' + fmtx(X[best]) + '</span></div>';
}
hl.data = {x: xs, y: ys, c: cs};
hl.change.emit();
if (rows === '') { if (tip) { tip.style.display = 'none'; } return; }
if (!tip) {
  tip = document.createElement('div');
  tip.id = 'os-hover-tip';
  tip.style.cssText =
    'position:fixed;z-index:10000;background:rgba(255,255,255,0.96);' +
    'border:1px solid #bbb;border-radius:3px;padding:4px 8px;font:12px sans-serif;' +
    'color:#222;pointer-events:none;box-shadow:0 1px 4px rgba(0,0,0,0.2);';
  document.body.appendChild(tip);
}
tip.innerHTML = rows;
tip.style.display = 'block';
const m = window.__osHoverMouse;
tip.style.left = (m.x + 14) + 'px';
tip.style.top = (m.y + 14) + 'px';
"""

# Clear the marker and hide the floating tooltip when the pointer leaves a plot.
_HOVER_LEAVE_JS = r"""
hl.data = {x: [], y: [], c: []};
hl.change.emit();
const tip = document.getElementById('os-hover-tip');
if (tip) { tip.style.display = 'none'; }
"""

# Optional dependencies for unit-aware data export (opensemantic.base[export]).
try:
    import pandas as _pd  # noqa: F401
    import pint_pandas as _pint_pandas  # noqa: F401

    _EXPORT_DEPS_OK = True
except Exception:  # pragma: no cover - optional
    _EXPORT_DEPS_OK = False


def _series_to_dataframe(series: List[dict], max_rows: int):
    """Tidy series records -> a DataFrame (one column per series).

    Each record ``{"label", "x", "y", "x_kind", "unit"}`` becomes a column,
    aligned (outer join) on the x index. Numeric records with a unit get a
    ``pint[<unit>]`` dtype; a record with no unit (e.g. a text-log channel) or
    non-numeric values becomes a plain object column, so all checked channels -
    not just the plotted numeric ones - are exportable. Rows are capped at
    ``max_rows`` via uniform np.linspace downsampling (logged). Returns ``None``
    if pandas/pint-pandas are missing or there is nothing to export.
    """
    try:
        import numpy as np
        import pandas as pd
        import pint_pandas  # noqa: F401
    except ImportError:
        _logger.warning(
            "Data export needs the 'export' extra (pandas, pint-pandas, openpyxl)."
        )
        return None
    if not series:
        return None

    parts = []
    seen = {}
    for s in series:
        idx = pd.Index(s["x"], name=s.get("x_kind") or "x")
        unit = s.get("unit")
        col = None
        if unit:
            try:
                col = pd.Series(s["y"], index=idx, dtype=f"pint[{unit}]")
            except Exception:
                col = None
        if col is None:
            # No unit (text/log) or an unparseable one: numerics fall back to a
            # dimensionless pint column, text values to a plain object column.
            try:
                col = pd.Series(s["y"], index=idx, dtype="pint[dimensionless]")
            except Exception:
                col = pd.Series(s["y"], index=idx, dtype="object")
        # A well-defined outer join needs a unique index per column.
        col = col[~col.index.duplicated(keep="last")]
        # Unique column names: pandas returns a DataFrame (not a Series) for a
        # duplicated label, which would break the per-column dequantify.
        label = s["label"]
        if label in seen:
            seen[label] += 1
            label = f"{label} ({seen[label]})"
        else:
            seen[label] = 0
        col.name = label
        parts.append(col)
    if not parts:
        return None

    df = pd.concat(parts, axis=1).sort_index()
    n = len(df)
    if n > max_rows:
        sel = np.linspace(0, n - 1, max_rows).astype(int)
        df = df.iloc[sel]
        _logger.info("Data export capped from %d to %d rows.", n, max_rows)
    return df


def _dequantify(df):
    """Like ``df.pint.dequantify()`` but tolerant of non-pint (text) columns.

    Produces a two-level column header ``(label, unit)`` so a dedicated unit
    header row is written on export and the column keys stay unit-free. Pint
    columns contribute their magnitude and unit; plain (text) columns keep their
    values under an empty unit.
    """
    import pandas as pd

    tuples = []
    mags = []
    for name in df.columns:
        s = df[name]
        if str(s.dtype).startswith("pint"):
            tuples.append((str(name), str(s.pint.units)))
            mags.append(s.pint.magnitude)
        else:
            tuples.append((str(name), ""))
            mags.append(s)
    out = pd.concat(mags, axis=1)
    out.columns = pd.MultiIndex.from_tuples(tuples, names=[None, "unit"])
    return out


class BaseDataView:
    """Mixin with the UI/plot pieces shared by the archive views.

    A concrete view owns its config class via the ``config_cls`` attribute (a
    ``BaseViewConfig`` subclass) and provides the config->view ``_apply_config``
    hook; ``set_config`` / ``get_config`` / ``on_config_change`` are inherited.
    """

    #: The config class this view owns (overridden by each concrete view).
    config_cls = BaseViewConfig

    @property
    def lang(self) -> str:
        return self._config.lang

    # -- Unit-switch controls (one dropdown per quantity group) --

    def _update_unit_controls(self):
        self._unit_controls.clear()
        for group_key, channels in self._groups.items():
            if not channels:
                continue
            sample_ch = channels[0][1]
            if resolve_value_type(sample_ch) != "quantity":
                continue
            units = get_available_units(sample_ch)
            if not units:
                continue
            char_label = resolve_characteristic_label(sample_ch, self.lang)
            current = self._unit_selections.get(group_key)
            options = {u["symbol"]: u["name"] for u in units}
            if current not in options.values():
                current = next(iter(options.values()))
                self._unit_selections[group_key] = current
            dropdown = pn.widgets.Select(
                name=f"{_t('unit', self.lang)}: {char_label}",
                options=options,
                value=current,
            )
            dropdown.param.watch(
                lambda event, _key=group_key: self._on_unit_change(_key, event),
                ["value"],
            )
            self._unit_controls.append(dropdown)

    def _on_unit_change(self, group_key: str, event):
        if getattr(self, "_applying_config", False):
            return
        self._unit_selections[group_key] = event.new
        self._config.plot.unit_selections = dict(self._unit_selections)
        self._refresh_plot()
        self._emit_config_change()

    # -- Shared plot-control widgets (a native field per config property) ----
    #
    # Every ``PlotControlsConfig`` property is editable through a native widget
    # here, so no JSON editor is needed. Each writes through to ``self._config``
    # and emits a change; ``_apply_config`` sets them back. Views place these in
    # their controls card and supply the ``_regroup`` / ``_has_active_selection``
    # hooks.

    def _build_grouping_control(self):
        options = {
            _t("group_none", self.lang): GroupingMode.NONE.value,
            _t("group_unique", self.lang): GroupingMode.UNIQUE.value,
            _t("group_sub", self.lang): GroupingMode.SUB.value,
        }
        self._grouping_select = pn.widgets.Select(
            name=_t("grouping", self.lang),
            options=options,
            value=self._config.plot.grouping.value,
        )
        self._grouping_select.param.watch(self._on_grouping_change, ["value"])
        return self._grouping_select

    def _on_grouping_change(self, event):
        if getattr(self, "_applying_config", False):
            return
        self._config.plot.grouping = GroupingMode(event.new)
        self._regroup()
        self._update_unit_controls()
        self._refresh_plot()
        self._emit_config_change()

    def _build_cache_control(self):
        self._cache_cb = pn.widgets.Checkbox(
            name=_t("cache_enabled", self.lang),
            value=self._config.plot.cache_enabled,
        )
        self._cache_cb.param.watch(self._on_cache_enabled_change, ["value"])
        return self._cache_cb

    def _on_cache_enabled_change(self, event):
        if getattr(self, "_applying_config", False):
            return
        self._config.plot.cache_enabled = event.new
        if getattr(self, "_cache", None) is not None:
            self._cache.enabled = event.new
        self._emit_config_change()

    def _build_downsample_controls(self):
        ds = self._config.plot.downsample
        self._ds_enabled = pn.widgets.Checkbox(
            name=_t("downsample", self.lang), value=ds.enabled
        )
        self._ds_max_points = pn.widgets.IntInput(
            name=_t("max_points", self.lang), value=ds.max_points, start=2, step=100
        )
        self._ds_method = pn.widgets.Select(
            name=_t("ds_method", self.lang),
            options={_t("ds_" + m.value, self.lang): m.value for m in DownsampleMethod},
            value=ds.method.value,
        )
        self._ds_edge = pn.widgets.Checkbox(
            name=_t("edge_anchors", self.lang), value=ds.edge_anchors
        )
        self._ds_enabled.param.watch(
            lambda e: self._on_downsample_change("enabled", e.new), ["value"]
        )
        self._ds_max_points.param.watch(
            lambda e: self._on_downsample_change("max_points", e.new), ["value"]
        )
        self._ds_method.param.watch(
            lambda e: self._on_downsample_change("method", e.new), ["value"]
        )
        self._ds_edge.param.watch(
            lambda e: self._on_downsample_change("edge_anchors", e.new), ["value"]
        )
        self._downsample_card = pn.Card(
            pn.Column(
                self._ds_enabled,
                self._ds_max_points,
                self._ds_method,
                self._ds_edge,
                sizing_mode="stretch_width",
            ),
            title=_t("downsample", self.lang),
            collapsed=True,
            sizing_mode="stretch_width",
            margin=(6, 5),
        )
        return self._downsample_card

    def _on_downsample_change(self, field, value):
        if getattr(self, "_applying_config", False):
            return
        if field == "method":
            value = DownsampleMethod(value)
        setattr(self._config.plot.downsample, field, value)
        self._emit_config_change()
        if self._config.plot.auto_fetch and self._has_active_selection():
            self._trigger_load()

    def _regroup(self):  # pragma: no cover - view hook
        """Recompute the channel grouping after a grouping-mode change."""

    def _has_active_selection(self) -> bool:  # pragma: no cover - view hook
        """Whether anything is selected (controls whether a change reloads)."""
        return False

    def _apply_plot_control_widgets(self, config):
        """Set the shared plot-control widgets from the config (guarded)."""
        if getattr(self, "_grouping_select", None) is not None:
            self._grouping_select.value = config.plot.grouping.value
        if getattr(self, "_cache_cb", None) is not None:
            self._cache_cb.value = config.plot.cache_enabled
        ds = config.plot.downsample
        for attr, name in (
            ("enabled", "_ds_enabled"),
            ("max_points", "_ds_max_points"),
            ("edge_anchors", "_ds_edge"),
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.value = getattr(ds, attr)
        if getattr(self, "_ds_method", None) is not None:
            self._ds_method.value = ds.method.value

    # -- Plot / log / config cards --

    def _build_plot(self):
        # No height cap / inner scroll here: the plots stack at their natural
        # height and the main content area (.content) scrolls as one. A fixed
        # max_height + scroll=True would add a second, nested scrollbar.
        self._plot_col = pn.Column(sizing_mode="stretch_width")
        self._plot_card = pn.Card(
            self._plot_col,
            title=_t("time_series", self.lang),
            sizing_mode="stretch_width",
        )

    # -- Export (data + plot) --

    _figures: List[Any] = []

    @property
    def figures(self) -> List[Any]:
        """The Bokeh figures currently rendered (updated by _build_figure).

        Exposed so host apps can add their own annotations; also used by the
        HTML plot export. Empty when nothing is plotted.
        """
        return list(getattr(self, "_figures", []) or [])

    def _build_export_toolbar(self):
        """Compact collapsed "Export" card of FileDownload buttons.

        A dashboard-level action (data CSV/XLSX + plot HTML), so it lives in the
        Plot Controls sidebar rather than inside the Time Series group. Returns
        the card and stores it on ``self._export_box`` for host embedding.
        """
        deps_note = "" if _EXPORT_DEPS_OK else " (needs opensemantic.base[export])"
        self._download_csv = pn.widgets.FileDownload(
            label=_t("download_csv", self.lang) + deps_note,
            filename="dashboard_data.csv",
            callback=lambda: self._build_data_export("csv"),
            button_type="default",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._download_xlsx = pn.widgets.FileDownload(
            label=_t("download_xlsx", self.lang) + deps_note,
            filename="dashboard_data.xlsx",
            callback=lambda: self._build_data_export("xlsx"),
            button_type="default",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._download_html = pn.widgets.FileDownload(
            label=_t("download_plot", self.lang),
            filename="dashboard_plot.html",
            callback=self._build_plot_html,
            button_type="default",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._export_box = pn.Card(
            pn.Column(
                self._download_csv,
                self._download_xlsx,
                self._download_html,
                sizing_mode="stretch_width",
            ),
            title=_t("export", self.lang),
            collapsed=True,
            sizing_mode="stretch_width",
            # Separate this tile from the controls above and the next section.
            margin=(10, 5),
        )
        return self._export_box

    def _has_export_log(self) -> bool:
        """Whether the log console currently has content to include in HTML."""
        pane = getattr(self, "_log_pane", None)
        card = getattr(self, "_log_card", None)
        return bool(getattr(pane, "object", "")) and bool(
            getattr(card, "visible", False)
        )

    def _update_export_state(self):
        """Enable/disable export buttons based on what is available.

        Data (CSV/XLSX) covers every checked channel, so it keys off
        ``export_series`` (numeric + text). The plot HTML covers the rendered
        figures and the log console, so it keys off either being present.
        """
        has_data = bool(self.export_series())
        has_html = bool(self.figures) or self._has_export_log()
        for w in ("_download_csv", "_download_xlsx"):
            btn = getattr(self, w, None)
            if btn is not None:
                btn.disabled = not (has_data and _EXPORT_DEPS_OK)
        if getattr(self, "_download_html", None) is not None:
            self._download_html.disabled = not has_html

    def _build_data_export(self, fmt: str) -> io.BytesIO:
        """Build a unit-aware CSV/XLSX of the plotted series via pint-pandas.

        Serialized with the pint-pandas ``dequantify`` accessor so a dedicated
        unit header row is written and the column keys stay unit-free. Returns a
        (possibly empty) BytesIO.
        """
        buf = io.BytesIO()
        cap = int(getattr(self, "EXPORT_MAX_ROWS", EXPORT_MAX_ROWS))
        df = _series_to_dataframe(self.export_series(), cap)
        if df is None or df.empty:
            return buf
        import pandas as pd

        dequantified = _dequantify(df)
        if fmt == "xlsx":
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    dequantified.to_excel(writer)
            except ImportError:
                _logger.warning("XLSX export needs openpyxl (base[export]).")
                return io.BytesIO()
        else:
            buf.write(dequantified.to_csv().encode("utf-8"))
        buf.seek(0)
        return buf

    def _export_figures(self) -> List[Any]:
        """Figures to serialize for HTML export.

        Defaults to the live ``figures``; views whose figures are attached to a
        Panel document override this to return a fresh, unattached copy, since
        ``file_html`` requires models that belong to no other document.
        """
        return self.figures

    #: Fixed frame size for HTML-exported figures.
    EXPORT_PLOT_WIDTH = 1000
    EXPORT_PLOT_HEIGHT = 350

    def _export_extra_models(self) -> List[Any]:
        """Extra Bokeh models to append below the figures in the HTML export.

        Includes the log console (as an HTML ``Div``) when it has content, so a
        dashboard with text-log channels carries them into the export too.
        """
        models: List[Any] = []
        if self._has_export_log():
            from bokeh.models import Div

            # Scrollable inner container so a long log does not stretch the page;
            # the heading stays fixed above it.
            width = self.EXPORT_PLOT_WIDTH
            models.append(
                Div(
                    text=(
                        f"<h3 style='margin:0 0 4px'>{_t('log_console', self.lang)}"
                        f"</h3>"
                        # Explicit width on the box: Bokeh's Div content wrapper
                        # is content-sized, so match the fixed plot width here.
                        f"<div style='width:{width}px;box-sizing:border-box;"
                        f"max-height:300px;overflow-y:auto;"
                        f"border:1px solid #ccc;padding:4px'>"
                        f"{self._log_pane.object}</div>"
                    ),
                    width=width,
                    sizing_mode="fixed",
                )
            )
        return models

    def _build_plot_html(self) -> io.BytesIO:
        """Standalone interactive HTML of the figures + log console."""
        buf = io.BytesIO()
        figs = self._export_figures()
        extras = self._export_extra_models()
        if not figs and not extras:
            return buf
        from bokeh.embed import file_html
        from bokeh.layouts import column as bk_column
        from bokeh.resources import CDN

        # The live figures are stretch_width, but a standalone HTML has no
        # container width to stretch into, so the plot frame collapses to zero
        # width. Pin an explicit size on the export copies so they render.
        for fig in figs:
            try:
                fig.sizing_mode = "fixed"
                fig.width = self.EXPORT_PLOT_WIDTH
                fig.height = self.EXPORT_PLOT_HEIGHT
            except Exception:  # pragma: no cover - non-figure layout
                pass

        roots = list(figs) + list(extras)
        root = roots[0] if len(roots) == 1 else bk_column(*roots)
        html = file_html(root, CDN, getattr(self, "_title", "Plot"))
        buf.write(html.encode("utf-8"))
        buf.seek(0)
        return buf

    def _build_log_console(self):
        self._log_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            height=200,
            styles={"overflow-y": "auto"},
        )
        self._log_card = pn.Card(
            self._log_pane,
            title=_t("log_console", self.lang),
            collapsed=False,
            visible=False,
            sizing_mode="stretch_width",
        )

    # -- Config binding (bidirectional) -------------------------------------
    #
    # The config is the single, JSON-serializable source of truth. User edits
    # write through into ``self._config`` and emit a change; programmatic / URL
    # updates come in via ``set_config`` and are applied to the widgets. A
    # single ``_applying_config`` flag guards against feedback loops.

    @classmethod
    def _coerce_config(cls, config):
        """Return a config of this view's ``config_cls``.

        ``None`` yields a default; a config that is already the right class (or a
        subclass) passes through; any other ``BaseViewConfig`` is upgraded by
        re-validating its dump, so passing a base config to a concrete view keeps
        working (missing component fields take their defaults).
        """
        if config is None:
            return cls.config_cls()
        if isinstance(config, cls.config_cls):
            return config
        return cls.config_cls.model_validate(config.model_dump())

    def get_config(self):
        """Return the current config instance (the concrete subclass)."""
        return self._config

    def set_config(self, config) -> None:
        """Apply a config as the new state - the single apply path.

        Used by URL sync and programmatic/host callers. Pushes every field into
        the widgets/state (guarded so it does not loop back), then notifies
        ``on_config_change`` listeners.
        """
        old = getattr(self, "_config", None)
        self._applying_config = True
        try:
            self._config = config
            self._config_cls = type(config)
            self._apply_config(old, config)
        finally:
            self._applying_config = False
        self._notify_config_change()

    def on_config_change(self, callback) -> None:
        """Register a callback invoked with the config on any change.

        A host that composes several views uses this to keep its aggregate
        (parent) config in sync with each view's sub-config.
        """
        self._config_change_cbs.append(callback)

    @property
    def _config_change_cbs(self) -> List[Any]:
        cbs = getattr(self, "_config_change_cbs_list", None)
        if cbs is None:
            cbs = []
            self._config_change_cbs_list = cbs
        return cbs

    def _emit_config_change(self) -> None:
        """Called after a user-driven write-through mutates ``self._config``."""
        if getattr(self, "_applying_config", False):
            return
        self._notify_config_change()

    def _notify_config_change(self) -> None:
        for cb in list(self._config_change_cbs):
            try:
                cb(self._config)
            except Exception as e:  # pragma: no cover - host callback guard
                _logger.error("on_config_change callback failed: %s", e)

    def _apply_initial_config(self) -> None:
        """Render any non-default state carried by the initial config.

        Called at the end of a view's ``__init__`` so a config passed in with a
        tree source / units / time range is reflected in the UI immediately.
        """
        if self._config_has_state(self._config):
            self.set_config(self._config)

    def _config_has_state(self, config) -> bool:
        """Whether the initial config carries non-default UI state to restore.

        Returning ``True`` makes ``__init__`` call ``set_config(config)`` once,
        so the view opens already reflecting a saved tree selection, unit choices
        or time window (and, if auto-fetch is on, loads that data). Returning
        ``False`` skips that initial apply for a fresh/default config, avoiding an
        unnecessary tree rebuild and data fetch. Views override to also inspect
        their own tree component(s); the base checks the shared unit selections.
        """
        return bool(getattr(config.plot, "unit_selections", None))

    def _enable_url_sync(self, param_name: str = "config", mode=None) -> None:
        """Opt-in: bind this view's config to the browser URL.

        Loads the config from the URL when a session is ready (if present) and
        writes it back on every change, using ``mode`` (default
        ``COMPRESSED_BASE64``). Default off, so a host that owns its own
        persistence - e.g. by composing several views into one parent config -
        is unaffected; that host would URL-sync the parent instead.
        """
        from opensemantic.base.view.url_config import UrlConfig, UrlConfigMode

        if mode is None:
            mode = UrlConfigMode.COMPRESSED_BASE64
        self._url_config = UrlConfig(self._config_cls, param_name=param_name)

        def _load():
            try:
                if self._url_config.has_config():
                    self.set_config(self._url_config.get_config())
            except Exception as e:  # pragma: no cover - URL is best effort
                _logger.warning("URL config load failed: %s", e)
            self.on_config_change(lambda cfg: self._url_config.set_config(cfg, mode))

        # pn.state.location is only populated once a session loads, so defer.
        try:
            pn.state.onload(_load)
        except Exception:  # pragma: no cover - no active session (tests)
            _load()

    def _apply_config(self, old, new) -> None:
        """Push shared config fields into the widgets/state.

        Runs inside ``set_config`` with ``_applying_config`` set, so widget
        watchers short-circuit. Subclasses override to also apply their
        selection / time range / grouping and to refresh, calling ``super()``.
        """
        self._unit_selections = dict(getattr(new.plot, "unit_selections", {}) or {})
        if getattr(self, "_auto_fetch_cb", None) is not None:
            self._auto_fetch_cb.value = new.plot.auto_fetch
        if getattr(self, "_row_limit_input", None) is not None:
            self._row_limit_input.value = new.plot.row_limit
        if getattr(self, "_cache", None) is not None:
            self._cache.enabled = new.plot.cache_enabled
        self._apply_plot_control_widgets(new)

    # -- Plot helpers --

    def _refresh_plot(self):
        """Rebuild plot and log console from the subclass's loaded data."""
        self._build_figure()
        self._update_log_console()
        self._update_export_state()
        self._set_plot_loading(False)

    def _set_plot_loading(self, flag):
        """Toggle the plot column's loading spinner (safe if not yet built)."""
        col = getattr(self, "_plot_col", None)
        if col is not None:
            try:
                col.loading = bool(flag)
            except Exception:  # noqa: BLE001 - loading is cosmetic
                pass

    def _numeric(self, value: Any, channel: Any, target_unit_name: Any) -> Any:
        """Convert a value to its display unit and return the numeric scalar.

        Handles typed Characteristic instances (with ``to_unit``), raw dicts
        and bare scalars. Returns ``None`` when no value can be extracted.
        """
        if target_unit_name and hasattr(value, "to_unit"):
            unit_enum = get_unit_enum(channel)
            if unit_enum is not None and target_unit_name in unit_enum.__members__:
                try:
                    value = value.to_unit(unit_enum[target_unit_name])
                except Exception:
                    pass
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, dict):
            return value.get("value")
        return value

    def _pint_unit(self, value: Any, target_unit_name: Any) -> str:
        """Pint-parseable unit string for a typed value in its display unit.

        Converts the value to ``target_unit_name`` (like the plot) and reads the
        pint unit off ``to_pint()``. Falls back to ``"dimensionless"`` for raw /
        untyped values so the export column is always a valid pint dtype.
        """
        if value is None or not hasattr(value, "to_pint"):
            return "dimensionless"
        try:
            if target_unit_name and hasattr(value, "to_unit"):
                unit = getattr(value, "unit", None)
                enum = type(unit) if unit is not None else None
                if enum is not None and hasattr(enum, "__members__"):
                    if target_unit_name in enum.__members__:
                        value = value.to_unit(enum[target_unit_name])
            return str(value.to_pint().units)
        except Exception:
            return "dimensionless"

    def _group_unit_symbol(self, group_key: str) -> str:
        """Display unit symbol for a group (e.g. ``K``), or ``""`` if none."""
        channels = self._groups.get(group_key, [])
        if not channels:
            return ""
        sample_ch = channels[0][1]

        unit_name = self._unit_selections.get(group_key)
        if unit_name:
            for u in get_available_units(sample_ch):
                if u["name"] == unit_name:
                    return u["symbol"]

        ch_unit = getattr(sample_ch, "unit", None)
        if ch_unit:
            cls = resolve_characteristic_class(sample_ch)
            symbol_map = _get_unit_symbol_map(cls)
            unit_enum = get_unit_enum(sample_ch)
            if unit_enum:
                for member in unit_enum:
                    if member.value == ch_unit or member.name == ch_unit:
                        return symbol_map.get(member.name, member.name)

        return ""

    def _get_axis_label(self, group_key: str) -> str:
        """Build y-axis label: characteristic name [unit symbol]."""
        channels = self._groups.get(group_key, [])
        if not channels:
            return ""
        char_label = resolve_characteristic_label(channels[0][1], self.lang)
        symbol = self._group_unit_symbol(group_key)
        return f"{char_label} [{symbol}]" if symbol else char_label

    def _attach_combined_hover(
        self, fig, lines, sources, colors, names, unit_symbol="", x_kind="datetime"
    ):
        """Attach the shared-crosshair hover to ``fig`` (see _HOVER_JS).

        One combined tooltip with a row per series and one nearest-point marker
        per series. ``lines``/``sources``/``colors``/``names`` are per-series and
        aligned. ``x_kind`` is ``"datetime"`` (x formatted as a UTC timestamp) or
        ``"number"`` (e.g. relative time). Returns the ``meta`` ColumnDataSource
        carrying the group's unit symbol and x-kind, so a caller can refresh the
        unit in place on a unit switch.
        """
        hl_src = ColumnDataSource(data={"x": [], "y": [], "c": []})
        # Drawn after the lines so the highlighted points sit on top.
        fig.scatter(
            "x",
            "y",
            source=hl_src,
            size=9,
            marker="circle",
            fill_color="c",
            line_color="c",
            fill_alpha=1.0,
            line_alpha=1.0,
        )
        meta_src = ColumnDataSource(
            data={"unit": [unit_symbol or ""], "xkind": [x_kind]}
        )
        hover = HoverTool(renderers=list(lines), mode="vline", tooltips=None)
        hover.callback = CustomJS(
            args={
                "hl": hl_src,
                "meta": meta_src,
                "sources": list(sources),
                "colors": list(colors),
                "names": list(names),
            },
            code=_HOVER_JS,
        )
        fig.add_tools(hover)
        fig.js_on_event(MouseLeave, CustomJS(args={"hl": hl_src}, code=_HOVER_LEAVE_JS))
        return meta_src

    # -- Data loading entry point (async-context aware) --

    def _trigger_load(self):
        """Start data loading, handling sync vs running-loop contexts."""
        # Show the spinner synchronously so it is visible even for the initial
        # load kicked off during page build (e.g. a URL-restored selection),
        # not only once the async task starts. _refresh_plot clears it.
        self._set_plot_loading(True)
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(self._load_and_plot())
        except RuntimeError:
            asyncio.run(self._load_and_plot())

    # -- Serving --

    def servable(self, **kwargs):
        if self._app is None:
            raise RuntimeError(
                f"{type(self).__name__}(embeddable=True) has no servable app; "
                "place sidebar_cards / main_cards into a host app instead."
            )
        return self._app.servable(**kwargs)

    def panel(self):
        """Return the Panelini app for embedding (None when embeddable)."""
        return self._app

    # -- Subclass contract (implemented by each view) --

    def _build_figure(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def export_series(self) -> List[dict]:  # pragma: no cover - overridden
        """Return the currently plotted series as tidy records.

        Each record: ``{"label", "x", "y", "x_kind": "datetime"|"seconds",
        "unit"}`` where ``unit`` is a pint-parseable string. Built by reusing
        the same trace extraction as ``_build_figure`` so the export matches
        exactly what is plotted (unit conversion + normalization).
        """
        raise NotImplementedError

    def _update_log_console(self):  # pragma: no cover - overridden
        raise NotImplementedError

    async def _load_and_plot(self):  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def sidebar_cards(self) -> List[Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def main_cards(self) -> List[Any]:  # pragma: no cover - overridden
        raise NotImplementedError


# Re-exported helper type for subclasses that annotate group maps.
GroupList = List[Tuple[Any, Any]]
