"""Pure utility functions for the DataTool dashboard UI.

Converts DataToolController instances into Wunderbaum tree source data,
resolves characteristic metadata, handles unit discovery and grouping.
No UI dependencies (no Panel/Plotly imports).
"""

import datetime as dt
import typing
from typing import Any, Dict, List, Optional, Tuple

from opensemantic.base.view._config import GroupingMode


def to_utc(value: Any) -> Optional[dt.datetime]:
    """Coerce a datetime (or ISO string) to tz-aware UTC; naive is read as UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value


# -- Display label helpers --


def get_display_label(obj: Any, lang: str = "en") -> str:
    """Get a human-readable display label from an OswBaseModel instance.

    Fallback chain: label matching lang -> label "en" -> first label -> name -> ""
    """
    labels = getattr(obj, "label", None)
    if labels and isinstance(labels, list) and len(labels) > 0:
        # Try matching lang
        for lbl in labels:
            lbl_lang = getattr(lbl, "lang", None)
            if lbl_lang == lang:
                text = getattr(lbl, "text", None)
                if text:
                    return text
        # Fallback to "en"
        if lang != "en":
            for lbl in labels:
                lbl_lang = getattr(lbl, "lang", None)
                if lbl_lang == "en":
                    text = getattr(lbl, "text", None)
                    if text:
                        return text
        # Fallback to first label
        text = getattr(labels[0], "text", None)
        if text:
            return text
    # Fallback to name
    name = getattr(obj, "name", None)
    if name:
        return str(name)
    return ""


def get_display_label_cls(cls: type, lang: str = "en") -> str:
    """Get a human-readable display label from a model class.

    Uses json_schema_extra: title*[lang] -> title*["en"] -> first title* value
    -> title -> ""
    """
    extra = _get_schema_extra(cls)
    if not extra:
        return ""
    title_star = extra.get("title*", {})
    if isinstance(title_star, dict):
        if lang in title_star:
            return title_star[lang]
        if lang != "en" and "en" in title_star:
            return title_star["en"]
        # First value
        vals = list(title_star.values())
        if vals:
            return vals[0]
    title = extra.get("title", "")
    return title or ""


def _get_schema_extra(cls: type) -> dict:
    """Extract json_schema_extra from a model class (v1 and v2)."""
    if hasattr(cls, "model_config"):
        config = cls.model_config
        if isinstance(config, dict):
            return config.get("json_schema_extra", {})
    # Pydantic v1 fallback
    config_cls = getattr(cls, "__config__", None) or getattr(cls, "Config", None)
    if config_cls is not None:
        return getattr(config_cls, "schema_extra", {}) or {}
    return {}


# -- Characteristic resolution --


def get_characteristic_iri(channel: Any) -> Optional[str]:
    """Extract the characteristic IRI string from a DataChannel.

    Uses __iris__ to avoid triggering backend resolution.
    """
    iris = getattr(channel, "__iris__", {})
    char_iri = iris.get("characteristic")
    if char_iri is None:
        try:
            char_iri = getattr(channel, "characteristic", None)
        except (ValueError, ImportError):
            return None
    if isinstance(char_iri, list):
        char_iri = char_iri[0] if char_iri else None
    if char_iri is None:
        return None
    if isinstance(char_iri, type):
        # Already a class - get its IRI
        if hasattr(char_iri, "get_cls_iri"):
            return char_iri.get_cls_iri()
        return None
    return str(char_iri) if char_iri else None


def resolve_characteristic_class(channel: Any) -> Optional[type]:
    """Resolve the characteristic IRI to a Python class via the _types registry."""
    iri = get_characteristic_iri(channel)
    if iri is None:
        return None
    if isinstance(iri, type):
        return iri
    try:
        from oold.model import _types

        cls = _types.get(iri)
        if cls is not None:
            return cls
        # Also check v1 registry (v1 models register separately)
        from oold.model.v1 import _types as _v1_types

        return _v1_types.get(iri)
    except ImportError:
        return None


def resolve_characteristic_label(channel: Any, lang: str = "en") -> str:
    """Get a human-readable characteristic label for a channel."""
    cls = resolve_characteristic_class(channel)
    if cls is not None:
        return get_display_label_cls(cls, lang)
    iri = get_characteristic_iri(channel)
    return str(iri) if iri else ""


# -- Unit resolution --


def _unwrap_optional(annotation: Any) -> Any:
    """Extract the non-None type from Optional[T] / Union[T, None] / T | None."""
    args = typing.get_args(annotation)
    if args:
        for a in args:
            if a is not None and a is not type(None):
                return a
    return annotation


def get_unit_enum(channel: Any) -> Optional[type]:
    """Resolve the UnitEnum type for a channel's characteristic.

    Returns the enum class (e.g., TemperatureUnit) or None.
    """
    cls = resolve_characteristic_class(channel)
    if cls is None:
        return None
    fields = getattr(cls, "model_fields", {})
    unit_field = fields.get("unit")
    if unit_field is None:
        return None
    unit_type = _unwrap_optional(unit_field.annotation)
    if unit_type is None or unit_type is type(None):
        return None
    # Check it's actually an enum
    import enum

    if isinstance(unit_type, type) and issubclass(unit_type, enum.Enum):
        return unit_type
    return None


def get_unit_enum_from_value(value: Any) -> Optional[type]:
    """Resolve the UnitEnum type from a typed value instance (via its ``unit``)."""
    unit = getattr(value, "unit", None)
    if unit is not None:
        return type(unit) if hasattr(type(unit), "__members__") else None
    return None


def align_pint_pandas_registry() -> bool:
    """Point pint-pandas at pint's application registry.

    ``pint[<unit>]`` dtypes must resolve the same (possibly custom OSW) units the
    Characteristic classes use via ``pint.get_application_registry()``. Called
    once at import; returns False when pandas/pint-pandas are unavailable.
    """
    try:
        import pint
        import pint_pandas
    except Exception:
        return False
    try:
        pint_pandas.PintType.ureg = pint.get_application_registry()
    except Exception:
        pass
    return True


def to_display_magnitudes(raw_values: Any, source_unit: Any, target_unit: Any) -> Any:
    """Vectorized unit conversion of a numeric series to its display unit.

    ``raw_values`` are assumed to be in ``source_unit``; returns a numpy float
    array of magnitudes in ``target_unit``, converted in a single pint-pandas
    operation. Falls back to the raw floats when pandas/pint-pandas are missing,
    when the units match, or on any conversion error (e.g. an unparseable or
    prefixed-offset unit), so a bad unit never blanks a plot.
    """
    import numpy as np

    arr = np.asarray(raw_values, dtype="float64")
    if not source_unit or not target_unit or source_unit == target_unit:
        return arr
    try:
        import pandas as pd
        import pint_pandas  # noqa: F401

        series = pd.Series(arr, dtype=f"pint[{source_unit}]")
        converted = series.pint.to(target_unit).pint.magnitude
        return np.asarray(converted, dtype="float64")
    except Exception:
        return arr


align_pint_pandas_registry()


def get_available_units(channel: Any) -> List[Dict[str, str]]:
    """Get all available units for a channel's characteristic.

    Returns a list of {name: enum_member_name, symbol: display_symbol, value: iri}
    for populating unit switch dropdowns.
    Reads symbols from the unit field's enum_titles in json_schema_extra.
    """
    unit_enum = get_unit_enum(channel)
    if unit_enum is None:
        return []

    # Get symbol mapping from the characteristic class's unit field metadata
    cls = resolve_characteristic_class(channel)
    symbol_map = _get_unit_symbol_map(cls)

    result = []
    for member in unit_enum:
        symbol = symbol_map.get(member.name, member.name.replace("_", " "))
        result.append(
            {
                "name": member.name,
                "symbol": symbol,
                "value": member.value if hasattr(member, "value") else member.name,
            }
        )
    return result


def _get_unit_symbol_map(cls: Optional[type]) -> Dict[str, str]:
    """Build a mapping from enum member name to display symbol.

    Reads from the unit field's enum_titles and x_enum_varnames in field metadata.
    """
    if cls is None:
        return {}

    # Pydantic v1: field_info.extra
    v1_fields = getattr(cls, "__fields__", {})
    unit_field = v1_fields.get("unit")
    if unit_field is not None:
        extra = getattr(unit_field, "field_info", None)
        if extra is not None:
            extra_dict = getattr(extra, "extra", {})
            titles = (extra_dict.get("options", {}) or {}).get("enum_titles", [])
            varnames = extra_dict.get("x_enum_varnames", [])
            if titles and varnames and len(titles) == len(varnames):
                return dict(zip(varnames, titles))

    # Pydantic v2: json_schema_extra on FieldInfo
    v2_fields = getattr(cls, "model_fields", {})
    unit_field_v2 = v2_fields.get("unit")
    if unit_field_v2 is not None:
        extra = getattr(unit_field_v2, "json_schema_extra", None) or {}
        titles = (extra.get("options", {}) or {}).get("enum_titles", [])
        # Key may be hyphenated or underscored
        varnames = extra.get("x-enum-varnames", extra.get("x_enum_varnames", []))
        if titles and varnames and len(titles) == len(varnames):
            return dict(zip(varnames, titles))

    return {}


# -- Value type resolution --


def resolve_value_type(channel: Any) -> str:
    """Determine the plot routing for a channel.

    Returns one of:
        "quantity"  - QuantityValue with unit (time series with unit switch)
        "number"    - unitless float/int/bool (time series, unitless)
        "text"      - string values (log console)
        "composite" - nested Characteristic (split recursively)
        "unknown"   - cannot determine
    """
    cls = resolve_characteristic_class(channel)
    if cls is None:
        return "unknown"

    # Check if it's a QuantityValue (has unit field with enum type)
    if get_unit_enum(channel) is not None:
        return "quantity"

    # Get fields (v2 model_fields or v1 __fields__)
    fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {})

    # Check for composite: has fields that are themselves Characteristic subclasses
    has_sub_characteristic = False
    for fname, finfo in fields.items():
        if fname in ("type",):
            continue
        annotation = getattr(finfo, "annotation", None)
        if annotation is None:
            annotation = getattr(finfo, "outer_type_", None)
        if annotation is None:
            continue
        ftype = _unwrap_optional(annotation)
        if isinstance(ftype, type) and _is_characteristic_subclass(ftype):
            has_sub_characteristic = True
            break

    if has_sub_characteristic:
        return "composite"

    # Check for primitive value field
    value_field = fields.get("value")
    if value_field is not None:
        annotation = getattr(value_field, "annotation", None)
        if annotation is None:
            annotation = getattr(value_field, "outer_type_", None)
        if annotation is not None:
            vtype = _unwrap_optional(annotation)
            if vtype in (float, int):
                return "number"
            if vtype is bool:
                return "number"
            if vtype is str:
                return "text"

    return "number"


def resolve_downsample_method(channel: Any, config_method: str = "auto") -> str:
    """Pick the downsampling strategy for a channel.

    When ``config_method`` is an explicit strategy ('sample'/'average'/
    'minmax') it is returned as-is. When it is 'auto' (or empty), numeric
    channels (quantity/number/composite) use 'minmax' (spike-safe, real
    points) and text/unknown channels use 'sample'.
    """
    if config_method and config_method != "auto":
        return config_method
    if resolve_value_type(channel) in ("quantity", "number", "composite"):
        return "minmax"
    return "sample"


def _is_characteristic_subclass(cls: type) -> bool:
    """Check if a class is a Characteristic subclass (v1 or v2)."""
    if not isinstance(cls, type):
        return False
    # Check by class name in MRO to handle both v1 and v2 class hierarchies
    return any(
        c.__name__ in ("Characteristic", "QuantityValue")
        for c in getattr(cls, "__mro__", [])
    )


# -- Composite channel flattening --


def flatten_composite_channels(
    controller: Any, channel: Any, _prefix: str = ""
) -> List[Tuple[Any, Any, str]]:
    """For composite characteristics, recursively split into leaf sub-channels.

    Returns list of (controller, channel, dotted_name_path) tuples.
    Each leaf has a synthetic channel-like object with its own characteristic class.
    """
    cls = resolve_characteristic_class(channel)
    if cls is None:
        name = _prefix or channel.name
        return [(controller, channel, name)]

    fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {})
    leaves = []
    for fname, finfo in fields.items():
        if fname in ("type",):
            continue
        annotation = getattr(finfo, "annotation", None)
        if annotation is None:
            annotation = getattr(finfo, "outer_type_", None)
        if annotation is None:
            continue
        ftype = _unwrap_optional(annotation)
        if not isinstance(ftype, type):
            continue
        if not _is_characteristic_subclass(ftype):
            continue
        path = f"{_prefix}.{fname}" if _prefix else f"{channel.name}.{fname}"
        # Create a synthetic channel-like object
        sub = _SyntheticChannel(
            name=fname,
            label=getattr(channel, "label", None),
            characteristic_class=ftype,
            parent_channel=channel,
        )
        # Check if this sub is itself composite
        sub_type = _resolve_value_type_for_class(ftype)
        if sub_type == "composite":
            leaves.extend(flatten_composite_channels(controller, sub, path))
        else:
            leaves.append((controller, sub, path))

    if not leaves:
        name = _prefix or channel.name
        return [(controller, channel, name)]
    return leaves


def _resolve_value_type_for_class(cls: type) -> str:
    """Resolve value type directly from a class (no channel needed)."""
    fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {})
    # Check for unit enum
    unit_field = fields.get("unit")
    if unit_field is not None:
        import enum

        annotation = getattr(unit_field, "annotation", None) or getattr(
            unit_field, "outer_type_", None
        )
        if annotation is not None:
            utype = _unwrap_optional(annotation)
            if isinstance(utype, type) and issubclass(utype, enum.Enum):
                return "quantity"
    # Check for sub-characteristics
    for fname, finfo in fields.items():
        if fname in ("type",):
            continue
        annotation = getattr(finfo, "annotation", None) or getattr(
            finfo, "outer_type_", None
        )
        if annotation is None:
            continue
        ftype = _unwrap_optional(annotation)
        if isinstance(ftype, type) and _is_characteristic_subclass(ftype):
            return "composite"
    return "number"


class _SyntheticChannel:
    """Lightweight channel-like object for composite sub-channels."""

    def __init__(self, name, label, characteristic_class, parent_channel):
        self.name = name
        self.label = label
        self.uuid = f"{parent_channel.uuid}_{name}"
        self._characteristic_class = characteristic_class
        self.parent_channel = parent_channel
        self.unit = None
        self.description = None
        # Provide __iris__ so resolve_characteristic_class works
        if hasattr(characteristic_class, "get_cls_iri"):
            self.__iris__ = {"characteristic": characteristic_class.get_cls_iri()}
        else:
            self.__iris__ = {}


# -- Grouping --


def resolve_fundamental_characteristic(cls: type) -> type:
    """Walk MRO to find the fundamental QuantityValue subclass.

    E.g., Width -> Length, Velocity -> Velocity (already fundamental).
    Used by SUB grouping mode.
    """
    try:
        from opensemantic.characteristics.quantitative._static import QuantityValue
    except ImportError:
        return cls

    # Walk MRO looking for the highest QuantityValue subclass
    # that still has its own unit field (not just inherited)
    best = cls
    for base in cls.__mro__:
        if base is QuantityValue or base is cls:
            continue
        if isinstance(base, type) and issubclass(base, QuantityValue):
            # Check if this base defines its own unit field
            if "unit" in getattr(base, "__annotations__", {}):
                best = base
    return best


def group_channels_by_characteristic(
    selected: List[Tuple[Any, Any]],
    mode: GroupingMode = GroupingMode.UNIQUE,
) -> Dict[str, List[Tuple[Any, Any]]]:
    """Group (controller, channel) tuples for y-axis assignment.

    Returns dict mapping group_key -> list of (controller, channel).
    """
    groups: Dict[str, List[Tuple[Any, Any]]] = {}

    for ctrl, ch in selected:
        if mode == GroupingMode.NONE:
            key = getattr(ch, "uuid", id(ch))
            groups.setdefault(str(key), []).append((ctrl, ch))
        elif mode == GroupingMode.UNIQUE:
            iri = get_characteristic_iri(ch) or "unknown"
            groups.setdefault(iri, []).append((ctrl, ch))
        elif mode == GroupingMode.SUB:
            cls = resolve_characteristic_class(ch)
            if cls is not None:
                fundamental = resolve_fundamental_characteristic(cls)
                if hasattr(fundamental, "get_cls_iri"):
                    key = fundamental.get_cls_iri()
                else:
                    key = fundamental.__name__
            else:
                key = get_characteristic_iri(ch) or "unknown"
            groups.setdefault(key, []).append((ctrl, ch))

    return groups


# -- Tree source building --


def build_tree_source(
    controllers: List[Any],
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """Build Wunderbaum-compatible tree source from DataToolControllers.

    Each controller becomes a checkable root node (toggles all children).
    Channels are checkable child nodes.
    """
    source = []
    for ctrl in controllers:
        ctrl_label = get_display_label(ctrl, lang)
        children = []
        for ch in ctrl.get_all_channels():
            char_label = resolve_characteristic_label(ch, lang)
            ch_label = get_display_label(ch, lang)
            # Build tooltip from all attributes
            tooltip_parts = []
            if ch_label:
                tooltip_parts.append(f"Label: {ch_label}")
            if hasattr(ch, "name") and ch.name:
                tooltip_parts.append(f"Name: {ch.name}")
            if char_label:
                tooltip_parts.append(f"Characteristic: {char_label}")
            unit_str = getattr(ch, "unit", None)
            if unit_str:
                tooltip_parts.append(f"Unit: {unit_str}")
            desc = _get_description_text(ch, lang)
            if desc:
                tooltip_parts.append(f"Description: {desc}")

            children.append(
                {
                    "title": ch_label or ch.name,
                    "key": ch.uuid,
                    "checkbox": True,
                    "selected": False,
                    "tooltip": "\n".join(tooltip_parts),
                    # Column data
                    "characteristic": char_label,
                    # Internal data for plot routing and grouping
                    "data": {
                        "controller_key": _get_ctrl_key(ctrl),
                        "channel_name": ch.name,
                        "characteristic_iri": get_characteristic_iri(ch),
                        "value_type": resolve_value_type(ch),
                    },
                }
            )

        source.append(
            {
                "title": ctrl_label,
                "key": _get_ctrl_key(ctrl),
                "expanded": True,
                "checkbox": True,
                "children": children,
            }
        )
    return source


def _get_ctrl_key(ctrl: Any) -> str:
    """Get a stable key for a controller."""
    if hasattr(ctrl, "get_osw_id"):
        return ctrl.get_osw_id()
    if hasattr(ctrl, "uuid"):
        return str(ctrl.uuid)
    return str(id(ctrl))


def _get_description_text(obj: Any, lang: str = "en") -> str:
    """Extract description text for the given language."""
    descs = getattr(obj, "description", None)
    if not descs or not isinstance(descs, list) or len(descs) == 0:
        return ""
    for d in descs:
        d_lang = getattr(d, "lang", None)
        if d_lang == lang:
            return getattr(d, "text", "")
    if lang != "en":
        for d in descs:
            if getattr(d, "lang", None) == "en":
                return getattr(d, "text", "")
    return getattr(descs[0], "text", "")


# -- Selection extraction --


def get_selected_channels(
    source: List[Dict],
    controllers: List[Any],
) -> List[Tuple[Any, Any]]:
    """Extract (controller, channel) tuples for all selected tree nodes.

    Walks the tree source, finds child nodes with selected=True,
    maps back to actual DataChannel instances.
    """
    # Build lookup: uuid -> (controller, channel)
    lookup: Dict[str, Tuple[Any, Any]] = {}
    for ctrl in controllers:
        for ch in ctrl.get_all_channels():
            lookup[ch.uuid] = (ctrl, ch)

    selected = []
    for root in source:
        for child in root.get("children", []):
            if child.get("selected", False):
                key = child.get("key")
                if key in lookup:
                    selected.append(lookup[key])
    return selected


# -- Translation helper --

_UI_STRINGS = {
    "load_data": {"en": "Load Data", "de": "Daten laden"},
    "auto_fetch": {"en": "Auto-fetch", "de": "Auto-Laden"},
    "row_limit": {"en": "Row limit", "de": "Zeilenlimit"},
    "time_range": {"en": "Time Range", "de": "Zeitbereich"},
    "start": {"en": "Start", "de": "Start"},
    "end": {"en": "End", "de": "Ende"},
    "data_channels": {"en": "Data Channels", "de": "Datenkanaele"},
    "plot_controls": {"en": "Plot Controls", "de": "Plot-Einstellungen"},
    "time_series": {"en": "Time Series", "de": "Zeitreihe"},
    "log_console": {"en": "Log Console", "de": "Protokoll"},
    "unit": {"en": "Unit", "de": "Einheit"},
    "channel": {"en": "Channel", "de": "Kanal"},
    "characteristic": {"en": "Characteristic", "de": "Charakteristik"},
    "config": {"en": "Configuration", "de": "Konfiguration"},
    "archive": {"en": "Archive", "de": "Archiv"},
    "live": {"en": "Live", "de": "Live"},
    "clear_cache": {"en": "Clear Cache", "de": "Cache leeren"},
    "load_range": {"en": "Load current range", "de": "Aktuellen Bereich laden"},
    "cache_enabled": {"en": "Cache enabled", "de": "Cache aktiviert"},
    "grouping": {"en": "Grouping", "de": "Gruppierung"},
    "group_none": {"en": "None", "de": "Keine"},
    "group_unique": {"en": "By characteristic", "de": "Nach Charakteristik"},
    "group_sub": {"en": "By sub-characteristic", "de": "Nach Subcharakteristik"},
    "downsample": {"en": "Downsampling", "de": "Downsampling"},
    "max_points": {"en": "Max points", "de": "Maximale Punkte"},
    "ds_method": {"en": "Method", "de": "Methode"},
    "ds_auto": {"en": "Auto", "de": "Auto"},
    "ds_sample": {"en": "Sample", "de": "Stichprobe"},
    "ds_average": {"en": "Average", "de": "Mittelwert"},
    "ds_minmax": {"en": "Min/Max", "de": "Min/Max"},
    "edge_anchors": {"en": "Edge anchors", "de": "Randpunkte"},
    "export": {"en": "Export", "de": "Export"},
    "download_csv": {"en": "Download CSV", "de": "CSV herunterladen"},
    "download_xlsx": {"en": "Download XLSX", "de": "XLSX herunterladen"},
    "download_plot": {"en": "Download plot (HTML)", "de": "Plot (HTML) herunterladen"},
    "no_data": {"en": "No data", "de": "Keine Daten"},
}


def _t(key: str, lang: str = "en") -> str:
    """Translate a UI string key to the given language."""
    entry = _UI_STRINGS.get(key, {})
    return entry.get(lang, entry.get("en", key))
