"""Mixin classes with all controller methods and inner types.

These mixins are composed with the appropriate v1 or v2 model base class
in _controller.py and v1/_controller.py respectively.
No model imports here - only stdlib, pydantic BaseModel, and _controller_logic.
"""

import asyncio
import datetime as dt
import json
import logging
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from oold.model import BaseController
from pydantic import BaseModel, ConfigDict

_logger = logging.getLogger(__name__)


def short_osw_id(osw_id: str) -> str:
    """The bare id of a subobject, without its parent prefix.

    Channels carry a composite osw_id (``Item:OSW<tool>#OSW<channel>``) but are
    stored and referenced by the part after the separator.
    """
    return osw_id.split("#")[-1]


class DownsampleParams(BaseModel):
    """Server-side downsampling request (PostgREST/TimescaleDB backend).

    Grouped into one sub-object so the read/load params stay tidy. Honored by
    backends that support it; others ignore or approximate it. All-None (or a
    None ``DownsampleParams``) means no downsampling - full resolution.
    """

    max_points: Optional[int] = None
    """Target point count: the read is bucketed into ~max_points buckets."""
    bin_size: Optional[str] = None
    """Explicit bucket width as a Postgres interval string (e.g. '5 seconds');
    overrides max_points when set."""
    method: Optional[str] = None
    """Strategy: 'sample', 'average' or 'minmax'. None disables downsampling.

    Note: 'average'/'minmax' aggregate the bare stored numbers per leaf and are
    only correct for unit-normalized (base-unit) data."""
    edge_anchors: Optional[bool] = None
    """Include the window's first/last real datapoints as endpoints
    (server default: True)."""

    def is_active(self) -> bool:
        """True if any downsampling is requested."""
        return (
            self.max_points is not None
            or self.bin_size is not None
            or bool(self.method)
        )


class DataToolMixin(BaseController):
    """Generic controller mixin for DataTool models.

    Provides: identity, channel/subdevice traversal, archiving, data change handling.
    Compose with a DataTool model class:
        class DataToolController(DataToolMixin, DataTool): pass
    """

    def __init__(self, *args, **data):
        super().__init__(*args, **data)
        self._compute_subobject_ids()
        self.rebuild_channel_dict()
        # Warn about unloaded channel characteristics
        self._check_channel_characteristics()
        # TODO: update wiki OpcUaServer model to include endpoint/url field
        # so it survives serialization. Currently url is controller-only.

        # Auto-init archive database from storage_locations
        _archive_db = getattr(self, "archive_database", None)
        _needs_init = _archive_db is None or isinstance(_archive_db, dict)
        _storage = getattr(self, "storage_locations", None)
        if self.auto_archive and _needs_init and _storage:
            self.archive_database = self._init_archive_database(_storage[0])

    # __setattr__ for private attrs inherited from BaseController
    # get_osw_id() and get_iri() are inherited from OswBaseModel
    # via the model base class (DataTool -> Entity -> OswBaseModel)

    def rebuild_channel_dict(self):
        """Index all channels of self and its subdevices for fast lookup.

        Called from __init__. Call it again after mutating data_channels or
        subdevices, otherwise incoming notifications for the new channels are
        not routed.
        """
        if not isinstance(getattr(self, "_channel_dict", None), dict):
            object.__setattr__(self, "_channel_dict", {})
        self._channel_dict.clear()
        for channel in self.get_all_channels():
            # Use node_id if available (OPC UA), fall back to uuid
            key = getattr(channel, "node_id", None) or channel.uuid
            self._channel_dict[key] = channel

    # TODO: Consider moving _compute_subobject_ids to OswBaseModel
    def _compute_subobject_ids(self, parent_chain=None):
        """Compute composite osw_ids for inline subobject children.

        For each field without 'range' in json_schema_extra (i.e. not a wiki
        reference), prefix the child's osw_id with the parent's chain.

        Called automatically in __init__. For mutations after construction,
        call this method manually to recompute.
        """
        my_uuid = self.get_uuid()
        if my_uuid is None:
            return
        base_id = f"OSW{str(my_uuid).replace('-', '')}"

        if parent_chain:
            self.osw_id = f"{parent_chain}#{base_id}"

        my_osw_id = getattr(self, "osw_id", None) or base_id

        fields = {}
        if hasattr(self, "model_fields"):
            fields = self.model_fields
        elif hasattr(self, "__fields__"):
            fields = self.__fields__

        for field_name, field_info in fields.items():
            # Check for 'range' in field metadata
            # v2: json_schema_extra, v1: field_info.extra
            extra = getattr(field_info, "json_schema_extra", None) or {}
            if not extra and hasattr(field_info, "field_info"):
                extra = getattr(field_info.field_info, "extra", {}) or {}
            if "range" in extra:
                continue

            value = getattr(self, field_name, None)
            if value is None:
                continue

            children = []
            items = value if isinstance(value, list) else [value]
            for item in items:
                if hasattr(item, "get_uuid") and hasattr(item, "osw_id"):
                    children.append(item)

            for child in children:
                child_uuid = child.get_uuid()
                if child_uuid is None:
                    continue
                child_base_id = f"OSW{str(child_uuid).replace('-', '')}"
                new_osw_id = f"{my_osw_id}#{child_base_id}"
                if child.osw_id != new_osw_id:
                    child.osw_id = new_osw_id
                # Recurse if child also has _compute_subobject_ids
                if hasattr(child, "_compute_subobject_ids"):
                    child._compute_subobject_ids(parent_chain=new_osw_id)

    def _check_channel_characteristics(self):
        """Warn if any channel has an unresolvable characteristic IRI.

        A characteristic IRI that is not in oold's _types registry
        means typed read/write will fail for that channel unless
        target_schema is passed explicitly.
        """
        try:
            from oold.model import _types
        except ImportError:
            return
        for ch in self.get_all_channels():
            iris = getattr(ch, "__iris__", {}).get("characteristic", [])
            if isinstance(iris, str):
                iris = [iris]
            for iri in iris:
                if iri and iri not in _types:
                    _logger.warning(
                        "Channel '%s': characteristic IRI '%s' is not "
                        "in the type registry. Import the corresponding "
                        "package (e.g. opensemantic.characteristics."
                        "quantitative) to enable typed read/write.",
                        ch.name,
                        iri,
                    )

    def get_credential(self, iri: str):
        """Look up a credential for the given IRI.

        Uses the instance's credential_manager if set, otherwise falls back
        to the global oold.backend.auth.get_credential store.

        Parameters
        ----------
        iri
            The IRI to look up credentials for.
        """
        from oold.backend.auth import get_credential as _global_get_credential

        if getattr(self, "credential_manager", None) is not None:
            from oold.backend.auth import CredentialManager

            config = CredentialManager.CredentialConfig(iri=iri)
            return self.credential_manager.get_credential(config)
        return _global_get_credential(iri)

    def _init_archive_database(self, db):
        """Create a TimeSeriesDatabaseController from a Database entity.

        Uses oold's backend resolution to get the full Database instance
        (if db is an IRI string), then casts it to the appropriate
        controller class.

        Parameters
        ----------
        db
            A Database model instance or IRI string from storage_locations.

        Returns
        -------
            A TimeSeriesDatabaseController instance, or None.
        """
        # Already a controller - return as-is
        if isinstance(db, BaseController):
            return db

        # If db is a string IRI, it hasn't been resolved yet
        if isinstance(db, str):
            _logger.warning(
                "storage_locations[0] is an unresolved IRI: %s. "
                "Register a backend with set_backend() to enable "
                "auto-resolution.",
                db,
            )
            return None

        # Determine target controller class and extra kwargs.
        # Try inline object first, then IRI resolution via backend.
        server = db.__dict__.get("server")
        server_url = getattr(server, "url", None) if server else None
        if server_url is None:
            # Server may be an IRI in __iris__ - try to resolve or
            # use the IRI itself if it looks like a URL
            server_iri = getattr(db, "__iris__", {}).get("server")
            if server_iri and isinstance(server_iri, str):
                if server_iri.startswith("http"):
                    server_url = server_iri
                else:
                    try:
                        server = getattr(db, "server", None)
                        server_url = getattr(server, "url", None) if server else None
                    except (ValueError, ImportError):
                        pass

        # Build API URL from server fields
        if not server_url and server is not None:
            schema = getattr(server, "schema_", None) or "http"
            domain = getattr(server, "domain", None)
            ports = getattr(server, "network_port", None)
            port = ports[0] if ports else None
            path = getattr(server, "url_path", None) or ""
            if domain:
                server_url = f"{schema}://{domain}"
                if port:
                    server_url += f":{port}"
                server_url += f"/{path}".rstrip("/")

        if server_url:
            try:
                from postgrest import AsyncPostgrestClient

                # Look up credentials for this server
                cred = self.get_credential(server_url)
                headers = {}
                if cred is not None:
                    token = getattr(cred, "token", None)
                    if token is not None:
                        secret = (
                            token.get_secret_value()
                            if hasattr(token, "get_secret_value")
                            else str(token)
                        )
                        headers["Authorization"] = f"Bearer {secret}"

                client = AsyncPostgrestClient(
                    base_url=server_url,
                    schema="api",
                    headers=headers,
                )

                # Use version-matching controller
                db_module = type(db).__module__
                if ".v1." in db_module or db_module.endswith(".v1"):
                    from opensemantic.base.v1._controller import (
                        PostgrestTimeSeriesDatabaseController,
                    )
                else:
                    from opensemantic.base._controller import (
                        PostgrestTimeSeriesDatabaseController,
                    )

                controller = db.cast(
                    PostgrestTimeSeriesDatabaseController,
                    remove_extra=True,
                )
                controller.set_client(client)
                _logger.info(
                    "Auto-initialized PostgREST controller" " for %s at %s",
                    db.name,
                    server_url,
                )
                return controller
            except ImportError:
                _logger.warning(
                    "postgrest package not installed. " "Falling back to local SQLite."
                )
            except Exception as e:
                _logger.warning(
                    "Could not create PostgREST controller for %s: %s."
                    " Falling back to local SQLite.",
                    db.name,
                    e,
                )

        # Fall back to local SQLite
        try:
            # Use version-matching controller (v1 db -> v1 controller)
            db_module = type(db).__module__
            if ".v1." in db_module or db_module.endswith(".v1"):
                from opensemantic.base.v1._controller import (
                    LocalTimeSeriesDatabaseController,
                )
            else:
                from opensemantic.base._controller import (
                    LocalTimeSeriesDatabaseController,
                )
            db_path = f"./{db.name}.sqlite"
            controller = db.cast(
                LocalTimeSeriesDatabaseController,
                remove_extra=True,
                db_path=db_path,
            )
            _logger.info(
                "Auto-initialized LocalTimeSeriesDatabaseController" " for %s at %s",
                db.name,
                db_path,
            )
            return controller
        except ImportError:
            _logger.error(
                "Cannot auto-initialize archive database: "
                "install opensemantic.base[controller] "
                "(aiosqlite or postgrest)"
            )
            return None

    # -- Component hierarchy --

    @staticmethod
    def _component_refs(entity) -> list:
        """Return (component_type_iri, component_instance) pairs of a tool.

        The IRIs are read from ``__iris__`` instead of the attributes so the
        lazy backend resolution of ``component_instance`` is not triggered.
        Resolving it would load the child with autofetch_schema=True and
        generate an ad-hoc model instead of using the installed package.
        """

        def _first(value):
            if isinstance(value, list):
                return value[0] if value else None
            return value

        refs = []
        for comp in getattr(entity, "components", None) or []:
            iris = getattr(comp, "__iris__", None) or {}
            instance = _first(iris.get("component_instance"))
            if instance is None:
                # Inline object instead of a reference
                instance = comp.__dict__.get("component_instance")
            if instance is None:
                continue
            refs.append((_first(iris.get("component_type")), instance))
        return refs

    @classmethod
    def load_from_osw(
        cls,
        osw,
        iri: str,
        model_by_component_type: Optional[Dict[str, type]] = None,
        default_model: Optional[type] = None,
        depth: int = -1,
        **kwargs,
    ):
        """Load a tool and its component hierarchy from the OSW backend.

        ``components`` is the OSW backend's parent/child relation for tools.
        The controller-only ``subdevices`` list is populated from it, so
        get_all_channels(), get_channel_owner() and archiving work on the
        whole hierarchy.

        Parameters
        ----------
        osw
            An ``osw.core.OSW`` instance. Only ``load_entity`` and its
            ``LoadEntityParam`` are used, so osw stays an optional dependency.
        iri
            IRI of the root tool, e.g. ``Item:OSW<uuid without dashes>``.
        model_by_component_type
            Maps a component type IRI to the class the child is loaded as.
            Use it when the children are not all of the same kind. The classes
            have to be controllers (composing this mixin), otherwise the tree
            traversal in get_subdevices() / get_all_channels() fails.
        default_model
            Controller class for children without a mapping. Defaults to
            ``cls``.
        depth
            Component levels to follow, -1 for unlimited, 0 for the root only.
        kwargs
            Extra attributes to set on the root, e.g. ``url=...``.
        """
        return cls._load_tree(
            osw=osw,
            iri=iri,
            model=cls,
            mapping=model_by_component_type or {},
            default_model=default_model or cls,
            depth=depth,
            extra=kwargs,
        )

    @classmethod
    def _load_tree(cls, osw, iri, model, mapping, default_model, depth, extra):
        param = type(osw).LoadEntityParam(
            titles=iri, autofetch_schema=False, model_to_use=model
        )
        entity = osw.load_entity(param).entities[0]
        for key, value in (extra or {}).items():
            setattr(entity, key, value)

        if depth == 0:
            return entity

        subdevices = []
        for component_type, ref in cls._component_refs(entity):
            child_model = mapping.get(component_type, default_model)
            if isinstance(ref, str):
                subdevices.append(
                    cls._load_tree(
                        osw=osw,
                        iri=ref,
                        model=child_model,
                        mapping=mapping,
                        default_model=default_model,
                        depth=depth - 1,
                        extra={},
                    )
                )
            elif isinstance(ref, child_model):
                subdevices.append(ref)
            else:
                subdevices.append(child_model(ref))

        if subdevices:
            # Bypass validation: assigning to the field would revalidate every
            # child (pydantic v1 replaces them with copies), which detaches the
            # controller state the caller still holds a reference to.
            entity.__dict__["subdevices"] = subdevices
            if hasattr(entity, "rebuild_channel_dict"):
                entity.rebuild_channel_dict()
        return entity

    def get_subdevices(self) -> list:
        if self.subdevices is None:
            return []
        result = list(self.subdevices)
        for sub in self.subdevices:
            result.extend(sub.get_subdevices())
        return result

    def get_all_channels(self) -> list:
        channels = list(self.data_channels or [])
        for sub in self.subdevices or []:
            channels.extend(sub.get_all_channels())
        return channels

    def get_channel_owner(self, channel):
        own_uuids = [c.uuid for c in (self.data_channels or [])]
        if channel.uuid in own_uuids:
            return self
        for sub in self.subdevices or []:
            try:
                return sub.get_channel_owner(channel)
            except ValueError:
                continue
        raise ValueError(
            f"Channel {channel.name} with uuid {channel.uuid} "
            f"not found in any device controller"
        )

    def get_channel_by_name(self, name: str):
        """Look up a channel by name across self and all subdevices.

        Raises ValueError if no channel with the given name is found.
        """
        for ch in self.get_all_channels():
            if ch.name == name:
                return ch
        raise ValueError(
            f"No channel named '{name}' found. "
            f"Available: {[ch.name for ch in self.get_all_channels()]}"
        )

    def get_channel_by_osw_id(self, osw_id: str):
        """Look up a channel by osw_id across self and all subdevices.

        Accepts the full subobject IRI (``Item:OSW<tool>#OSW<channel>``) as
        well as the bare ``OSW<channel>`` suffix, so a reference stored in the
        OSW backend resolves regardless of which form it uses.

        Raises ValueError if no channel with the given osw_id is found.
        """
        suffix = short_osw_id(osw_id)
        for ch in self.get_all_channels():
            ch_osw_id = getattr(ch, "osw_id", None)
            if ch_osw_id and short_osw_id(ch_osw_id) == suffix:
                return ch
        raise ValueError(f"No channel with osw_id '{osw_id}' found.")

    def _resolve_channel(self, channel):
        """Resolve a channel argument: pass through if already an object,
        look up by name if string."""
        if isinstance(channel, str):
            return self.get_channel_by_name(channel)
        return channel

    # -- Inner param/result classes --

    class StoreChannelDataParams(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        channel: Any = None
        """DataChannel instance or channel name (str)."""
        value: Any = None
        """Raw dict/scalar or Characteristic instance."""
        timestamp: Optional[dt.datetime] = None
        """Timestamp for the data point. Defaults to now(UTC)."""

    class StoreChannelSeriesParams(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        channel: Any = None
        """DataChannel instance or channel name (str)."""
        timestamps: List[dt.datetime] = []
        """Timestamps, parallel to ``values``."""
        values: List[Any] = []
        """Raw dicts/scalars or Characteristic instances, parallel to
        ``timestamps``."""

    class StoreChannelDataBulkParams(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        series: List["DataToolMixin.StoreChannelSeriesParams"] = []
        """One entry per channel, each with parallel timestamps/values."""
        chunk_size: int = 5000
        """Rows per write_tool_channel_raw batch."""
        ensure_tool: bool = True
        """Create the tool table first if it does not exist."""
        schema_reload_wait: Optional[float] = None
        """Seconds to wait after creating the tool (PostgREST reloads its
        schema cache after the DDL). None auto-selects 1.5 for PostgREST and
        0 for local SQLite."""

    class LoadChannelDataParams(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        channel: Union[str, List[str], Any, None] = None
        """Channel name (str), list of names, DataChannel instance,
        list of DataChannel instances, or None (all channels)."""
        start: Optional[dt.datetime] = None
        """Start of the time range (inclusive). None reads from the earliest
        stored point."""
        end: Optional[dt.datetime] = None
        """End of the time range (inclusive). None reads up to the latest
        stored point."""
        limit: Optional[int] = None
        """Maximum number of rows to return. None means no limit."""
        downsample: Optional[DownsampleParams] = None
        """Optional server-side downsampling request. None = full resolution."""
        typed: bool = True
        """If True, deserialize values using channel characteristic or
        target_schema. If False, return raw dicts (faster)."""
        target_schema: Any = None
        """Explicit class for typed deserialization (e.g. Temperature).
        Overrides channel characteristic resolution."""

    class ChannelDataPoint(BaseModel):
        """A single data point returned by load_channel_data."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        timestamp: dt.datetime
        channel: Any = None
        value: Any = None
        """Typed Characteristic instance or raw dict, depending on
        the typed parameter and channel characteristic."""

    class ChannelDataChangeNotificationParams(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        channel: Any = None
        value: Any = None
        timestamp: Optional[dt.datetime] = None

    class AutoArchiveParams(BaseModel):
        enable: bool = True

    # -- Async methods --

    async def _handle_data_change(
        self,
        params: "DataToolMixin.ChannelDataChangeNotificationParams",
    ):
        if not hasattr(self, "_last_values"):
            self._last_values = {}
        if params.channel.uuid in self._last_values:
            last = self._last_values[params.channel.uuid]
            if last.value == params.value and last.timestamp == params.timestamp:
                _logger.warning(
                    "Duplicate data change for %s, ignoring", params.channel.name
                )
                return
        self._last_values[params.channel.uuid] = params

        if self.auto_archive and self.archive_database is not None:
            owner = self.get_channel_owner(params.channel)
            if owner.auto_archive:
                try:
                    tool_osw_id = owner.get_osw_id()
                    value = self._value_to_store_data(params.value, params.channel)
                    # Use just the channel's own ID (child part of subobject ID)
                    ch_osw_id = short_osw_id(params.channel.get_osw_id())
                    offline_before = getattr(self.archive_database, "_offline", False)
                    await self.archive_database.write_tool_channel_raw(
                        TSDCMixin.WriteToolChannelRawParams(
                            tool_osw_id=tool_osw_id,
                            data=[
                                {
                                    "ts": params.timestamp.isoformat(),
                                    "ch": ch_osw_id,
                                    "data": value,
                                }
                            ],
                        )
                    )
                    if not offline_before and getattr(
                        self.archive_database, "_offline", False
                    ):
                        _logger.warning("Database went offline")
                        self._on_archive_error()
                except Exception as e:
                    _logger.error("Error archiving data change: %s", e)

        if self._channel_datachange_notification_callback is not None:
            try:
                await self._channel_datachange_notification_callback(
                    type(self).ChannelDataChangeNotificationParams(
                        channel=params.channel,
                        value=params.value,
                        timestamp=params.timestamp,
                    )
                )
            except Exception as e:
                _logger.error("Error in data change callback: %s", e)
                import traceback

                _logger.error(traceback.format_exc())

    def _on_archive_error(self):
        """Called when the archive DB goes offline. Override in subclasses."""
        pass

    async def configure_auto_archive(self, params: "DataToolMixin.AutoArchiveParams"):
        if params.enable and self.archive_database is None:
            raise ValueError("Auto archive enabled but no archive database set")
        self.auto_archive = params.enable
        if params.enable:
            _logger.warning("Auto archive enabled")
        else:
            _logger.warning("Auto archive disabled")
        if params.enable and self.archive_database is not None:
            existing_tools = await self.archive_database.get_tools_list()
            required_tools = [self.get_osw_id()]
            for device in self.get_subdevices():
                required_tools.append(device.get_osw_id())
            for osw_id in required_tools:
                if osw_id not in existing_tools:
                    try:
                        await self.archive_database.create_tool(
                            TSDCMixin.CreateToolParams(tool_osw_id=osw_id)
                        )
                    except Exception as e:
                        _logger.error("Error creating tool %s: %s", osw_id, e)
            await asyncio.sleep(1)

    # read_archive_data, store_typed_data, read_typed_data removed.
    # Use store_channel_data / load_channel_data instead.

    def _value_to_store_data(self, value, channel):
        """Convert a value to a dict suitable for DB storage.

        Handles typed (Characteristic), dict, and raw scalar values.
        For raw scalars, uses channel's characteristic + unit to convert
        to base unit if available.
        """
        if hasattr(value, "to_json"):
            # Warn if typed value's unit differs from channel's unit
            ch_unit = getattr(channel, "unit", None)
            val_unit = getattr(value, "unit", None)
            if ch_unit is not None and val_unit is not None:
                # Resolve channel unit IRI to enum for comparison
                try:
                    resolved = type(value)(value=0, unit=ch_unit).unit
                    if str(resolved) != str(val_unit):
                        _logger.warning(
                            "Value unit %s differs from channel '%s' "
                            "unit %s. Storing in base unit.",
                            val_unit,
                            getattr(channel, "name", "?"),
                            resolved,
                        )
                except Exception:
                    pass
            if hasattr(value, "to_base"):
                try:
                    value = value.to_base()
                except Exception:
                    pass
            try:
                return value.to_json(exclude_defaults=True)
            except Exception:
                return value.to_json()
        if isinstance(value, dict):
            return json.loads(json.dumps(value, default=str))
        # Raw scalar: wrap with channel unit, convert to base, serialize
        typed = self._wrap_raw_value(value, channel)
        if hasattr(typed, "to_json"):
            if hasattr(typed, "to_base"):
                try:
                    typed = typed.to_base()
                except Exception:
                    pass
            try:
                return typed.to_json(exclude_defaults=True)
            except Exception:
                return typed.to_json()
        return {"value": value}

    async def set_buffered(self, enabled: bool = True, batch_size: int = 100):
        """Enable or disable buffered writes on the archive database.

        When enabled, writes are collected in memory and flushed to disk
        in batches, which is much faster for bulk inserts.
        Call flush_buffer() when done to persist any remaining data.
        When disabling, any pending buffered data is flushed automatically.
        """
        db = self.archive_database
        if db is None:
            _logger.warning("No archive database configured")
            return
        driver = getattr(db, "_driver", None)
        if driver is None:
            _logger.warning("Archive database has no _driver attribute")
            return
        if not enabled and driver.buffered:
            await self.flush_buffer()
        driver.buffered = enabled
        driver.buffer_batch_size = batch_size
        if enabled:
            _logger.info(
                "Buffered write mode enabled (batch_size=%d). "
                "Call flush_buffer() when done to persist remaining data.",
                batch_size,
            )

    async def flush_buffer(self):
        """Flush buffered writes to the archive database."""
        if self.archive_database is not None:
            await self.archive_database.flush_buffer()

    async def stop(self):
        _logger.warning("Stopping")
        await self.flush_buffer()

    # -- High-level store/load API --

    async def store_channel_data(
        self, params: "DataToolMixin.StoreChannelDataParams"
    ) -> None:
        """Store a single channel value to the archive database.

        Resolves channel by name if a string is passed.
        If value is a Characteristic instance, converts to base unit
        and serializes. Otherwise stores as raw dict/scalar.
        Auto-creates the tool table on first write (for SQLite).
        """
        if self.archive_database is None:
            raise ValueError("No archive database configured")
        channel = self._resolve_channel(params.channel)
        ts = params.timestamp or dt.datetime.now(dt.timezone.utc)
        value = params.value

        data = self._value_to_store_data(value, channel)

        ch_osw_id = short_osw_id(channel.get_osw_id())
        tool_osw_id = self.get_osw_id()

        await self.archive_database.write_tool_channel_raw(
            TSDCMixin.WriteToolChannelRawParams(
                tool_osw_id=tool_osw_id,
                data=[
                    {
                        "ts": ts.isoformat(),
                        "ch": ch_osw_id,
                        "data": data,
                    }
                ],
            )
        )

    async def _ensure_tool_exists(self, tool_osw_id: str) -> bool:
        """Create the tool table if it does not exist. Returns True if created.

        For PostgREST the CREATE triggers a schema-cache reload, so callers
        should wait briefly before the first write (see store_channel_data_bulk).
        """
        try:
            existing = await self.archive_database.get_tools_list()
        except Exception:
            existing = []
        if tool_osw_id in existing:
            return False
        await self.archive_database.create_tool(
            TSDCMixin.CreateToolParams(tool_osw_id=tool_osw_id)
        )
        return True

    async def store_channel_data_bulk(
        self, params: "DataToolMixin.StoreChannelDataBulkParams"
    ) -> int:
        """Store many channel values at once (efficient for large series).

        Each entry in ``params.series`` carries a channel (name or instance)
        with parallel ``timestamps``/``values`` arrays. Values are converted
        exactly like store_channel_data (Characteristic -> base unit; dict or
        scalar otherwise) and written in ``chunk_size`` batches via
        write_tool_channel_raw. When ``ensure_tool`` the tool table is created
        first if missing (with a schema-reload wait for PostgREST). Returns the
        number of points written.
        """
        if self.archive_database is None:
            raise ValueError("No archive database configured")
        tool_osw_id = self.get_osw_id()

        if params.ensure_tool and await self._ensure_tool_exists(tool_osw_id):
            driver = getattr(self.archive_database, "_driver", None)
            is_pgrst = getattr(driver, "client", None) is not None
            wait = (
                params.schema_reload_wait
                if params.schema_reload_wait is not None
                else (1.5 if is_pgrst else 0.0)
            )
            if wait > 0:
                await asyncio.sleep(wait)

        written = 0
        batch: List[dict] = []

        async def _flush():
            nonlocal written, batch
            if not batch:
                return
            await self.archive_database.write_tool_channel_raw(
                TSDCMixin.WriteToolChannelRawParams(tool_osw_id=tool_osw_id, data=batch)
            )
            written += len(batch)
            batch = []

        for series in params.series:
            channel = self._resolve_channel(series.channel)
            ch_osw_id = short_osw_id(channel.get_osw_id())
            n = min(len(series.timestamps), len(series.values))
            for i in range(n):
                data = self._value_to_store_data(series.values[i], channel)
                batch.append(
                    {
                        "ts": series.timestamps[i].isoformat(),
                        "ch": ch_osw_id,
                        "data": data,
                    }
                )
                if len(batch) >= params.chunk_size:
                    await _flush()
        await _flush()
        return written

    async def load_channel_data(
        self,
        params: "DataToolMixin.LoadChannelDataParams",
    ) -> List["DataToolMixin.ChannelDataPoint"]:
        """Load channel data from the archive database.

        Parameters
        ----------
        params
            LoadChannelDataParams with channel (str, list, instance, or
            None for all), time range, limit, typed flag, target_schema.

        Returns
        -------
        List[ChannelDataPoint]
            Each point has timestamp, channel, and value (typed
            Characteristic if typed=True, raw dict if typed=False).
        """
        if self.archive_database is None:
            raise ValueError("No archive database configured")

        # Resolve channels
        channels = params.channel
        if channels is None:
            channels = self.get_all_channels()
        elif isinstance(channels, str):
            channels = [self._resolve_channel(channels)]
        elif isinstance(channels, list):
            channels = [
                self._resolve_channel(ch) if isinstance(ch, str) else ch
                for ch in channels
            ]
        else:
            channels = [channels]

        # Build channel ID -> channel lookup
        ch_by_id = {short_osw_id(ch.get_osw_id()): ch for ch in channels}

        # Query: if single channel, filter by ID; otherwise get all
        ch_osw_id = None
        if len(channels) == 1:
            ch_osw_id = list(ch_by_id.keys())[0]

        raw = await self.archive_database.read_tool_channel_raw(
            TSDCMixin.ReadToolChannelRawParams(
                tool_osw_id=self.get_osw_id(),
                channel_osw_id=ch_osw_id,
                start=params.start,
                end=params.end,
                limit=params.limit,
                downsample=params.downsample,
            )
        )

        results: List["DataToolMixin.ChannelDataPoint"] = []
        # The typed branch's display unit is loop-invariant per characteristic
        # class, so resolve it once per class instead of rebuilding
        # ``cls(value=0, unit=ch_unit)`` for every row. Cached by class object;
        # a value of None means "no conversion" (channel has no declared unit).
        target_unit_cache: dict = {}

        for row in raw:
            ch = ch_by_id.get(row["ch"])
            if ch is None and len(ch_by_id) > 1:
                continue  # skip rows for channels not in the request

            value = row["data"]
            if params.typed:
                cls = params.target_schema
                if cls is None and ch is not None:
                    cls = self._resolve_characteristic_class(ch)
                if cls is not None:
                    value = cls.from_json(value)
                    if cls not in target_unit_cache:
                        ch_unit = getattr(ch, "unit", None) if ch else None
                        resolved = None
                        if ch_unit is not None:
                            try:
                                resolved = cls(value=0, unit=ch_unit).unit
                            except Exception:
                                resolved = None
                        target_unit_cache[cls] = resolved
                    target_unit = target_unit_cache[cls]
                    if target_unit is not None and hasattr(value, "to_unit"):
                        try:
                            value = value.to_unit(target_unit)
                        except Exception:
                            pass

            results.append(
                type(self).ChannelDataPoint(
                    timestamp=row["ts"],
                    channel=ch,
                    value=value,
                )
            )
        return results

    def _wrap_raw_value(self, value, channel):
        """Wrap a raw scalar in the channel's characteristic class.

        If the channel has a characteristic class and a unit, creates a
        typed instance (e.g., Temperature(value=22.5, unit=Celsius)).
        Returns the original value if wrapping is not possible.
        """
        if hasattr(value, "to_json") or isinstance(value, dict):
            return value
        cls = self._resolve_characteristic_class(channel)
        ch_unit = getattr(channel, "unit", None)
        if cls is not None and ch_unit is not None:
            try:
                return cls(value=value, unit=ch_unit)
            except Exception:
                pass
        return value

    def _resolve_characteristic_class(self, channel):
        """Try to resolve the characteristic class for a channel.

        Checks __iris__ for the characteristic IRI (avoids triggering
        backend resolution), then looks up the class in the _types registry.
        Returns the class or None.
        """
        # Get IRI from __iris__ (avoids backend resolution via __getattribute__)
        iris = getattr(channel, "__iris__", {})
        char_iri = iris.get("characteristic")
        if char_iri is None:
            # Fall back to direct attribute (may trigger backend)
            try:
                char_iri = getattr(channel, "characteristic", None)
            except (ValueError, ImportError):
                return None
        if char_iri is None:
            return None
        # Handle list of IRIs (take first)
        if isinstance(char_iri, list):
            char_iri = char_iri[0] if char_iri else None
        if char_iri is None:
            return None
        # If it's already a class, return it
        if isinstance(char_iri, type) and hasattr(char_iri, "from_json"):
            return char_iri
        # Look up IRI string in the _types registry
        if isinstance(char_iri, str):
            try:
                from oold.model import _types

                return _types.get(char_iri)
            except ImportError:
                return None
        return None


class TSDCMixin(BaseController):
    """Mixin providing TimeSeriesDatabaseController methods and inner types.

    Compose with a Database model class to create a concrete controller:
        class TimeSeriesDatabaseController(TSDCMixin, Database): pass
    """

    class CreateToolParams(BaseModel):
        tool_osw_id: str
        """OSW ID of the tool"""

    @abstractmethod
    async def create_tool(self, params: "TSDCMixin.CreateToolParams"):
        pass

    class DeleteToolParams(BaseModel):
        tool_osw_id: str
        """OSW ID of the tool"""

    @abstractmethod
    async def delete_tool(self, params: "TSDCMixin.DeleteToolParams"):
        pass

    @abstractmethod
    async def get_tools_list(self) -> List[str]:
        """Returns a list of all registered tools."""
        pass

    class WriteToolChannelRawParams(BaseModel):
        tool_osw_id: str
        """OSW ID of the tool"""
        data: list
        """List of data rows to store"""

    @abstractmethod
    async def write_tool_channel_raw(
        self, params: "TSDCMixin.WriteToolChannelRawParams"
    ):
        """Stores data for a tool with a predefined OSW ID."""
        pass

    def write_tool_channel_raw_sync(
        self, params: "TSDCMixin.WriteToolChannelRawParams"
    ):
        return asyncio.run(self.write_tool_channel_raw(params=params))

    class DataRow(BaseModel):
        ts: datetime
        """Timestamp of the data row"""
        ch: str
        """Channel OSW ID"""
        data: Any

    class StoreDataParams(BaseModel):
        tool_osw_id: str
        """OSW ID of the tool"""
        rows: List["TSDCMixin.DataRow"]
        """Data rows to store"""

    async def store_data(self, params: "TSDCMixin.StoreDataParams"):
        """Stores data for a tool with a predefined OSW ID."""
        rows = [row.model_dump(mode="json") for row in params.rows]
        return await self.write_tool_channel_raw(
            params=TSDCMixin.WriteToolChannelRawParams(
                tool_osw_id=params.tool_osw_id,
                data=rows,
            )
        )

    def store_data_sync(self, params: "TSDCMixin.StoreDataParams"):
        return asyncio.run(self.store_data(params=params))

    class FilterColumn(str, Enum):
        channel = "ch"
        timestamp = "ts"
        data = "data"

    class FilterOperator(str, Enum):
        eq = "eq"
        gt = "gt"
        gte = "gte"
        lt = "lt"
        lte = "lte"
        neq = "neq"
        like = "like"
        ilike = "ilike"
        match = "match"
        imatch = "imatch"
        in_ = "in"
        is_ = "is"
        isdistinct = "isdistinct"
        fts = "fts"
        plfts = "plfts"
        phfts = "phfts"
        wfts = "wfts"
        cs = "cs"
        cd = "cd"
        ov = "ov"
        sl = "sl"
        sr = "sr"
        nxr = "nxr"
        nxl = "nxl"
        adj = "adj"
        not_ = "not"
        or_ = "or"
        and_ = "and"
        all_ = "all"
        any_ = "any"

    class Filter(BaseModel):
        column: Union["TSDCMixin.FilterColumn", str]
        """Column name or column name + jsonb selector"""
        operator: "TSDCMixin.FilterOperator"
        """Filter operator"""
        criteria: Any
        """Criteria value for the filter"""

    class ReadToolChannelRawParams(BaseModel):
        tool_osw_id: str
        """OSW ID of the tool"""
        channel_osw_id: Optional[str] = None
        """OSW ID of the channel, all are read if None"""
        start: Optional[datetime] = None
        """Start time for reading data"""
        end: Optional[datetime] = None
        """End time for reading data"""
        filter: Optional[List["TSDCMixin.Filter"]] = None
        """Filters for reading data"""
        limit: Optional[int] = None
        """Limit the number of returned rows"""
        downsample: Optional[DownsampleParams] = None
        """Optional server-side downsampling request. None = full resolution."""

    async def read_tool_channel_raw(self, params: "TSDCMixin.ReadToolChannelRawParams"):
        """Retrieve data for a tool within a time range.

        Shared by every driver-backed controller (v1 and v2, Local and
        PostgREST): it builds the filter list and forwards all parameters -
        including the optional server-side downsampling parameters
        (``max_points`` / ``bin_size`` / ``downsample_method`` /
        ``edge_anchors``) - to the backend driver via ``self._driver.read``.
        """
        driver = getattr(self, "_driver", None)
        if driver is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no _driver to read from"
            )
        filters = None
        if params.filter:
            filters = [
                {
                    "column": (
                        f.column.value
                        if isinstance(f.column, TSDCMixin.FilterColumn)
                        else f.column
                    ),
                    "operator": f.operator.value,
                    "criteria": f.criteria,
                }
                for f in params.filter
            ]
        ds = params.downsample
        return await driver.read(
            tool_osw_id=params.tool_osw_id,
            channel_osw_id=params.channel_osw_id,
            start=params.start,
            end=params.end,
            filters=filters,
            limit=params.limit,
            max_points=ds.max_points if ds else None,
            bin_size=ds.bin_size if ds else None,
            downsample_method=ds.method if ds else None,
            edge_anchors=ds.edge_anchors if ds else None,
        )

    def read_tool_channel_raw_sync(self, params: "TSDCMixin.ReadToolChannelRawParams"):
        return asyncio.run(self.read_tool_channel_raw(params=params))

    async def flush_buffer(self, tool_osw_id: Optional[str] = None):
        """Flush any buffered data. No-op for non-buffered implementations."""
        pass


# LocalTSDCMixin and PostgrestTSDCMixin removed.
# Replaced by LocalDatabaseDriver and PostgrestDatabaseDriver in _drivers.py.
# Controllers use driver composition via _driver PrivateAttr.
