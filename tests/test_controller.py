import asyncio
import os
import tempfile

import pytest

from opensemantic.base import (
    LocalTimeSeriesDatabaseController,
    TimeSeriesDatabaseController,
)
from opensemantic.base._controller_logic import (
    build_sqlite_read_query,
    check_buffer_duplicates,
    make_osw_id,
    parse_sqlite_rows,
)
from opensemantic.base.v1 import LocalTimeSeriesDatabaseController as LocalTSDC_v1
from opensemantic.base.v1 import TimeSeriesDatabaseController as TSDC_v1
from opensemantic.core._model import Label
from opensemantic.core.v1._model import Label as Label_v1

# -- Controller logic tests --


def test_make_osw_id():
    assert make_osw_id("12345678-1234-1234-1234-123456789abc") == (
        "OSW12345678123412341234123456789abc"
    )


def test_build_sqlite_read_query_basic():
    query, params = build_sqlite_read_query(tool_osw_id="OSW_test")
    assert "FROM OSW_test" in query
    assert "ORDER BY ts ASC" in query
    assert params == []


def test_build_sqlite_read_query_with_filters():
    query, params = build_sqlite_read_query(
        tool_osw_id="OSW_test",
        channel_osw_id="OSW_ch1",
        limit=10,
        filters=[{"column": "data->>'value'", "operator": "gt", "criteria": 5}],
    )
    assert "ch = ?" in query
    assert "LIMIT ?" in query
    assert "data->>'value' > ?" in query
    assert "OSW_ch1" in params
    assert 10 in params
    assert 5 in params


def test_parse_sqlite_rows():
    raw = [(1, "2024-01-01T00:00:00", "ch1", '{"value": 42}')]
    result = parse_sqlite_rows(raw)
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["data"]["value"] == 42


def test_check_buffer_duplicates_none():
    buf = {"tool1": [{"ts": "t1", "ch": "c1", "data": {"v": 1}}]}
    assert check_buffer_duplicates(buf) == {}


def test_check_buffer_duplicates_found():
    row = {"ts": "t1", "ch": "c1", "data": {"v": 1}}
    buf = {"tool1": [row, row]}
    dupes = check_buffer_duplicates(buf)
    assert "tool1" in dupes
    assert len(dupes["tool1"]) == 1


# -- LocalTimeSeriesDatabaseController tests --


@pytest.fixture
def sqlite_db():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    db = LocalTimeSeriesDatabaseController(
        name="test_db",
        label=[Label(text="Test DB", lang="en")],
        db_path=path,
    )
    yield db
    os.unlink(path)


def test_local_db_is_database_subclass(sqlite_db):
    from opensemantic.base._model import Database

    assert isinstance(sqlite_db, Database)


def test_local_db_create_and_list_tools(sqlite_db):
    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_t1")
        )
        tools = await sqlite_db.get_tools_list()
        assert "OSW_t1" in tools

    asyncio.run(_test())


def test_local_db_write_and_read(sqlite_db):
    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_rw")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_rw",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "ch1", "data": {"value": 42.0}},
                    {"ts": "2024-01-01T00:01:00", "ch": "ch2", "data": {"value": 99.0}},
                ],
            )
        )
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id="OSW_rw")
        )
        assert len(rows) == 2
        assert rows[0]["data"]["value"] == 42.0
        assert rows[1]["data"]["value"] == 99.0

    asyncio.run(_test())


def test_local_db_read_with_channel_filter(sqlite_db):
    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_cf")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_cf",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "ch_a", "data": {"v": 1}},
                    {"ts": "2024-01-01T00:00:01", "ch": "ch_b", "data": {"v": 2}},
                ],
            )
        )
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(
                tool_osw_id="OSW_cf",
                channel_osw_id="ch_a",
            )
        )
        assert len(rows) == 1
        assert rows[0]["ch"] == "ch_a"

    asyncio.run(_test())


def test_local_db_delete_tool(sqlite_db):
    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_del")
        )
        assert "OSW_del" in await sqlite_db.get_tools_list()
        await sqlite_db.delete_tool(
            TimeSeriesDatabaseController.DeleteToolParams(tool_osw_id="OSW_del")
        )
        assert "OSW_del" not in await sqlite_db.get_tools_list()

    asyncio.run(_test())


def test_local_db_store_data(sqlite_db):
    async def _test():
        from datetime import datetime

        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_sd")
        )
        await sqlite_db.store_data(
            TimeSeriesDatabaseController.StoreDataParams(
                tool_osw_id="OSW_sd",
                rows=[
                    TimeSeriesDatabaseController.DataRow(
                        ts=datetime(2024, 1, 1), ch="ch1", data={"v": 10}
                    )
                ],
            )
        )
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id="OSW_sd")
        )
        assert len(rows) == 1

    asyncio.run(_test())


def test_local_db_get_table_size(sqlite_db):
    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_sz")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_sz",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "c", "data": {"v": 1}},
                    {"ts": "2024-01-01T00:00:01", "ch": "c", "data": {"v": 2}},
                ],
            )
        )
        size = await sqlite_db.get_table_size("OSW_sz")
        assert size == 2

    asyncio.run(_test())


def test_local_db_timestamp_range_filter(sqlite_db):
    """Test reading with start/end timestamp filters."""

    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_tr")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_tr",
                data=[
                    {
                        "ts": "2024-01-01T12:00:00.000+00:00",
                        "ch": "ch1",
                        "data": {"value": 1},
                    },
                    {
                        "ts": "2024-01-01T12:00:00.100+00:00",
                        "ch": "ch1",
                        "data": {"value": 2},
                    },
                    {
                        "ts": "2024-01-01T12:00:01.000+00:00",
                        "ch": "ch1",
                        "data": {"value": 3},
                    },
                ],
            )
        )
        # Full range
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(
                tool_osw_id="OSW_tr",
                start="2024-01-01T12:00:00.000Z",
                end="2024-01-01T12:00:00.999Z",
            )
        )
        assert len(rows) == 2

        # Narrow range
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(
                tool_osw_id="OSW_tr",
                start="2024-01-01T12:00:00.000Z",
                end="2024-01-01T12:00:00.050Z",
            )
        )
        assert len(rows) == 1

    asyncio.run(_test())


def test_local_db_jsonb_filter(sqlite_db):
    """Test reading with JSONB column filter."""

    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_jb")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_jb",
                data=[
                    {
                        "ts": "2024-01-01T12:00:00.000+00:00",
                        "ch": "ch1",
                        "data": {"value": 42},
                    },
                    {
                        "ts": "2024-01-01T12:00:00.100+00:00",
                        "ch": "ch1",
                        "data": {"value": {"nested": 43}},
                    },
                ],
            )
        )
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(
                tool_osw_id="OSW_jb",
                filter=[
                    TimeSeriesDatabaseController.Filter(
                        column="json_extract(data, '$.value.nested')",
                        operator=TimeSeriesDatabaseController.FilterOperator.eq,
                        criteria=43,
                    )
                ],
            )
        )
        assert len(rows) == 1

    asyncio.run(_test())


def test_local_db_delete_by_ids(sqlite_db):
    """Test deleting specific rows by ID."""

    async def _test():
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id="OSW_dbi")
        )
        await sqlite_db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id="OSW_dbi",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "c", "data": {"v": 1}},
                    {"ts": "2024-01-01T00:00:01", "ch": "c", "data": {"v": 2}},
                ],
            )
        )
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id="OSW_dbi")
        )
        assert len(rows) == 2
        ids = [row["id"] for row in rows]
        await sqlite_db.delete_by_ids("OSW_dbi", ids)
        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id="OSW_dbi")
        )
        assert len(rows) == 0

    asyncio.run(_test())


def test_local_db_periodic_cleanup(sqlite_db):
    """Test concurrent write + periodic cleanup."""
    import datetime

    async def _test():
        tool = "OSW_cleanup"
        await sqlite_db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id=tool)
        )

        async def writer():
            for i in range(20):
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                await sqlite_db.write_tool_channel_raw(
                    TimeSeriesDatabaseController.WriteToolChannelRawParams(
                        tool_osw_id=tool,
                        data=[{"ts": ts, "ch": "ch1", "data": {"value": i}}],
                    )
                )
                await asyncio.sleep(0.05)

        async def cleanup():
            while True:
                rows = await sqlite_db.read_tool_channel_raw(
                    TimeSeriesDatabaseController.ReadToolChannelRawParams(
                        tool_osw_id=tool, limit=5
                    )
                )
                if rows:
                    ids = [row["id"] for row in rows]
                    await sqlite_db.delete_by_ids(tool, ids)
                await asyncio.sleep(0.2)

        writer_task = asyncio.create_task(writer())
        cleanup_task = asyncio.create_task(cleanup())

        await writer_task
        await asyncio.sleep(3)

        if cleanup_task.done() and cleanup_task.exception():
            raise cleanup_task.exception()
        cleanup_task.cancel()

        rows = await sqlite_db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id=tool)
        )
        assert len(rows) == 0, f"Expected 0 rows after cleanup, got {len(rows)}"

    asyncio.run(_test())


# -- PostgREST integration tests --
# These require a running PostgREST instance (pgstack).
# Set TEST_PGRST_URL and TEST_PGRST_JWT_SECRET env vars to enable.
# See .env.example for configuration.

_PGRST_URL = os.environ.get("TEST_PGRST_URL")
_PGRST_JWT_SECRET = os.environ.get("TEST_PGRST_JWT_SECRET")
_PGRST_JWT_ROLE = os.environ.get("TEST_PGRST_JWT_ROLE", "api_user")
_PGRST_SCHEMA = os.environ.get("TEST_PGRST_SCHEMA", "api")
_PGRST_CONFIGURED = bool(_PGRST_URL and _PGRST_JWT_SECRET)


def _get_postgrest_db(buffered=False, **kwargs):
    """Create a PostgrestTimeSeriesDatabaseController from env vars."""
    import jwt
    from postgrest import AsyncPostgrestClient

    from opensemantic.base import PostgrestTimeSeriesDatabaseController

    token = jwt.encode({"role": _PGRST_JWT_ROLE}, _PGRST_JWT_SECRET, algorithm="HS256")
    client = AsyncPostgrestClient(
        base_url=_PGRST_URL,
        schema=_PGRST_SCHEMA,
        headers={"Authorization": f"Bearer {token}"},
    )
    db = PostgrestTimeSeriesDatabaseController(
        name="pgrst_test",
        label=[Label(text="PostgREST Test")],
        buffered=buffered,
        **kwargs,
    )
    db.set_client(client)
    return db


@pytest.mark.skipif(
    not _PGRST_CONFIGURED,
    reason="TEST_PGRST_URL and TEST_PGRST_JWT_SECRET not set",
)
def test_postgrest_crud():
    """Full CRUD cycle via PostgREST."""

    async def _test():
        db = _get_postgrest_db()
        tool_id = "OSWaad8e2eaa5b0412da008182386ebab68"
        ch_id = "OSWaad1e2eaa5b0412da008182386ebab68"

        # Clean up if exists
        tools = await db.get_tools_list()
        if tool_id in tools:
            await db.delete_tool(
                TimeSeriesDatabaseController.DeleteToolParams(tool_osw_id=tool_id)
            )

        # Create
        c_res = await db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id=tool_id)
        )
        assert tool_id in c_res.data
        tools = await db.get_tools_list()
        assert tool_id in tools
        await asyncio.sleep(1.0)

        # Write
        data_rows = [
            {"ts": "2023-10-01T12:00:00Z", "ch": ch_id, "data": {"value": 23.4}}
        ]
        await db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id=tool_id, data=data_rows
            )
        )

        # Read
        loaded = await db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id=tool_id)
        )
        assert len(loaded) == 1
        assert loaded[0]["data"]["value"] == 23.4

        # JSONB filter
        res = (
            await db._driver.client.table(tool_id)
            .select("*")
            .filter("data->>value", "eq", 23.4)
            .execute()
        )
        assert len(res.data) == 1

        # Tool config
        tool_config = await db.get_tool_config()
        assert any(t["osw_id"] == tool_id for t in tool_config)

        # Delete
        d_res = await db.delete_tool(
            TimeSeriesDatabaseController.DeleteToolParams(tool_osw_id=tool_id)
        )
        assert tool_id in d_res.data
        tools = await db.get_tools_list()
        assert tool_id not in tools

    asyncio.run(_test())


@pytest.mark.skipif(
    not _PGRST_CONFIGURED,
    reason="TEST_PGRST_URL and TEST_PGRST_JWT_SECRET not set",
)
def test_postgrest_offline_buffer():
    """Offline buffer + sync."""

    async def _test():
        db = _get_postgrest_db(buffered=True, buffer_batch_size=1)
        await db.start_offline_sync()

        tool_id = "OSWbad8e2eaa5b0412da008182386ebab68"
        ch_id = "OSWbad1e2eaa5b0412da008182386ebab68"

        # Clean up
        tools = await db.get_tools_list()
        if tool_id in tools:
            await db.delete_tool(
                TimeSeriesDatabaseController.DeleteToolParams(tool_osw_id=tool_id)
            )

        await db.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id=tool_id)
        )
        await asyncio.sleep(1.0)

        # Write online
        await db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id=tool_id,
                data=[
                    {"ts": "2023-10-01T12:00:00Z", "ch": ch_id, "data": {"value": 23.4}}
                ],
            )
        )

        # Go offline
        db._driver._emulate_offline = True
        await db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id=tool_id,
                data=[
                    {"ts": "2023-10-01T12:05:00Z", "ch": ch_id, "data": {"value": 25.6}}
                ],
            )
        )

        # Come back online
        db._driver._emulate_offline = False
        await db.write_tool_channel_raw(
            TimeSeriesDatabaseController.WriteToolChannelRawParams(
                tool_osw_id=tool_id,
                data=[
                    {"ts": "2023-10-01T12:10:00Z", "ch": ch_id, "data": {"value": 27.8}}
                ],
            )
        )

        # Wait for offline buffer sync
        await asyncio.sleep(5.0)

        loaded = await db.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(tool_osw_id=tool_id)
        )
        values = [row["data"]["value"] for row in loaded]
        assert 23.4 in values, "Online write missing"
        assert 25.6 in values, "Offline buffered write missing"
        assert 27.8 in values, "Post-offline write missing"

        # Cleanup
        await db.delete_tool(
            TimeSeriesDatabaseController.DeleteToolParams(tool_osw_id=tool_id)
        )

    asyncio.run(_test())


# -- v1 controller tests --


def test_v1_tsdc_subclasses_v1_database():
    from opensemantic.base.v1._model import Database as Database_v1

    assert issubclass(TSDC_v1, Database_v1)


def test_v1_tsdc_not_subclass_of_v2_database():
    from opensemantic.base._model import Database as Database_v2

    assert not issubclass(TSDC_v1, Database_v2)


def test_v2_tsdc_subclasses_v2_database():
    from opensemantic.base._model import Database as Database_v2

    assert issubclass(TimeSeriesDatabaseController, Database_v2)


def test_v2_tsdc_not_subclass_of_v1_database():
    from opensemantic.base.v1._model import Database as Database_v1

    assert not issubclass(TimeSeriesDatabaseController, Database_v1)


@pytest.fixture
def sqlite_db_v1():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    db = LocalTSDC_v1(
        name="test_db_v1",
        label=[Label_v1(text="Test DB v1", lang="en")],
        db_path=path,
    )
    yield db
    os.unlink(path)


def test_v1_local_db_is_v1_database_subclass(sqlite_db_v1):
    from opensemantic.base.v1._model import Database as Database_v1

    assert isinstance(sqlite_db_v1, Database_v1)


def test_v1_local_db_crud(sqlite_db_v1):
    async def _test():
        await sqlite_db_v1.create_tool(TSDC_v1.CreateToolParams(tool_osw_id="OSW_v1t"))
        tools = await sqlite_db_v1.get_tools_list()
        assert "OSW_v1t" in tools

        await sqlite_db_v1.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_v1t",
                data=[{"ts": "2024-01-01T00:00:00", "ch": "c1", "data": {"v": 7}}],
            )
        )
        rows = await sqlite_db_v1.read_tool_channel_raw(
            TSDC_v1.ReadToolChannelRawParams(tool_osw_id="OSW_v1t")
        )
        assert len(rows) == 1
        assert rows[0]["data"]["v"] == 7

        await sqlite_db_v1.delete_tool(TSDC_v1.DeleteToolParams(tool_osw_id="OSW_v1t"))
        assert "OSW_v1t" not in await sqlite_db_v1.get_tools_list()

    asyncio.run(_test())


# -- DataToolController tests --


def test_datatool_controller_get_all_channels():
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel

    ch1 = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    ch2 = DataChannel(uuid=str(uuid4()), osw_id="ch2", name="ch2")
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch1, ch2],
    )
    assert len(dt.get_all_channels()) == 2


def test_datatool_controller_get_subdevices():
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel

    ch = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    sub = DataToolController(
        name="sub",
        label=[Label(text="Sub")],
        data_channels=[ch],
    )
    parent = DataToolController(
        name="parent",
        label=[Label(text="Parent")],
        data_channels=[],
        subdevices=[sub],
    )
    assert len(parent.get_subdevices()) == 1
    assert len(parent.get_all_channels()) == 1


def test_datatool_controller_get_channel_owner():
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel

    ch1 = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    ch2 = DataChannel(uuid=str(uuid4()), osw_id="ch2", name="ch2")
    sub = DataToolController(
        name="sub",
        label=[Label(text="Sub")],
        data_channels=[ch2],
    )
    parent = DataToolController(
        name="parent",
        label=[Label(text="Parent")],
        data_channels=[ch1],
        subdevices=[sub],
    )
    assert parent.get_channel_owner(ch1).name == "parent"
    assert parent.get_channel_owner(ch2).name == "sub"


def test_datatool_controller_osw_id():
    from opensemantic.base import DataToolController

    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
    )
    osw_id = dt.get_osw_id()
    assert osw_id.startswith("OSW")
    assert "-" not in osw_id


def test_datatool_controller_subobject_ids():
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel

    ch = DataChannel(uuid=str(uuid4()), osw_id="temp", name="ch1")
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch],
    )
    parent_osw = dt.get_osw_id()
    ch_osw = dt.data_channels[0].osw_id
    assert ch_osw.startswith(parent_osw + "#")


def test_datatool_controller_auto_archive_from_storage():
    from opensemantic.base import Database, DataToolController

    db = Database(name="auto_archive_test", label=[Label(text="Archive")])
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        storage_locations=[db],
        auto_archive=True,
    )
    assert dt.archive_database is not None
    assert dt.archive_database.name == "auto_archive_test"
    import os

    db_path = dt.archive_database.db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_datatool_controller_explicit_archive_not_overwritten():
    from opensemantic.base import Database, DataToolController

    db = Database(name="ignored", label=[Label(text="Ignored")])
    explicit = LocalTimeSeriesDatabaseController(
        name="explicit",
        label=[Label(text="Explicit")],
        db_path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name,
    )
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        storage_locations=[db],
        archive_database=explicit,
        auto_archive=True,
    )
    assert dt.archive_database.name == "explicit"


def test_datatool_controller_handle_data_change():
    """Test _handle_data_change archives data correctly."""
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel

    ch = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    archive = LocalTimeSeriesDatabaseController(
        name="hdc_test",
        label=[Label(text="Test")],
        db_path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name,
    )
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch],
        archive_database=archive,
        auto_archive=True,
    )

    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)

    async def _test():
        await archive.create_tool(
            TimeSeriesDatabaseController.CreateToolParams(tool_osw_id=dt.get_osw_id())
        )
        from opensemantic.base._controller_mixin import DataToolMixin

        await dt._handle_data_change(
            DataToolMixin.ChannelDataChangeNotificationParams(
                channel=dt.data_channels[0],
                value=42.0,
                timestamp=now,
            )
        )
        rows = await archive.read_tool_channel_raw(
            TimeSeriesDatabaseController.ReadToolChannelRawParams(
                tool_osw_id=dt.get_osw_id()
            )
        )
        assert len(rows) == 1
        assert rows[0]["data"]["value"] == 42.0

    asyncio.run(_test())

    import os

    db_path = archive.db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_datatool_controller_typed_write_read():
    """Test typed write/read with Temperature characteristic."""
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._model import DataChannel
    from opensemantic.characteristics.quantitative import (
        Temperature,
        TemperatureUnit,
    )

    ch = DataChannel(
        uuid=str(uuid4()),
        osw_id="ch_typed",
        name="typed_ch",
        characteristic=Temperature.get_cls_iri(),
    )
    archive = LocalTimeSeriesDatabaseController(
        name="typed_base_test",
        label=[Label(text="Test")],
        db_path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name,
    )
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch],
        archive_database=archive,
    )

    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)

    async def _test():
        await dt.store_channel_data(
            dt.StoreChannelDataParams(
                channel="typed_ch",
                value=Temperature(value=300.0, unit=TemperatureUnit.kelvin),
                timestamp=now,
            )
        )
        results = await dt.load_channel_data(
            dt.LoadChannelDataParams(
                channel="typed_ch",
                target_schema=Temperature,
            )
        )
        assert len(results) == 1
        assert hasattr(results[0], "value")
        assert results[0].value.value == pytest.approx(300.0)

    asyncio.run(_test())

    import os

    db_path = archive.db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_datatool_controller_configure_auto_archive():
    """Test configure_auto_archive creates tool tables."""
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._controller_mixin import DataToolMixin
    from opensemantic.base._model import DataChannel

    ch = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    archive = LocalTimeSeriesDatabaseController(
        name="cfg_test",
        label=[Label(text="Test")],
        db_path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name,
    )
    dt = DataToolController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch],
        archive_database=archive,
    )

    async def _test():
        await dt.configure_auto_archive(DataToolMixin.AutoArchiveParams(enable=True))
        tools = await archive.get_tools_list()
        assert dt.get_osw_id() in tools

    asyncio.run(_test())

    import os

    db_path = archive.db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


# -- store_channel_data / load_channel_data tests --


def _make_controller_with_channels():
    """Helper: create a DataToolController with temp + pressure channels.
    Uses a random UUID each time to avoid test interference."""
    from uuid import uuid4

    from opensemantic import compute_scoped_uuid
    from opensemantic.base.v1 import (
        Database,
        DataChannel,
        DataTool,
        DataToolController,
    )
    from opensemantic.characteristics.quantitative.v1 import Temperature
    from opensemantic.core.v1 import Label as LabelV1

    parent_uuid = uuid4()
    dt = DataTool(
        uuid=parent_uuid,
        name="StoreLoadTest",
        label=[LabelV1(text="Test")],
        data_channels=[
            DataChannel(
                uuid=str(compute_scoped_uuid(parent_uuid, "temp")),
                osw_id="placeholder",
                name="temperature",
                label=[LabelV1(text="Temp")],
                characteristic=Temperature.get_cls_iri(),
            ),
            DataChannel(
                uuid=str(compute_scoped_uuid(parent_uuid, "press")),
                osw_id="placeholder",
                name="pressure",
                label=[LabelV1(text="Press")],
            ),
        ],
        storage_locations=[
            Database(name="test_archive", label=[LabelV1(text="Archive")]),
        ],
    )
    ctrl = DataToolController(dt, auto_archive=True)
    return ctrl


def test_store_channel_data_by_name():
    """Store by channel name string, load raw from pressure (no characteristic)."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        # Use pressure channel (no characteristic) so load returns raw dict
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="pressure",
                value={"value": 23.5},
                timestamp=now,
            )
        )
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="pressure")
        )
        assert len(results) == 1
        assert results[0].value["value"] == 23.5

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_store_channel_data_bulk():
    """Bulk-store many points across two channels; load returns them all."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        base = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        n = 2500  # 2 channels * 2500 = 5000 rows, chunked below
        ts = [base + datetime.timedelta(seconds=i) for i in range(n)]
        written = await ctrl.store_channel_data_bulk(
            ctrl.StoreChannelDataBulkParams(
                series=[
                    ctrl.StoreChannelSeriesParams(
                        channel="pressure",
                        timestamps=ts,
                        values=[{"value": float(i)} for i in range(n)],
                    ),
                    ctrl.StoreChannelSeriesParams(
                        channel="temperature",
                        timestamps=ts,
                        values=[{"value": float(i)} for i in range(n)],
                    ),
                ],
                chunk_size=1000,
            )
        )
        assert written == 2 * n
        # pressure has no characteristic -> raw dicts back
        rows = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="pressure")
        )
        assert len(rows) == n
        assert sorted(r.value["value"] for r in rows) == [float(i) for i in range(n)]

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_store_channel_data_typed():
    """Store Temperature instance, verify base unit conversion."""
    import datetime

    from opensemantic.characteristics.quantitative.v1 import (
        Temperature,
        TemperatureUnit,
    )

    ctrl = _make_controller_with_channels()

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="temperature",
                value=Temperature(value=300.0, unit=TemperatureUnit.kelvin),
                timestamp=now,
            )
        )
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(
                channel="temperature",
                target_schema=Temperature,
            )
        )
        assert len(results) == 1
        assert hasattr(results[0].value, "value")
        assert results[0].value.value == pytest.approx(300.0)

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_load_channel_data_auto_typed():
    """Load without target_schema, auto-resolve from channel characteristic."""
    import datetime

    from opensemantic.characteristics.quantitative.v1 import Temperature

    ctrl = _make_controller_with_channels()

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="temperature",
                value=Temperature(value=295.0),
                timestamp=now,
            )
        )
        # No target_schema - should resolve from characteristic IRI
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="temperature")
        )
        assert len(results) == 1
        # Auto-resolved class may be v2 Temperature from _types registry
        assert hasattr(results[0].value, "value")
        assert results[0].value.value == pytest.approx(295.0)

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_load_channel_data_raw():
    """Load channel without characteristic returns raw dicts."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="pressure",
                value={"value": 1013.25, "unit": "hPa"},
                timestamp=now,
            )
        )
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="pressure")
        )
        assert len(results) == 1
        assert isinstance(results[0].value, dict)
        assert results[0].value["value"] == 1013.25

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_auto_create_table_on_first_write():
    """No manual create_tool needed - table auto-created on first write."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        # No create_tool call - should work anyway
        now = datetime.datetime.now(datetime.timezone.utc)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="temperature",
                value={"value": 20.0},
                timestamp=now,
            )
        )
        tools = await ctrl.archive_database.get_tools_list()
        assert ctrl.get_osw_id() in tools

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_channel_name_not_found():
    """Raises ValueError for unknown channel name."""
    ctrl = _make_controller_with_channels()
    with pytest.raises(ValueError, match="No channel named"):
        ctrl.get_channel_by_name("nonexistent")
    _cleanup_archive(ctrl)


def test_store_load_roundtrip():
    """Store typed, load typed, verify equality."""
    import datetime

    from opensemantic.characteristics.quantitative.v1 import (
        Temperature,
        TemperatureUnit,
    )

    ctrl = _make_controller_with_channels()

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        original = Temperature(value=298.15, unit=TemperatureUnit.kelvin)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="temperature",
                value=original,
                timestamp=now,
            )
        )
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(
                channel="temperature",
                target_schema=Temperature,
            )
        )
        assert len(results) == 1
        loaded = results[0].value
        assert loaded.value == pytest.approx(original.value)
        assert loaded.unit == original.unit

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_v2_datatool_store_load_auto_typed():
    """v2 DataToolController: store + load with auto-typed resolution."""
    import datetime
    from uuid import uuid4 as _uuid4

    from opensemantic import compute_scoped_uuid
    from opensemantic.base import (
        DataChannel,
        DataTool,
        DataToolController,
        LocalTimeSeriesDatabaseController,
    )
    from opensemantic.characteristics.quantitative import Temperature
    from opensemantic.core import Label

    parent = _uuid4()
    dt = DataTool(
        uuid=parent,
        name="V2Test",
        label=[Label(text="V2")],
        data_channels=[
            DataChannel(
                uuid=str(compute_scoped_uuid(parent, "temp")),
                osw_id="placeholder",
                name="temperature",
                label=[Label(text="Temp")],
                characteristic=Temperature.get_cls_iri(),
            ),
        ],
    )
    archive = LocalTimeSeriesDatabaseController(
        name="v2test",
        label=[Label(text="V2 Archive")],
        db_path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name,
    )
    ctrl = DataToolController(dt)
    ctrl.archive_database = archive
    ctrl.auto_archive = True

    async def _test():
        now = datetime.datetime.now(datetime.timezone.utc)
        await ctrl.store_channel_data(
            ctrl.StoreChannelDataParams(
                channel="temperature",
                value=Temperature(value=300.0),
                timestamp=now,
            )
        )
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="temperature")
        )
        assert len(results) == 1
        assert hasattr(results[0].value, "value")
        assert results[0].value.value == pytest.approx(300.0)

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def _cleanup_archive(ctrl):
    """Clean up SQLite file created by auto-init."""
    import os
    from pathlib import Path

    db_path = getattr(ctrl.archive_database, "db_path", None)
    if db_path and Path(db_path).exists():
        os.unlink(db_path)


# -- Buffered LocalDriver tests --


@pytest.fixture
def buffered_sqlite_db():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    db = LocalTSDC_v1(
        name="test_buf",
        label=[Label_v1(text="Buffered Test", lang="en")],
        db_path=path,
        buffered=True,
        buffer_batch_size=5,
    )
    yield db
    os.unlink(path)


def test_buffered_driver_init(buffered_sqlite_db):
    """Buffered driver is initialized with correct settings."""
    assert buffered_sqlite_db._driver.buffered is True
    assert buffered_sqlite_db._driver.buffer_batch_size == 5


def test_buffered_write_does_not_persist_below_batch(buffered_sqlite_db):
    """Rows below batch size stay in buffer, not in DB."""

    async def _test():
        db = buffered_sqlite_db
        await db.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_buf",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "c1", "data": {"v": i}}
                    for i in range(3)
                ],
            )
        )
        # 3 rows < batch_size=5, should still be in buffer
        assert len(db._driver._buffer.get("OSW_buf", [])) == 3
        # Table not created yet since nothing flushed
        tools = await db.get_tools_list()
        assert "OSW_buf" not in tools

    asyncio.run(_test())


def test_buffered_write_flushes_at_batch_size(buffered_sqlite_db):
    """Rows are flushed when batch size is reached."""

    async def _test():
        db = buffered_sqlite_db
        await db.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_buf2",
                data=[
                    {"ts": f"2024-01-01T00:00:0{i}", "ch": "c1", "data": {"v": i}}
                    for i in range(6)
                ],
            )
        )
        # 6 rows >= batch_size=5, should have flushed
        rows = await db.read_tool_channel_raw(
            TSDC_v1.ReadToolChannelRawParams(tool_osw_id="OSW_buf2")
        )
        assert len(rows) == 6
        assert len(db._driver._buffer.get("OSW_buf2", [])) == 0

    asyncio.run(_test())


def test_buffered_flush_buffer_persists_remaining(buffered_sqlite_db):
    """flush_buffer() persists all remaining buffered rows."""

    async def _test():
        db = buffered_sqlite_db
        await db.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_buf3",
                data=[
                    {"ts": "2024-01-01T00:00:00", "ch": "c1", "data": {"v": 1}},
                    {"ts": "2024-01-01T00:00:01", "ch": "c1", "data": {"v": 2}},
                ],
            )
        )
        # 2 rows < batch_size=5, table not created yet
        tools = await db.get_tools_list()
        assert "OSW_buf3" not in tools

        await db.flush_buffer()
        rows = await db.read_tool_channel_raw(
            TSDC_v1.ReadToolChannelRawParams(tool_osw_id="OSW_buf3")
        )
        assert len(rows) == 2

    asyncio.run(_test())


def test_buffered_flush_specific_tool(buffered_sqlite_db):
    """flush_buffer(tool_id) only flushes that tool's buffer."""

    async def _test():
        db = buffered_sqlite_db
        await db.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_a",
                data=[{"ts": "2024-01-01T00:00:00", "ch": "c1", "data": {"v": 1}}],
            )
        )
        await db.write_tool_channel_raw(
            TSDC_v1.WriteToolChannelRawParams(
                tool_osw_id="OSW_b",
                data=[{"ts": "2024-01-01T00:00:00", "ch": "c1", "data": {"v": 2}}],
            )
        )
        await db.flush_buffer("OSW_a")

        rows_a = await db.read_tool_channel_raw(
            TSDC_v1.ReadToolChannelRawParams(tool_osw_id="OSW_a")
        )
        assert len(rows_a) == 1

        # OSW_b should still be only in buffer
        assert len(db._driver._buffer.get("OSW_b", [])) == 1

    asyncio.run(_test())


def test_buffered_multiple_flushes(buffered_sqlite_db):
    """Multiple write + flush cycles accumulate correctly."""

    async def _test():
        db = buffered_sqlite_db
        for batch in range(3):
            await db.write_tool_channel_raw(
                TSDC_v1.WriteToolChannelRawParams(
                    tool_osw_id="OSW_multi",
                    data=[
                        {
                            "ts": f"2024-01-01T00:0{batch}:0{i}",
                            "ch": "c1",
                            "data": {"v": batch * 10 + i},
                        }
                        for i in range(3)
                    ],
                )
            )
            await db.flush_buffer()

        rows = await db.read_tool_channel_raw(
            TSDC_v1.ReadToolChannelRawParams(tool_osw_id="OSW_multi")
        )
        assert len(rows) == 9

    asyncio.run(_test())


# -- DataToolController set_buffered / flush_buffer tests --


def test_set_buffered_enables_buffering():
    """set_buffered(True) enables buffered writes on the driver."""
    ctrl = _make_controller_with_channels()

    async def _test():
        await ctrl.set_buffered(True, batch_size=50)
        assert ctrl.archive_database._driver.buffered is True
        assert ctrl.archive_database._driver.buffer_batch_size == 50

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_set_buffered_false_flushes():
    """set_buffered(False) flushes any pending data."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        await ctrl.set_buffered(True, batch_size=1000)
        now = datetime.datetime.now(datetime.timezone.utc)
        for i in range(5):
            await ctrl.store_channel_data(
                ctrl.StoreChannelDataParams(
                    channel="pressure",
                    value={"value": float(i)},
                    timestamp=now + datetime.timedelta(seconds=i),
                )
            )
        # Data is in buffer, not persisted
        pending = ctrl.archive_database._driver._pending_count()
        assert pending == 5

        # Disabling flushes automatically
        await ctrl.set_buffered(False)
        assert ctrl.archive_database._driver.buffered is False

        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="pressure")
        )
        assert len(results) == 5

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_flush_buffer_on_controller():
    """ctrl.flush_buffer() persists buffered data."""
    import datetime

    ctrl = _make_controller_with_channels()

    async def _test():
        await ctrl.set_buffered(True, batch_size=1000)
        now = datetime.datetime.now(datetime.timezone.utc)
        for i in range(10):
            await ctrl.store_channel_data(
                ctrl.StoreChannelDataParams(
                    channel="pressure",
                    value={"value": float(i)},
                    timestamp=now + datetime.timedelta(seconds=i),
                )
            )
        await ctrl.flush_buffer()
        results = await ctrl.load_channel_data(
            ctrl.LoadChannelDataParams(channel="pressure")
        )
        assert len(results) == 10

    asyncio.run(_test())
    _cleanup_archive(ctrl)


def test_buffered_performance():
    """Buffered mode is significantly faster than unbuffered for bulk writes."""
    import datetime
    import time

    N = 200

    async def _bench(buffered):
        ctrl = _make_controller_with_channels()
        if buffered:
            await ctrl.set_buffered(True, batch_size=100)
        now = datetime.datetime.now(datetime.timezone.utc)
        t0 = time.perf_counter()
        for i in range(N):
            await ctrl.store_channel_data(
                ctrl.StoreChannelDataParams(
                    channel="pressure",
                    value={"value": float(i)},
                    timestamp=now + datetime.timedelta(seconds=i),
                )
            )
        if buffered:
            await ctrl.flush_buffer()
        elapsed = time.perf_counter() - t0
        _cleanup_archive(ctrl)
        return elapsed

    t_unbuf = asyncio.run(_bench(False))
    t_buf = asyncio.run(_bench(True))
    # Buffered should be at least 5x faster
    assert (
        t_buf < t_unbuf / 5
    ), f"Buffered ({t_buf:.3f}s) not 5x faster than unbuffered ({t_unbuf:.3f}s)"


# -- Offline flag forwarding tests --


def _make_offline_archive(**kwargs):
    """PostgREST controller without a client, offline buffer in a temp file."""
    from opensemantic.base import PostgrestTimeSeriesDatabaseController

    return PostgrestTimeSeriesDatabaseController(
        name="offline_test",
        label=[Label(text="Offline Test")],
        buffer_offline_location=tempfile.NamedTemporaryFile(
            suffix=".sqlite", delete=False
        ).name,
        **kwargs,
    )


def test_postgrest_controller_forwards_offline_flag():
    """The driver's _offline flag must be readable on the controller."""
    db = _make_offline_archive()
    assert db._offline is False
    db._driver._offline = True
    assert db._offline is True


def test_postgrest_controller_forwards_emulate_offline_flag():
    """_emulate_offline is settable on the controller and reaches the driver."""
    db = _make_offline_archive()
    assert db._emulate_offline is False
    db._emulate_offline = True
    assert db._driver._emulate_offline is True


def test_handle_data_change_calls_on_archive_error_when_offline():
    """Going offline during archiving must trigger the _on_archive_error hook."""
    import datetime
    from uuid import uuid4

    from opensemantic.base import DataToolController
    from opensemantic.base._controller_mixin import DataToolMixin
    from opensemantic.base._model import DataChannel

    class RecordingController(DataToolController):
        def _on_archive_error(self):
            self._archive_errors = getattr(self, "_archive_errors", 0) + 1

    archive = _make_offline_archive(buffered=True, buffer_batch_size=1)
    archive._emulate_offline = True

    ch = DataChannel(uuid=str(uuid4()), osw_id="ch1", name="ch1")
    ctrl = RecordingController(
        name="test",
        label=[Label(text="Test")],
        data_channels=[ch],
        auto_archive=True,
    )
    ctrl.archive_database = archive

    async def _test():
        await ctrl._handle_data_change(
            DataToolMixin.ChannelDataChangeNotificationParams(
                channel=ch,
                value=42.0,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            )
        )

    asyncio.run(_test())

    assert archive._offline is True
    assert getattr(ctrl, "_archive_errors", 0) == 1

    if os.path.exists(archive.buffer_offline_location):
        os.unlink(archive.buffer_offline_location)


# -- Component hierarchy tests --


class _FakeOsw:
    """Minimal stand-in for osw.core.OSW used by load_from_osw."""

    class LoadEntityParam:
        def __init__(self, titles, autofetch_schema=True, model_to_use=None):
            self.titles = titles if isinstance(titles, list) else [titles]
            self.autofetch_schema = autofetch_schema
            self.model_to_use = model_to_use

    class _Result:
        def __init__(self, entities):
            self.entities = entities

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def load_entity(self, param):
        title = param.titles[0]
        self.calls.append((title, param.model_to_use, param.autofetch_schema))
        return self._Result([param.model_to_use(**self.pages[title])])


def _osw_fixture():
    from uuid import uuid4

    from opensemantic.base._model import Component, DataChannel

    def _channel(name):
        return DataChannel(uuid=str(uuid4()), osw_id=name, name=name)

    def _component(instance_iri, type_iri):
        return Component(
            uuid=str(uuid4()),
            component_id="Component01",
            component_instance=instance_iri,
            component_type=type_iri,
        )

    pages = {
        "Item:OSWroot": {
            "name": "root",
            "label": [Label(text="Root")],
            "data_channels": [_channel("root_ch")],
            "components": [
                _component("Item:OSWchildA", "Category:OSWtypeA"),
                _component("Item:OSWchildB", "Category:OSWtypeB"),
            ],
        },
        "Item:OSWchildA": {
            "name": "child_a",
            "label": [Label(text="Child A")],
            "data_channels": [_channel("a_ch")],
            "components": [_component("Item:OSWgrandchild", "Category:OSWtypeA")],
        },
        "Item:OSWchildB": {
            "name": "child_b",
            "label": [Label(text="Child B")],
            "data_channels": [_channel("b_ch")],
        },
        "Item:OSWgrandchild": {
            "name": "grandchild",
            "label": [Label(text="Grandchild")],
            "data_channels": [_channel("g_ch")],
        },
    }
    return _FakeOsw(pages)


def test_load_from_osw_builds_subdevice_tree():
    from opensemantic.base import DataToolController

    osw = _osw_fixture()
    root = DataToolController.load_from_osw(osw, "Item:OSWroot")

    assert root.name == "root"
    assert [sub.name for sub in root.subdevices] == ["child_a", "child_b"]
    assert {sub.name for sub in root.get_subdevices()} == {
        "child_a",
        "child_b",
        "grandchild",
    }
    assert {ch.name for ch in root.get_all_channels()} == {
        "root_ch",
        "a_ch",
        "b_ch",
        "g_ch",
    }


def test_load_from_osw_indexes_subdevice_channels():
    """The channel dict has to cover the subdevices wired in after __init__."""
    from opensemantic.base import DataToolController

    osw = _osw_fixture()
    root = DataToolController.load_from_osw(osw, "Item:OSWroot")

    indexed = {ch.name for ch in root._channel_dict.values()}
    assert indexed == {"root_ch", "a_ch", "b_ch", "g_ch"}


def test_load_from_osw_never_autofetches_schemas():
    """Schemas come from the installed packages, not from ad-hoc generation."""
    from opensemantic.base import DataToolController

    osw = _osw_fixture()
    DataToolController.load_from_osw(osw, "Item:OSWroot")

    assert osw.calls, "no entity was loaded"
    assert all(autofetch is False for _, _, autofetch in osw.calls)


def test_load_from_osw_selects_model_per_component_type():
    from opensemantic.base import DataToolController

    class TypeAController(DataToolController):
        pass

    osw = _osw_fixture()
    root = DataToolController.load_from_osw(
        osw,
        "Item:OSWroot",
        model_by_component_type={"Category:OSWtypeA": TypeAController},
    )

    by_name = {sub.name: sub for sub in root.get_subdevices()}
    assert isinstance(by_name["child_a"], TypeAController)
    assert isinstance(by_name["grandchild"], TypeAController)
    assert not isinstance(by_name["child_b"], TypeAController)


def test_load_from_osw_respects_depth_and_extra_kwargs():
    from opensemantic.base import DataToolController

    osw = _osw_fixture()
    root = DataToolController.load_from_osw(
        osw, "Item:OSWroot", depth=1, auto_archive=False
    )

    assert {sub.name for sub in root.get_subdevices()} == {"child_a", "child_b"}
    assert root.subdevices[0].subdevices == []
