"""Integration tests for the downsample_tool_channel RPC.

These run against a live pgstack (TimescaleDB + PostgREST) with the updated
``100_init_tsdb_schema.sql`` applied. They are skipped unless TEST_PGRST_URL
and TEST_PGRST_JWT_SECRET are set, and self-skip when the server or the RPC
is unavailable.

All coroutines run on a single module-scoped event loop so the shared async
PostgREST client (an httpx.AsyncClient) is not torn down between calls.
"""

import asyncio
import datetime as dt
import os
import time

import pytest

from opensemantic.base import (
    DeleteToolParams,
    DownsampleParams,
    ReadToolChannelRawParams,
)

_PGRST_URL = os.environ.get("TEST_PGRST_URL")
_PGRST_JWT_SECRET = os.environ.get("TEST_PGRST_JWT_SECRET")
_PGRST_JWT_ROLE = os.environ.get("TEST_PGRST_JWT_ROLE", "api_user")
_PGRST_SCHEMA = os.environ.get("TEST_PGRST_SCHEMA", "api")
_PGRST_CONFIGURED = bool(_PGRST_URL and _PGRST_JWT_SECRET)

pytestmark = pytest.mark.skipif(
    not _PGRST_CONFIGURED,
    reason="TEST_PGRST_URL and TEST_PGRST_JWT_SECRET not set",
)

WINDOW_SECONDS = 2000
N_POINTS = 2000
MAX_POINTS = 200  # -> 10 s buckets, ~10 points each


def _make_db():
    import jwt
    from postgrest import AsyncPostgrestClient

    from opensemantic.base import PostgrestTimeSeriesDatabaseController
    from opensemantic.core import Label

    token = jwt.encode({"role": _PGRST_JWT_ROLE}, _PGRST_JWT_SECRET, algorithm="HS256")
    client = AsyncPostgrestClient(
        base_url=_PGRST_URL,
        schema=_PGRST_SCHEMA,
        headers={"Authorization": f"Bearer {token}"},
    )
    db = PostgrestTimeSeriesDatabaseController(
        name="downsample_test",
        label=[Label(text="Downsample Test")],
        buffered=False,
    )
    db.set_client(client)
    return db


def _read(
    seeded,
    ch,
    start=None,
    end=None,
    max_points=None,
    bin_size=None,
    downsample_method=None,
    edge_anchors=None,
):
    ds = None
    if max_points is not None or bin_size is not None or downsample_method:
        ds = DownsampleParams(
            max_points=max_points,
            bin_size=bin_size,
            method=downsample_method,
            edge_anchors=edge_anchors,
        )
    return seeded["run"](
        seeded["db"].read_tool_channel_raw(
            ReadToolChannelRawParams(
                tool_osw_id=seeded["tool_id"],
                channel_osw_id=ch,
                start=start,
                end=end,
                downsample=ds,
            )
        )
    )


def _make_controller(db):
    """A DataToolController with scalar, composite and text channels, bound to db."""
    from uuid import uuid4

    from opensemantic import compute_scoped_uuid
    from opensemantic.base.v1 import DataChannel, DataToolController
    from opensemantic.core.v1 import Label

    parent = uuid4()
    ctrl = DataToolController(
        uuid=str(parent),
        name="DownsampleTest",
        label=[Label(text="Test")],
        data_channels=[
            DataChannel(
                uuid=str(compute_scoped_uuid(parent, name)),
                osw_id="placeholder",
                name=name,
                label=[Label(text=name)],
            )
            for name in ("scalar", "composite", "text")
        ],
    )
    ctrl.archive_database = db
    return ctrl


@pytest.fixture(scope="module")
def seeded():
    """Create a tool with a scalar, a composite, and a text channel.

    Seeds via the shared high-level bulk helper (seed_channel_series). One
    event loop is used for setup, all reads and teardown.
    """
    from opensemantic.base._demo_data import seed_channel_series

    loop = asyncio.new_event_loop()

    def run(coro):
        return loop.run_until_complete(coro)

    db = _make_db()
    ctrl = _make_controller(db)
    tool_id = ctrl.get_osw_id()
    base = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)

    def short(name):
        return ctrl.get_channel_by_name(name).get_osw_id().split("#")[-1]

    def _value(channel, i):
        if channel.name == "composite":
            return {
                "temperature": {"value": float(i)},
                "humidity": {"value": float(N_POINTS - i)},
                "comment": "ok",
            }
        if channel.name == "text":
            return {"text": "hello"}
        return {"value": float(i)}

    try:
        run(seed_channel_series(ctrl, n_points=N_POINTS, base_ts=base, value_fn=_value))
    except Exception as e:
        loop.close()
        pytest.skip(f"pgstack unavailable: {e}")

    # Probe the RPC now the tool + data exist; skip the whole module if absent.
    try:
        run(
            db._driver.client.rpc(
                "downsample_tool_channel", {"osw_tool": tool_id}
            ).execute()
        )
    except Exception as e:
        loop.close()
        pytest.skip(f"downsample_tool_channel RPC unavailable: {e}")
    run(asyncio.sleep(1.0))

    yield {
        "run": run,
        "db": db,
        "tool_id": tool_id,
        "ch_scalar": short("scalar"),
        "ch_comp": short("composite"),
        "ch_text": short("text"),
        "start": base,
        "end": base + dt.timedelta(seconds=WINDOW_SECONDS - 1),
    }

    try:
        run(db.delete_tool(DeleteToolParams(tool_osw_id=tool_id)))
    except Exception:
        pass
    loop.close()


def _monotonic(rows):
    ts = [r["ts"] for r in rows]
    return ts == sorted(ts)


def test_sample_count_and_monotonic(seeded):
    rows = _read(
        seeded,
        seeded["ch_scalar"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="sample",
    )
    # ~one point per bucket, plus up to 2 edge anchors.
    assert 0 < len(rows) <= MAX_POINTS + 4
    assert _monotonic(rows)


def test_average_count_and_values(seeded):
    rows = _read(
        seeded,
        seeded["ch_scalar"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="average",
    )
    assert 0 < len(rows) <= MAX_POINTS + 4
    assert _monotonic(rows)
    # Averaged values stay within the seeded range [0, N_POINTS).
    vals = [r["data"]["value"] for r in rows if "value" in r["data"]]
    assert vals and all(0.0 <= float(v) <= float(N_POINTS) for v in vals)


def test_minmax_returns_more_rows_than_sample(seeded):
    sample = _read(
        seeded,
        seeded["ch_scalar"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="sample",
    )
    minmax = _read(
        seeded,
        seeded["ch_scalar"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="minmax",
    )
    # Scalar minmax yields about two rows per bucket.
    assert len(minmax) >= len(sample)
    assert _monotonic(minmax)
    # All returned rows are real stored datapoints (integer-valued here).
    for r in minmax:
        assert float(r["data"]["value"]).is_integer()


def test_composite_sample_preserves_structure(seeded):
    rows = _read(
        seeded,
        seeded["ch_comp"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="sample",
    )
    assert rows
    d = rows[0]["data"]
    assert "temperature" in d and "value" in d["temperature"]
    assert "humidity" in d and "value" in d["humidity"]


def test_composite_average_deep_averages_leaves(seeded):
    rows = _read(
        seeded,
        seeded["ch_comp"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="average",
    )
    assert rows
    d = rows[len(rows) // 2]["data"]
    # Deep-averaged numeric leaves present; non-numeric 'comment' carried.
    assert isinstance(d["temperature"]["value"], (int, float))
    assert isinstance(d["humidity"]["value"], (int, float))
    assert d.get("comment") == "ok"


def test_composite_minmax_returns_real_rows(seeded):
    sample = _read(
        seeded,
        seeded["ch_comp"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="sample",
    )
    minmax = _read(
        seeded,
        seeded["ch_comp"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="minmax",
    )
    assert len(minmax) >= len(sample)
    assert _monotonic(minmax)
    # Composite minmax rows are real stored rows with full structure.
    d = minmax[0]["data"]
    assert "temperature" in d and "humidity" in d


def test_text_channel_falls_back_to_sample(seeded):
    rows = _read(
        seeded,
        seeded["ch_text"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="minmax",
    )
    # No numeric leaf -> falls back to sample; rows are the real text rows.
    assert rows
    assert all(r["data"].get("text") == "hello" for r in rows)


def test_edge_anchors_present(seeded):
    with_anchors = _read(
        seeded,
        seeded["ch_scalar"],
        start=seeded["start"],
        end=seeded["end"],
        max_points=MAX_POINTS,
        downsample_method="sample",
        edge_anchors=True,
    )
    # First/last returned rows are the window's first/last real datapoints.
    assert float(with_anchors[0]["data"]["value"]) == 0.0
    assert float(with_anchors[-1]["data"]["value"]) == float(WINDOW_SECONDS - 1)


@pytest.mark.parametrize("channel_key", ["ch_scalar", "ch_comp"])
def test_benchmark_methods(seeded, record_property, channel_key):
    """Time a full-resolution read against each strategy and report.

    Only deterministic payload invariants are asserted; the wall times are
    attached as test properties (and printed under ``-s``) so a regression is
    visible without tying the suite to machine speed. For real numbers across
    escalating series sizes use ``benchmarks/bench_downsample.py``.
    """
    ch = seeded[channel_key]
    results = {}
    for method in (None, "sample", "average", "minmax"):
        t0 = time.perf_counter()
        rows = _read(
            seeded,
            ch,
            start=seeded["start"],
            end=seeded["end"],
            max_points=None if method is None else MAX_POINTS,
            downsample_method=method,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        name = method or "raw"
        results[name] = (elapsed_ms, len(rows))
        record_property(f"{channel_key}_{name}_ms", round(elapsed_ms, 1))
        record_property(f"{channel_key}_{name}_rows", len(rows))

    print(
        f"\n{channel_key}: "
        + "  ".join(f"{name}={ms:.0f}ms/{n}rows" for name, (ms, n) in results.items())
    )

    raw_rows = results["raw"][1]
    assert raw_rows == N_POINTS
    # Every strategy must actually shrink the payload for this window.
    for method in ("sample", "average", "minmax"):
        assert results[method][1] < raw_rows / 2, f"{method} did not reduce rows"
    # minmax keeps up to two real rows per bucket, so it never returns less
    # than sample, which keeps one.
    assert results["minmax"][1] >= results["sample"][1]
