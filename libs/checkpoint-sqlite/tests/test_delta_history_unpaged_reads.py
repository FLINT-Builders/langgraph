"""Specs for SQLite delta-history reads.

`SqliteSaver.get_delta_channel_history` and
`AsyncSqliteSaver.aget_delta_channel_history` currently run one unpaged
query that returns every checkpoint blob for the thread while the saver
lock is held, then bind every ancestor id into one stage-2 statement.

These tests are the contract for a fix:

* A range scan must not copy checkpoint blobs that are not on the ancestor
  chain. Forked checkpoints in the same id range are the case that matters.
* The saver lock must be released between pages of that scan. One hold may
  stream at most `MAX_CHECKPOINTS_PER_LOCK_HOLD` checkpoint rows (the same
  bound `BasePostgresSaver` uses) so another thread's read can run before
  the scan finishes.
* Stage 2 must succeed when the ancestor chain, across the requested
  channels, needs more bound parameters than SQLite's compile-time
  `MAX_VARIABLE_NUMBER`. That limit cannot be raised at runtime.

They fail while the current queries are in place.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Same bound as langgraph.checkpoint.postgres.base._DELTA_PAGE_SIZE.
MAX_CHECKPOINTS_PER_LOCK_HOLD = 1024
OFF_PATH_BLOB_BYTES = 100_000
FORKS = 16


def _cid(index: int) -> str:
    return f"{index:08d}-0000-4000-8000-000000000000"


def _off_path_cid(index: int) -> str:
    """Id that sorts after checkpoint 1 and before checkpoint 2."""
    return f"00000001-ffff-4000-8000-{index:012d}"


def _config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoint_ns": "",
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(
    index: int, channel_values: Mapping[str, Any], *, versions: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "v": 4,
        "ts": "2024-01-01T00:00:00+00:00",
        "id": _cid(index),
        "channel_values": dict(channel_values),
        "channel_versions": dict(versions),
        "versions_seen": {},
    }


def _put_chain(
    saver: SqliteSaver,
    *,
    thread_id: str,
    length: int,
    channels: Sequence[str],
    write_indexes: Sequence[int] = (),
) -> str:
    """Store `length` linked checkpoints. Returns the newest checkpoint id.

    Only the oldest checkpoint carries channel values, so a delta read has to
    walk the whole parent chain. Writes are stored on the requested ancestors.
    """
    parent: str | None = None
    newest = ""
    seed_values = {channel: f"seed-{channel}" for channel in channels}
    seed_versions = {channel: 1 for channel in channels}
    for index in range(length):
        values = seed_values if index == 0 else {}
        versions = seed_versions if index == 0 else {}
        saved = saver.put(
            _config(thread_id, parent),
            _checkpoint(index, values, versions=versions),
            {"source": "loop", "step": index},
            {},
        )
        newest = saved["configurable"]["checkpoint_id"]
        if index in write_indexes:
            saver.put_writes(
                saved,
                [(channels[0], f"w{index}")],
                task_id=f"task-{index:08d}",
            )
        parent = newest
    return newest


def _insert_off_path_blobs(
    conn: sqlite3.Connection, thread_id: str, count: int
) -> None:
    row = conn.execute(
        "SELECT type FROM checkpoints WHERE thread_id = ? LIMIT 1",
        (thread_id,),
    ).fetchone()
    assert row is not None
    blob = b"X" * OFF_PATH_BLOB_BYTES
    conn.executemany(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
        "parent_checkpoint_id, type, checkpoint, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                thread_id,
                "",
                _off_path_cid(index),
                "not-an-ancestor",
                row[0],
                blob,
                b"{}",
            )
            for index in range(count)
        ],
    )
    conn.commit()


def _compile_time_variable_limit(conn: sqlite3.Connection) -> int:
    """Highest `SQLITE_LIMIT_VARIABLE_NUMBER` this build will accept.

    `sqlite3_limit` cannot raise the limit above the compile-time maximum.
    """
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 2**31 - 1)
    return conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)


def _chain_length_over_variable_limit(conn: sqlite3.Connection, channels: int) -> int:
    """Ancestor-chain length whose stage-2 binds exceed the compile-time max.

    Each channel contributes `thread_id`, `checkpoint_ns`, `channel`, and one
    bind per ancestor checkpoint.
    """
    hard_max = _compile_time_variable_limit(conn)
    chain_length = 0
    while channels * (3 + chain_length) <= hard_max:
        chain_length += 1
    return chain_length


def _is_select_from_checkpoints(sql: str) -> bool:
    compact = " ".join(sql.lower().split())
    return compact.startswith("select") and "from checkpoints" in compact


def _is_range_scan(sql: str) -> bool:
    compact = " ".join(sql.lower().split())
    if not _is_select_from_checkpoints(compact):
        return False
    return "checkpoint_id <" in compact


class _ScanObserver:
    def __init__(self) -> None:
        self.recording = False
        self.blob_bytes = 0
        self.range_rows_seen = 0
        self.rows_this_hold = 0
        self.max_range_rows_per_hold = 0
        self.depth = 0
        self._probing = False
        self.interleaved_id: str | None = None
        self.saver: Any = None
        self.other_config: RunnableConfig | None = None
        self.expected_range_rows: int | None = None

    def on_acquire(self) -> None:
        if not self.recording or self._probing:
            return
        self.depth += 1
        if self.depth == 1:
            self.rows_this_hold = 0

    def on_release_start(self) -> None:
        if not self.recording or self._probing or self.depth == 0:
            return
        self.depth -= 1
        if self.depth == 0:
            self.max_range_rows_per_hold = max(
                self.max_range_rows_per_hold, self.rows_this_hold
            )

    def note_row(self, sql: str, row: Sequence[Any]) -> None:
        if not self.recording or self._probing:
            return
        if _is_select_from_checkpoints(sql):
            for value in row:
                if isinstance(value, (bytes, memoryview)):
                    self.blob_bytes += len(value)
        if self.depth > 0 and _is_range_scan(sql):
            self.rows_this_hold += 1
            self.range_rows_seen += 1

    def _remember_interleave(self, found: Any) -> None:
        if found is None:
            return
        self.interleaved_id = found.config["configurable"]["checkpoint_id"]

    def maybe_interleave_sync(self) -> None:
        if not self._should_interleave():
            return
        assert self.saver is not None
        assert self.other_config is not None
        self._probing = True
        try:
            self._remember_interleave(self.saver.get_tuple(self.other_config))
        finally:
            self._probing = False

    async def maybe_interleave_async(self) -> None:
        if not self._should_interleave():
            return
        assert self.saver is not None
        assert self.other_config is not None
        self._probing = True
        try:
            found = await self.saver.aget_tuple(self.other_config)
            self._remember_interleave(found)
        finally:
            self._probing = False

    def _should_interleave(self) -> bool:
        if (
            self._probing
            or not self.recording
            or self.saver is None
            or self.other_config is None
            or self.expected_range_rows is None
            or self.interleaved_id is not None
        ):
            return False
        return 0 < self.range_rows_seen < self.expected_range_rows


class _ObservingCursor:
    def __init__(self, cursor: sqlite3.Cursor, observer: _ScanObserver) -> None:
        self._cursor = cursor
        self._observer = observer
        self._sql = ""

    def execute(self, sql: str, parameters: Any = ()) -> _ObservingCursor:
        self._sql = sql
        self._cursor.execute(sql, parameters)
        return self

    def executemany(self, sql: str, seq_of_parameters: Any) -> _ObservingCursor:
        self._cursor.executemany(sql, seq_of_parameters)
        return self

    def fetchone(self) -> Any:
        row = self._cursor.fetchone()
        if row is not None:
            self._observer.note_row(self._sql, row)
        return row

    def fetchall(self) -> list[Any]:
        rows = self._cursor.fetchall()
        for row in rows:
            self._observer.note_row(self._sql, row)
        return rows

    def __iter__(self) -> Any:
        while True:
            row = self.fetchone()
            if row is None:
                break
            yield row

    def close(self) -> None:
        self._cursor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _ObservingConnection:
    def __init__(self, connection: sqlite3.Connection, observer: _ScanObserver) -> None:
        self.raw = connection
        self._observer = observer

    def cursor(self, *args: Any, **kwargs: Any) -> _ObservingCursor:
        return _ObservingCursor(self.raw.cursor(*args, **kwargs), self._observer)

    def commit(self) -> None:
        self.raw.commit()

    def executescript(self, script: str) -> sqlite3.Cursor:
        return self.raw.executescript(script)

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        return self.raw.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any) -> sqlite3.Cursor:
        return self.raw.executemany(sql, seq_of_parameters)

    def close(self) -> None:
        self.raw.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


class _ObservingLock:
    def __init__(self, inner: Any, observer: _ScanObserver) -> None:
        self._inner = inner
        self._observer = observer

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._inner.acquire(blocking, timeout)
        if acquired:
            self._observer.on_acquire()
        return acquired

    def release(self) -> None:
        self._observer.on_release_start()
        self._inner.release()
        self._observer.maybe_interleave_sync()

    def __enter__(self) -> _ObservingLock:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def locked(self) -> bool:
        return self._inner.locked()


class _ObservingAsyncLock:
    def __init__(self, inner: Any, observer: _ScanObserver) -> None:
        self._inner = inner
        self._observer = observer

    async def __aenter__(self) -> _ObservingAsyncLock:
        await self._inner.acquire()
        self._observer.on_acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._observer.on_release_start()
        self._inner.release()
        await self._observer.maybe_interleave_async()

    def locked(self) -> bool:
        return self._inner.locked()


class _ObservingAsyncCursor:
    def __init__(self, cursor: Any, observer: _ScanObserver) -> None:
        self._cursor = cursor
        self._observer = observer
        self._sql = ""

    async def execute(self, sql: str, parameters: Any = ()) -> _ObservingAsyncCursor:
        self._sql = sql
        await self._cursor.execute(sql, parameters)
        return self

    async def executemany(self, sql: str, parameters: Any) -> Any:
        return await self._cursor.executemany(sql, parameters)

    async def fetchall(self) -> list[Any]:
        rows = await self._cursor.fetchall()
        for row in rows:
            self._observer.note_row(self._sql, row)
        return rows

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        async for row in self._cursor:
            self._observer.note_row(self._sql, row)
            yield row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _AsyncCursorContext:
    def __init__(self, inner: Any, observer: _ScanObserver) -> None:
        self._inner = inner
        self._observer = observer

    async def __aenter__(self) -> _ObservingAsyncCursor:
        cursor = await self._inner.__aenter__()
        return _ObservingAsyncCursor(cursor, self._observer)

    async def __aexit__(self, *args: Any) -> Any:
        return await self._inner.__aexit__(*args)


def _install_async_observer(saver: AsyncSqliteSaver, observer: _ScanObserver) -> None:
    original_cursor = saver.conn.cursor

    def cursor() -> _AsyncCursorContext:
        return _AsyncCursorContext(original_cursor(), observer)

    saver.conn.cursor = cursor  # type: ignore[method-assign]
    saver.lock = _ObservingAsyncLock(saver.lock, observer)  # type: ignore[assignment]
    observer.saver = saver


def _assert_history(
    result: Mapping[str, Any],
    *,
    channels: Sequence[str],
    write_indexes: Sequence[int],
) -> None:
    for channel in channels:
        assert result[channel]["seed"] == f"seed-{channel}"
    assert [write[2] for write in result[channels[0]]["writes"]] == [
        f"w{index}" for index in write_indexes
    ]


def _assert_off_path_blobs_not_loaded(observer: _ScanObserver) -> None:
    assert observer.blob_bytes < OFF_PATH_BLOB_BYTES, (
        f"checkpoint scan copied {observer.blob_bytes} bytes, including "
        f"off-path blobs of {OFF_PATH_BLOB_BYTES} bytes"
    )


def _payload_bytes(conn: sqlite3.Connection, thread_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(length(checkpoint) + length(metadata)), 0) "
        "FROM checkpoints WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


async def _async_payload_bytes(saver: AsyncSqliteSaver, thread_id: str) -> int:
    async with saver.conn.execute(
        "SELECT COALESCE(SUM(length(checkpoint) + length(metadata)), 0) "
        "FROM checkpoints WHERE thread_id = ?",
        (thread_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


def _build_off_path_database(conn: sqlite3.Connection) -> tuple[SqliteSaver, str]:
    observer = _ScanObserver()
    saver = SqliteSaver(_ObservingConnection(conn, observer))  # type: ignore[arg-type]
    channels = ("items", "other")
    target = _put_chain(
        saver,
        thread_id="victim",
        length=3,
        channels=channels,
        write_indexes=(1,),
    )
    _insert_off_path_blobs(conn, "victim", 20)
    return saver, target


def test_sync_read_does_not_load_off_path_checkpoint_blobs() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver, target = _build_off_path_database(conn)
    observer = saver.conn._observer  # type: ignore[attr-defined]
    channels = ["items", "other"]
    observer.recording = True
    try:
        result = saver.get_delta_channel_history(
            config=_config("victim", target), channels=channels
        )
    finally:
        observer.recording = False
    _assert_history(result, channels=channels, write_indexes=(1,))
    _assert_off_path_blobs_not_loaded(observer)
    conn.close()


def test_stage1_releases_lock_between_pages_sync() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    observer = _ScanObserver()
    saver = SqliteSaver(_ObservingConnection(conn, observer))  # type: ignore[arg-type]
    length = MAX_CHECKPOINTS_PER_LOCK_HOLD + 256
    write_indexes = (0, length // 2)
    target = _put_chain(
        saver,
        thread_id="victim",
        length=length,
        channels=("items",),
        write_indexes=write_indexes,
    )
    bystander = _put_chain(
        saver,
        thread_id="bystander",
        length=1,
        channels=("items",),
    )
    saver.lock = _ObservingLock(saver.lock, observer)  # type: ignore[assignment]
    observer.saver = saver
    observer.other_config = _config("bystander")
    observer.expected_range_rows = length
    observer.recording = True
    try:
        result = saver.get_delta_channel_history(
            config=_config("victim", target), channels=["items"]
        )
    finally:
        observer.recording = False
    _assert_history(result, channels=("items",), write_indexes=write_indexes)
    assert observer.max_range_rows_per_hold <= MAX_CHECKPOINTS_PER_LOCK_HOLD, (
        f"one lock hold streamed {observer.max_range_rows_per_hold} checkpoint rows"
    )
    assert observer.interleaved_id == bystander, (
        "a read of another thread did not run until the full scan released "
        f"the saver lock (interleaved checkpoint: {observer.interleaved_id})"
    )
    conn.close()


def test_stage2_stays_under_sqlite_variable_limit_sync() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    channels = ("items", "other")
    chain_length = _chain_length_over_variable_limit(conn, len(channels))
    length = chain_length + 1
    write_indexes = tuple(range(0, chain_length, max(chain_length // 5, 1)))
    saver = SqliteSaver(conn)
    target = _put_chain(
        saver,
        thread_id="victim",
        length=length,
        channels=channels,
        write_indexes=write_indexes,
    )
    result = saver.get_delta_channel_history(
        config=_config("victim", target), channels=list(channels)
    )
    _assert_history(result, channels=channels, write_indexes=write_indexes)
    conn.close()


@pytest.mark.anyio
async def test_async_read_does_not_load_off_path_checkpoint_blobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "checkpoints.sqlite")
        conn = sqlite3.connect(path, check_same_thread=False)
        _saver, target = _build_off_path_database(conn)
        conn.close()
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            observer = _ScanObserver()
            _install_async_observer(saver, observer)
            channels = ["items", "other"]
            await saver.setup()
            observer.recording = True
            try:
                result = await saver.aget_delta_channel_history(
                    config=_config("victim", target), channels=channels
                )
            finally:
                observer.recording = False
            _assert_history(result, channels=channels, write_indexes=(1,))
            _assert_off_path_blobs_not_loaded(observer)

            observer.blob_bytes = 0
            observer.recording = True
            try:
                bridged = await asyncio.to_thread(
                    lambda: saver.get_delta_channel_history(
                        config=_config("victim", target), channels=channels
                    )
                )
            finally:
                observer.recording = False
            _assert_history(bridged, channels=channels, write_indexes=(1,))
            _assert_off_path_blobs_not_loaded(observer)


@pytest.mark.anyio
async def test_stage1_releases_lock_between_pages_async() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "checkpoints.sqlite")
        conn = sqlite3.connect(path, check_same_thread=False)
        observer = _ScanObserver()
        saver = SqliteSaver(_ObservingConnection(conn, observer))  # type: ignore[arg-type]
        length = MAX_CHECKPOINTS_PER_LOCK_HOLD + 256
        write_indexes = (0, length // 2)
        target = _put_chain(
            saver,
            thread_id="victim",
            length=length,
            channels=("items",),
            write_indexes=write_indexes,
        )
        bystander = _put_chain(
            saver, thread_id="bystander", length=1, channels=("items",)
        )
        conn.close()
        async with AsyncSqliteSaver.from_conn_string(path) as async_saver:
            async_observer = _ScanObserver()
            _install_async_observer(async_saver, async_observer)
            async_observer.other_config = _config("bystander")
            async_observer.expected_range_rows = length
            await async_saver.setup()
            async_observer.recording = True
            try:
                result = await async_saver.aget_delta_channel_history(
                    config=_config("victim", target), channels=["items"]
                )
            finally:
                async_observer.recording = False
            _assert_history(result, channels=("items",), write_indexes=write_indexes)
            assert (
                async_observer.max_range_rows_per_hold <= MAX_CHECKPOINTS_PER_LOCK_HOLD
            ), (
                f"one lock hold streamed {async_observer.max_range_rows_per_hold} "
                f"checkpoint rows"
            )
            assert async_observer.interleaved_id == bystander, (
                "a read of another thread did not run until the full scan "
                "released the saver lock "
                f"(interleaved checkpoint: {async_observer.interleaved_id})"
            )


@pytest.mark.anyio
async def test_stage2_stays_under_sqlite_variable_limit_async() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "checkpoints.sqlite")
        conn = sqlite3.connect(path, check_same_thread=False)
        channels = ("items", "other")
        chain_length = _chain_length_over_variable_limit(conn, len(channels))
        length = chain_length + 1
        write_indexes = tuple(range(0, chain_length, max(chain_length // 5, 1)))
        saver = SqliteSaver(conn)
        target = _put_chain(
            saver,
            thread_id="victim",
            length=length,
            channels=channels,
            write_indexes=write_indexes,
        )
        conn.close()
        async with AsyncSqliteSaver.from_conn_string(path) as async_saver:
            result = await async_saver.aget_delta_channel_history(
                config=_config("victim", target), channels=list(channels)
            )
            _assert_history(result, channels=channels, write_indexes=write_indexes)


def _graph_fixtures() -> tuple[Any, Any, RunnableConfig, str]:
    pytest.importorskip("langgraph.graph", reason="langgraph core not installed")
    from typing import Annotated  # noqa: PLC0415

    from langgraph.channels.delta import DeltaChannel  # noqa: PLC0415
    from langgraph.graph import StateGraph  # noqa: PLC0415
    from typing_extensions import TypedDict  # noqa: PLC0415

    def _extend(current: list[str], writes: list[list[str]]) -> list[str]:
        merged = list(current)
        for write in writes:
            merged.extend(write)
        return merged

    # `from __future__ import annotations` evaluates this class's hints in
    # the module globals. StateGraph reads those hints, so the names it
    # needs have to live there.
    globals()["Annotated"] = Annotated
    globals()["DeltaChannel"] = DeltaChannel
    globals()["_extend"] = _extend

    class _Conversation(TypedDict):
        items: Annotated[list[str], DeltaChannel(_extend, snapshot_frequency=10_000)]
        note: str

    return (
        _Conversation,
        StateGraph,
        {"configurable": {"thread_id": "conversation"}},
        "N" * 8000,
    )


def _compile_graph(saver: Any, state: Any, graph_cls: Any) -> Any:
    from langgraph.graph import END, START  # noqa: PLC0415

    return (
        graph_cls(state)
        .add_node("noop", lambda _state: {})
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile(checkpointer=saver)
    )


def _assert_graph_scan_skipped_off_path_blobs(
    observer: _ScanObserver, total_payload: int
) -> None:
    assert observer.blob_bytes > 0, "delta read did not load a checkpoint blob"
    assert observer.blob_bytes * 2 < total_payload, (
        f"replay copied {observer.blob_bytes} checkpoint bytes; "
        f"the thread stores {total_payload}, most of it off the ancestor chain"
    )


def test_graph_replay_does_not_load_off_path_checkpoint_blobs() -> None:
    state, graph_cls, config, note = _graph_fixtures()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    observer = _ScanObserver()
    saver = SqliteSaver(_ObservingConnection(conn, observer))  # type: ignore[arg-type]
    graph = _compile_graph(saver, state, graph_cls)
    graph.invoke({"items": ["root"], "note": note}, config, durability="sync")
    oldest = list(graph.get_state_history(config))[-1].config
    for index in range(FORKS):
        graph.update_state(
            oldest,
            {"items": [f"fork{index}"], "note": note},
            as_node="noop",
        )

    calls = {"n": 0}
    original = saver.get_delta_channel_history

    def _counting(**kwargs: Any) -> Any:
        calls["n"] += 1
        return original(**kwargs)

    saver.get_delta_channel_history = _counting  # type: ignore[method-assign]

    observer.recording = True
    try:
        replayed = graph.get_state(config)
    finally:
        observer.recording = False
    assert calls["n"] >= 1
    assert replayed.values["note"] == note
    # Parent writes are replayed through get_delta_channel_history. Later
    # forks of that same parent stay in the scanned id range.
    assert set(replayed.values["items"]) == {"fork0", "root"}
    _assert_graph_scan_skipped_off_path_blobs(
        observer, _payload_bytes(conn, "conversation")
    )

    observer.blob_bytes = 0
    calls["n"] = 0
    observer.recording = True
    try:
        graph.invoke({"items": ["tail"], "note": note}, config, durability="sync")
    finally:
        observer.recording = False
    assert calls["n"] >= 1
    _assert_graph_scan_skipped_off_path_blobs(
        observer, _payload_bytes(conn, "conversation")
    )
    resumed = graph.get_state(config)
    assert set(resumed.values["items"]) == {"fork0", "root", "tail"}
    conn.close()


@pytest.mark.anyio
async def test_async_graph_replay_does_not_load_off_path_checkpoint_blobs() -> None:
    pytest.importorskip("langgraph.graph", reason="langgraph core not installed")
    state, graph_cls, config, note = _graph_fixtures()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        observer = _ScanObserver()
        _install_async_observer(saver, observer)
        graph = _compile_graph(saver, state, graph_cls)
        await graph.ainvoke(
            {"items": ["root"], "note": note}, config, durability="sync"
        )
        history = [item async for item in graph.aget_state_history(config)]
        oldest = history[-1].config
        for index in range(FORKS):
            await graph.aupdate_state(
                oldest,
                {"items": [f"fork{index}"], "note": note},
                as_node="noop",
            )

        calls = {"n": 0}
        original = saver.aget_delta_channel_history

        async def _counting(**kwargs: Any) -> Any:
            calls["n"] += 1
            return await original(**kwargs)

        saver.aget_delta_channel_history = _counting  # type: ignore[method-assign]

        observer.recording = True
        try:
            replayed = await graph.aget_state(config)
        finally:
            observer.recording = False
        assert calls["n"] >= 1
        assert replayed.values["note"] == note
        assert set(replayed.values["items"]) == {"fork0", "root"}
        _assert_graph_scan_skipped_off_path_blobs(
            observer, await _async_payload_bytes(saver, "conversation")
        )

        observer.blob_bytes = 0
        calls["n"] = 0
        observer.recording = True
        try:
            await graph.ainvoke(
                {"items": ["tail"], "note": note}, config, durability="sync"
            )
        finally:
            observer.recording = False
        assert calls["n"] >= 1
        _assert_graph_scan_skipped_off_path_blobs(
            observer, await _async_payload_bytes(saver, "conversation")
        )
        resumed = await graph.aget_state(config)
        assert set(resumed.values["items"]) == {"fork0", "root", "tail"}
