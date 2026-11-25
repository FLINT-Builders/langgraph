"""Shared helpers for `get_delta_channel_history` on sqlite savers.

Mirrors the two-stage shape of `BasePostgresSaver` (ancestor walk +
per-channel UNION ALL writes fetch), but adapted for sqlite's
constraints. The structural differences:

* No JSONB — to inspect `channel_values` for a checkpoint we must
  deserialize the full blob. Stage 1 therefore scans only
  `(checkpoint_id, parent_checkpoint_id)` and loads the blob by primary key
  for the checkpoints the walk actually visits. Sibling branches that share
  the scanned id range never have their blobs read.
* No separate blob table — `channel_values` lives inline in the
  checkpoint, so seeds come back from stage 1 with no second fetch.
* Single merged walk (not K independent walks): each visited cid is
  deserialized exactly once, regardless of how many channels are still
  seeking their seed.
* No array binds — stage 2 inlines one placeholder per chain cid, so the
  statements are split to stay under `SQLITE_MAX_VARIABLE_NUMBER`.

Both stages run as a sequence of short steps; callers take the saver lock
per step so other threads sharing the saver are not starved by one long
history read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langgraph.checkpoint.base import DeltaChannelHistory, PendingWrite

# Upper bound on stage-1 rows streamed per saver-lock hold. Same bound as
# `langgraph.checkpoint.postgres.base._DELTA_PAGE_SIZE`.
DELTA_PAGE_SIZE = 1024

# SQLite builds before 3.32.0 default `SQLITE_MAX_VARIABLE_NUMBER` to 999,
# and the compile-time maximum cannot be raised at runtime.
DELTA_MAX_VARIABLES = 999

# Stage 1 page: ids only, newest-first, starting at the cid the walk is
# looking for. The `<=` bound lets each page seek straight to the next
# ancestor, skipping any sibling-branch rows between it and the previous one.
DELTA_STAGE1_SQL = (
    "SELECT checkpoint_id, parent_checkpoint_id "
    "FROM checkpoints "
    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <= ? "
    "ORDER BY checkpoint_id DESC LIMIT ?"
)

DELTA_CHECKPOINT_BLOB_SQL = (
    "SELECT type, checkpoint "
    "FROM checkpoints "
    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
)

_STAGE2_BRANCH_FIXED_PARAMS = 3  # thread_id, checkpoint_ns, channel


class DeltaAncestorWalk:
    """Resumable merged ancestor walk for stage 1.

    Callers drive it one saver-lock hold at a time. Each hold runs
    `DELTA_STAGE1_SQL` with `stage1_params(limit)` (repeatedly, until the
    hold's row budget is spent or `done`), passing every row in order to
    `accept_row`. Whenever `pending_cid` is set after a row, the caller
    loads that checkpoint with `DELTA_CHECKPOINT_BLOB_SQL` and hands the
    row to `visit` before reading on. When the rows run out without
    `accept_row` asking to stop, the caller reports it via `end_page`.

    Each page starts at exactly the cid the walk needs, so a row is either
    that cid, an older one (the cid is not stored and the chain ends), or,
    after a match, a newer one from a sibling branch (the page stops and
    the next page seeks past the branch by index). Only ids are scanned;
    blobs are loaded solely for ancestors.

    Target itself is located but never visited — its own writes/seed are
    not part of the `BaseCheckpointSaver` contract.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        target_id: str,
        channels: Sequence[str],
        serde: Any,
    ) -> None:
        self.thread_id = thread_id
        self.checkpoint_ns = checkpoint_ns
        self.channels = list(channels)
        self.serde = serde
        self.chain_by_ch: dict[str, list[str]] = {ch: [] for ch in self.channels}
        self.seed_val_by_ch: dict[str, Any] = {}
        self.seeded: set[str] = set()
        self.pending_cid: str | None = None
        self._active: set[str] = set(self.channels)
        self._next_cid: str | None = target_id
        self._found_target = False

    @property
    def done(self) -> bool:
        return self._next_cid is None and self.pending_cid is None

    def stage1_params(self, limit: int) -> tuple[str, str, str, int]:
        assert self._next_cid is not None
        return (self.thread_id, self.checkpoint_ns, self._next_cid, limit)

    def blob_params(self) -> tuple[str, str, str]:
        assert self.pending_cid is not None
        return (self.thread_id, self.checkpoint_ns, self.pending_cid)

    def accept_row(self, cid: str, parent_cid: str | None) -> bool:
        """Consume one stage-1 row. Returns False when the page should stop."""
        next_cid = self._next_cid
        if next_cid is None:
            return False
        if cid != next_cid:
            if cid < next_cid:
                self._next_cid = None
            return False
        if self._found_target:
            self.pending_cid = cid
        else:
            self._found_target = True
        if parent_cid is not None and parent_cid >= cid:
            # A newest-first walk can never reach such a parent. Ending here
            # also keeps page bounds strictly decreasing, so the walk
            # terminates even if stored parent links form a cycle.
            parent_cid = None
        self._next_cid = parent_cid
        return parent_cid is not None

    def end_page(self) -> None:
        """The page returned fewer rows than requested: nothing older exists."""
        self._next_cid = None

    def visit(self, row: Sequence[Any] | None) -> None:
        """Process the blob row loaded for `pending_cid`."""
        cid = self.pending_cid
        assert cid is not None
        self.pending_cid = None
        if row is None:
            # Deleted after stage 1 located it; the chain cannot continue.
            self._next_cid = None
            return
        type_tag, blob = row
        for ch in self._active:
            self.chain_by_ch[ch].append(cid)
        ckpt = self.serde.loads_typed((type_tag, blob))
        channel_values: Mapping[str, Any] = ckpt.get("channel_values") or {}
        for ch in [ch for ch in self._active if ch in channel_values]:
            self.seed_val_by_ch[ch] = channel_values[ch]
            self.seeded.add(ch)
            self._active.discard(ch)
        del ckpt, channel_values
        if not self._active:
            self._next_cid = None


def iter_delta_stage2_queries(
    *,
    thread_id: str,
    checkpoint_ns: str,
    channels: Sequence[str],
    chain_by_ch: Mapping[str, Sequence[str]],
    max_variables: int = DELTA_MAX_VARIABLES,
) -> Iterator[tuple[str, list[Any]]]:
    """Yield stage-2 `(sql, params)` statements reading `writes` per channel.

    Each statement is a UNION ALL with one branch per `(channel, cid chunk)`;
    a branch inlines its own `IN (?, ?, ...)` list because sqlite has no
    array-bind equivalent of postgres's `= ANY(%s)`. Branches are packed so a
    statement never binds more than `max_variables` parameters, and a long
    chain is split across branches/statements. Per-channel branches avoid
    the over-fetch of a single `channel IN (...)` filter when channels have
    different chain depths — same rationale as postgres.
    """
    branch_sizes: list[int] = []
    params: list[Any] = []
    for ch in channels:
        cids = chain_by_ch.get(ch) or ()
        start = 0
        while start < len(cids):
            room = max_variables - len(params) - _STAGE2_BRANCH_FIXED_PARAMS
            if room <= 0:
                yield _build_delta_stage2_sql(branch_sizes), params
                branch_sizes, params = [], []
                continue
            chunk = cids[start : start + room]
            branch_sizes.append(len(chunk))
            params.extend([thread_id, checkpoint_ns, ch, *chunk])
            start += len(chunk)
    if branch_sizes:
        yield _build_delta_stage2_sql(branch_sizes), params


def _build_delta_stage2_sql(branch_sizes: Sequence[int]) -> str:
    return " UNION ALL ".join(
        "SELECT checkpoint_id, channel, task_id, idx, type, value "
        "FROM writes "
        "WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? "
        f"AND checkpoint_id IN ({','.join('?' * n)})"
        for n in branch_sizes
    )


def build_delta_channels_writes_history(
    *,
    channels: Sequence[str],
    chain_by_ch: Mapping[str, list[str]],
    seed_val_by_ch: Mapping[str, Any],
    seeded: set[str],
    stage2_rows: Sequence[tuple[str, str, str, int, str, bytes]],
    serde: Any,
) -> dict[str, DeltaChannelHistory]:
    """Demux stage-2 rows per channel; produce per-channel histories.

    Stage-2 rows are `(checkpoint_id, channel, task_id, idx, type, value)`.
    Final write order is oldest→newest globally and `(task_id, idx)` within
    a checkpoint, matching the contract on `DeltaChannelHistory.writes`.

    `seed` is omitted when the walk reached a true root with no snapshot
    found (channel never entered `seeded`); consumers treat absence as
    "start empty".
    """
    writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes, str, int]]]] = {
        ch: {} for ch in channels
    }
    for cid, ch, task_id, idx, type_tag, value_blob in stage2_rows:
        writes_by_ch_by_cid.setdefault(ch, {}).setdefault(cid, []).append(
            (type_tag, value_blob, task_id, idx)
        )
    for cid_map in writes_by_ch_by_cid.values():
        for ws in cid_map.values():
            ws.sort(key=lambda w: (w[2], w[3]))

    result: dict[str, DeltaChannelHistory] = {}
    for ch in channels:
        chain_cids = chain_by_ch.get(ch, [])
        cid_writes = writes_by_ch_by_cid.get(ch, {})
        collected: list[PendingWrite] = []
        # Chain is newest-first; iterate oldest-first for the public order.
        for cid in reversed(chain_cids):
            for type_tag, value_blob, task_id, _idx in cid_writes.get(cid, []):
                collected.append(
                    (task_id, ch, serde.loads_typed((type_tag, value_blob)))
                )
        entry: DeltaChannelHistory = {"writes": collected}
        if ch in seeded:
            entry["seed"] = seed_val_by_ch[ch]
        result[ch] = entry
    return result
