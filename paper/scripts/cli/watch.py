"""Live progress dashboard for experiments with interactive keyboard controls.

Uses Rich Live display with a daemon input thread for real-time filtering.
Keyboard controls: [t]ask, [c]ategory, [s]tage, [q]uit.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from rich import box
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from paper.scripts.adapters.store import S3Store
from paper.scripts.cli._console import console

# ---------------------------------------------------------------------------
# Cycle definitions
# ---------------------------------------------------------------------------

TASK_CYCLE: list[str | None] = [None, "classification", "regression"]
CATEGORY_CYCLE: list[str | None] = [None, "filter", "ptest", "embedding", "wrapper"]
STAGE_CYCLE: list[str | None] = [None, "rankings", "metrics"]

_TASK_LABELS: dict[str | None, str] = {
    None: "both",
    "classification": "clf",
    "regression": "reg",
}
_CATEGORY_LABELS: dict[str | None, str] = {
    None: "all",
    "filter": "filter",
    "ptest": "ptest",
    "embedding": "embedding",
    "wrapper": "wrapper",
}
_STAGE_LABELS: dict[str | None, str] = {
    None: "both",
    "rankings": "rankings",
    "metrics": "metrics",
}

# Max rows the table can ever have (2 tasks * 2 stages)
_MAX_TABLE_ROWS = 4


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class DashboardState:
    """Mutable shared state between input thread and render loop."""

    task_idx: int = 0
    category_idx: int = 0
    stage_idx: int = 0
    dirty: bool = True
    quit: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Current filter values (derived from indices)
    @property
    def task(self) -> str | None:
        return TASK_CYCLE[self.task_idx]

    @property
    def category(self) -> str | None:
        return CATEGORY_CYCLE[self.category_idx]

    @property
    def stage(self) -> str | None:
        return STAGE_CYCLE[self.stage_idx]

    def cycle_task(self) -> None:
        with self._lock:
            self.task_idx = (self.task_idx + 1) % len(TASK_CYCLE)
            self.dirty = True

    def cycle_category(self) -> None:
        with self._lock:
            self.category_idx = (self.category_idx + 1) % len(CATEGORY_CYCLE)
            self.dirty = True

    def cycle_stage(self) -> None:
        with self._lock:
            self.stage_idx = (self.stage_idx + 1) % len(STAGE_CYCLE)
            self.dirty = True


# ---------------------------------------------------------------------------
# Background S3 fetcher
# ---------------------------------------------------------------------------


class _BackgroundFetcher:
    """Fetch S3 progress on a background thread so the UI never blocks."""

    def __init__(self, store: S3Store, state: DashboardState, interval: float = 10.0) -> None:
        self._store = store
        self._state = state
        self._interval = interval
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], dict[str, set[tuple[str, int]]]] = {}
        self._thread: threading.Thread | None = None

    @property
    def progress_data(self) -> dict[tuple[str, str], dict[str, set[tuple[str, int]]]]:
        with self._lock:
            return dict(self._data)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._state.quit = True

    def _run(self) -> None:
        while not self._state.quit:
            snapshot: dict[tuple[str, str], dict[str, set[tuple[str, int]]]] = {}
            for task in ("classification", "regression"):
                for stage in ("rankings", "metrics"):
                    snapshot[(task, stage)] = _fetch_progress(self._store, stage, task)
            with self._lock:
                self._data = snapshot
            with self._state._lock:
                self._state.dirty = True
            deadline = time.monotonic() + self._interval
            while time.monotonic() < deadline and not self._state.quit:
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Input thread
# ---------------------------------------------------------------------------

_KEY_HANDLERS: dict[str, str] = {
    "t": "cycle_task",
    "c": "cycle_category",
    "s": "cycle_stage",
    "q": "quit",
}


def _input_thread(state: DashboardState) -> None:
    """Read single keypresses in cbreak mode (daemon thread)."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not state.quit:
            rlist, _, _ = select.select([fd], [], [], 0.1)
            if not rlist:
                continue
            ch = sys.stdin.read(1)
            if ch in _KEY_HANDLERS:
                action = _KEY_HANDLERS[ch]
                if action == "quit":
                    state.quit = True
                else:
                    getattr(state, action)()
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _base_name(label: str) -> str:
    """Extract base method name from a label like 'cit__816e78cf71d84843'."""
    return label.split("__", 1)[0]


def _resolve_context(
    task: str,
    category: str | None,
    method_configs_by_task: dict[str, list[str]],
    datasets_by_task: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Return (method_labels, dataset_list) for a task, filtered by category."""
    from paper.scripts.pipeline.methods import METHOD_INFO

    method_labels = method_configs_by_task[task]
    datasets = datasets_by_task[task]

    if category is not None:
        method_labels = [
            m
            for m in method_labels
            if (info := METHOD_INFO.get(_base_name(m))) is not None and info.category == category
        ]

    return method_labels, datasets


def _fetch_progress(
    store: S3Store,
    stage: str,
    task: str,
) -> dict[str, set[tuple[str, int]]]:
    """Fetch progress from S3 via S3Store.list_completed, organized by dataset."""
    completed_by_dataset: dict[str, set[tuple[str, int]]] = defaultdict(set)
    try:
        completed = store.list_completed(stage, task)
        for method_label, dataset, seed in completed:
            completed_by_dataset[dataset].add((method_label, seed))
    except Exception:
        pass
    return completed_by_dataset


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_TASK_SHORT = {"classification": "CLF", "regression": "REG"}


def _build_output(
    state: DashboardState,
    bucket: str,
    progress_data: dict[tuple[str, str], dict[str, set[tuple[str, int]]]],
    method_configs_by_task: dict[str, list[str]],
    datasets_by_task: dict[str, list[str]],
    n_seeds: int,
) -> Group:
    """Build the display with a fixed-height layout to prevent ghosting."""
    tasks = ["classification", "regression"] if state.task is None else [state.task]
    stages = ["rankings", "metrics"] if state.stage is None else [state.stage]

    header = Text()
    header.append("citrees-exp", style="bold cyan")
    header.append(f"  {bucket}", style="dim")
    header.append(f"  {datetime.now().strftime('%H:%M:%S')}", style="dim italic")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold", expand=False)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Done", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("%", justify="right")

    row_count = 0
    for task in tasks:
        method_labels, datasets = _resolve_context(
            task, state.category, method_configs_by_task, datasets_by_task
        )
        method_set = set(method_labels)
        total_expected = len(datasets) * len(method_labels) * n_seeds
        for stage in stages:
            completed = progress_data.get((task, stage), {})
            done = sum(
                1 for pairs in completed.values() for method, _ in pairs if method in method_set
            )
            pct = 100 * done / total_expected if total_expected else 0
            style = "green" if pct >= 100 else "yellow" if pct >= 50 else ""
            table.add_row(
                f"{_TASK_SHORT[task]} {stage.title()}",
                f"{done:,}",
                f"{total_expected:,}",
                Text(f"{pct:.0f}%", style=style),
            )
            row_count += 1

    # Pad to fixed height so Rich Live doesn't ghost when rows shrink
    for _ in range(row_count, _MAX_TABLE_ROWS):
        table.add_row("", "", "", "")

    footer = Text()
    footer.append("[t]", style="bold")
    footer.append(f"ask={_TASK_LABELS[state.task]}  ", style="dim")
    footer.append("[c]", style="bold")
    footer.append(f"at={_CATEGORY_LABELS[state.category]}  ", style="dim")
    footer.append("[s]", style="bold")
    footer.append(f"tage={_STAGE_LABELS[state.stage]}  ", style="dim")
    footer.append("[q]", style="bold")
    footer.append("uit", style="dim")

    return Group(header, Text(), table, Text(), footer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_watch() -> None:
    """Run the interactive live progress dashboard.

    Shows both tasks by default. Use keyboard controls to filter:
    [t] cycle task, [c] cycle category, [s] cycle stage, [q] quit.
    """
    from paper.scripts.adapters import S3Store, get_datasets
    from paper.scripts.config import load_config
    from paper.scripts.pipeline import get_full_method_configs, get_methods

    config = load_config()
    n_seeds = config.experiment.n_seeds
    store = S3Store.from_config()
    bucket = store.bucket

    # Pre-compute method configs and datasets for both tasks
    method_configs_by_task: dict[str, list[str]] = {}
    datasets_by_task: dict[str, list[str]] = {}

    for task in ("classification", "regression"):
        method_list = get_methods(task)
        method_configs = get_full_method_configs(method_list, task)
        method_configs_by_task[task] = [cfg.label for cfg in method_configs]
        datasets_by_task[task] = get_datasets(task)

    state = DashboardState()

    # Start background S3 fetcher
    fetcher = _BackgroundFetcher(store, state)
    fetcher.start()

    # Start input thread
    input_t = threading.Thread(target=_input_thread, args=(state,), daemon=True)
    input_t.start()

    loading = _build_output(state, bucket, {}, method_configs_by_task, datasets_by_task, n_seeds)

    try:
        with Live(loading, console=console, refresh_per_second=4) as live:
            while not state.quit:
                if state.dirty:
                    progress_data = fetcher.progress_data
                    output = _build_output(
                        state, bucket, progress_data, method_configs_by_task, datasets_by_task, n_seeds
                    )
                    live.update(output)
                    with state._lock:
                        state.dirty = False

                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        fetcher.stop()
        console.print("[dim]Dashboard stopped.[/]")
