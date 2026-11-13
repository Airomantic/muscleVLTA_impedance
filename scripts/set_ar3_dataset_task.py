"""Replaces the task text of a LeRobot v3.0 dataset, i.e. the prompt its frames are trained with.

Training takes the prompt of every frame from the dataset (``meta/tasks.parquet``), so a recording made with the task
text of another task trains the policy under the wrong instruction. This rewrites the text in ``meta/tasks.parquet``
and in the ``tasks`` lists of ``meta/episodes/**/*.parquet``; frames, videos and task indices stay as they are:

    .venv/bin/python scripts/set_ar3_dataset_task.py --dataset /data/jiang/take_medicine_bottle \
        --task "Pick up a bottle and shake it, ..."

A dataset with several tasks needs --old, the text to replace. Send the same text as the prompt on the robot.
"""

import argparse
from collections.abc import Sequence
import contextlib
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

# LeRobot stores the task strings as the dataframe index, named "task" (unnamed in older files).
TASK_COLUMNS = ("task", "__index_level_0__")


def _replace_file(path: pathlib.Path, table: pa.Table) -> None:
    """Writes ``table`` in place of ``path`` with its owner and permissions: recordings often belong to the account
    that recorded them."""
    status = path.stat()
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary)
    temporary.chmod(status.st_mode & 0o7777)
    with contextlib.suppress(PermissionError):
        os.chown(temporary, status.st_uid, status.st_gid)
    temporary.replace(path)


def _with_column(table: pa.Table, name: str, values: list) -> pa.Table:
    index = table.column_names.index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type)).replace_schema_metadata(
        table.schema.metadata
    )


def set_task(root: str | os.PathLike[str], task: str, old: str | None = None) -> tuple[str, int]:
    """Replaces the task text ``old`` (the only task of the dataset if None) by ``task``.

    Returns the replaced text and the number of episode metadata files that were rewritten.
    """
    root = pathlib.Path(root)
    if not task.strip():
        raise ValueError("The task text is empty")
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise ValueError(f"{root} is not a LeRobot v3.0 dataset (missing meta/tasks.parquet)")
    table = pq.read_table(tasks_path)
    column = next((name for name in TASK_COLUMNS if name in table.column_names), None)
    if column is None:
        raise ValueError(f"Unexpected columns in {tasks_path}: {table.column_names}")
    texts = [str(text) for text in table.column(column).to_pylist()]
    if old is None:
        if len(texts) != 1:
            raise ValueError(f"{root} has {len(texts)} tasks; name the one to replace with --old: {texts}")
        old = texts[0]
    if old not in texts:
        raise ValueError(f"{root} has no task {old!r}; its tasks: {texts}")
    if task != old and task in texts:
        raise ValueError(f"{task!r} is already another task of {root}")

    # The episode metadata first: if this is interrupted, tasks.parquet still names the old text and a second run
    # finishes the job.
    rewritten = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        episodes = pq.read_table(path)
        if "tasks" not in episodes.column_names:
            continue
        lists = episodes.column("tasks").to_pylist()
        updated = [None if items is None else [task if text == old else text for text in items] for items in lists]
        if updated != lists:
            _replace_file(path, _with_column(episodes, "tasks", updated))
            rewritten += 1
    if task != old:
        _replace_file(tasks_path, _with_column(table, column, [task if text == old else text for text in texts]))
    return old, rewritten


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="LeRobot v3.0 dataset directory")
    parser.add_argument("--task", required=True, help="The new task text (the prompt)")
    parser.add_argument("--old", help="The task text to replace (default: the only task of the dataset)")
    args = parser.parse_args(argv)
    try:
        old, rewritten = set_task(args.dataset, args.task, args.old)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if old == args.task and not rewritten:
        print(f"{args.dataset}: the task already is {args.task!r}")
    else:
        print(f"{args.dataset}: task {old!r} -> {args.task!r} (meta/tasks.parquet and {rewritten} episode files)")


if __name__ == "__main__":
    main()
