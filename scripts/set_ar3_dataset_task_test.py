import numpy as np
import pyarrow.parquet as pq
import pytest

from openpi.data import lerobot_v3
from scripts import ar3_lift_recorder as recorder_lib
from scripts import set_ar3_dataset_task as tool

OLD, NEW = "plug the charger into the socket", "pick up a bottle and shake it"


def _record(root, task: str, frames: int = 3) -> None:
    recorder = recorder_lib.EpisodeRecorder(root, task=task, fps=10, image_size=(48, 64))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.start_episode()
    for step in range(frames):
        values = {key: np.full(recorder.features[key]["shape"], float(step)) for key in recorder.frame_keys}
        recorder.add_frame(values, {"front": image, "wrist": image})
    recorder.save_episode()


def _episode_tasks(root) -> list[list[str]]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return [tasks for path in files for tasks in pq.read_table(path, columns=["tasks"]).column("tasks").to_pylist()]


def test_the_task_text_is_replaced_in_the_metadata_and_nowhere_else(tmp_path, capsys):
    root = tmp_path / "bottles"
    _record(root, OLD)
    _record(root, OLD)
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_path.chmod(0o664)
    untouched = [*sorted((root / "data").rglob("*.parquet")), *sorted((root / "videos").rglob("*.mp4"))]
    before = [path.read_bytes() for path in untouched]

    tool.main(["--dataset", str(root), "--task", NEW])

    assert f"{OLD!r} -> {NEW!r} (meta/tasks.parquet and 2 episode files)" in capsys.readouterr().out
    dataset = lerobot_v3.LeRobotV3Dataset(root, options=lerobot_v3.LeRobotV3Options(decode_videos=False))
    assert {dataset[index]["task"] for index in range(len(dataset))} == {NEW}
    assert _episode_tasks(root) == [[NEW], [NEW]]
    # Frames, videos and task indices are as recorded; the file keeps its permissions and no temporary file is left.
    assert [path.read_bytes() for path in untouched] == before
    assert tasks_path.stat().st_mode & 0o777 == 0o664
    assert not list(root.rglob("*.tmp"))

    # A second run changes nothing, and a later recording session with the new text stays the same task.
    assert tool.set_task(root, NEW) == (NEW, 0)
    _record(root, NEW)
    assert lerobot_v3.LeRobotV3Metadata(root).tasks == {0: NEW}


def test_datasets_with_several_tasks_name_the_text_to_replace(tmp_path):
    root = tmp_path / "mixed"
    _record(root, OLD)
    _record(root, "unplug the charger")
    with pytest.raises(SystemExit, match="has 2 tasks; name the one to replace with --old"):
        tool.main(["--dataset", str(root), "--task", NEW])
    with pytest.raises(SystemExit, match="is already another task"):
        tool.main(["--dataset", str(root), "--task", "unplug the charger", "--old", OLD])
    with pytest.raises(SystemExit, match="has no task 'open the drawer'"):
        tool.main(["--dataset", str(root), "--task", NEW, "--old", "open the drawer"])
    with pytest.raises(SystemExit, match="not a LeRobot v3.0 dataset"):
        tool.main(["--dataset", str(tmp_path / "missing"), "--task", NEW])

    tool.main(["--dataset", str(root), "--task", NEW, "--old", OLD])
    assert lerobot_v3.LeRobotV3Metadata(root).tasks == {0: NEW, 1: "unplug the charger"}
    assert _episode_tasks(root) == [[NEW], ["unplug the charger"]]
