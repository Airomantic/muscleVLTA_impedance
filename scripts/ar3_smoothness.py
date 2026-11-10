"""Smoothness of AR3 motion in LeRobot v3.0 recordings: demonstrations and robot rollouts recorded with
``scripts/ar3_lift_robot_client.py --record-dir``.

For every episode it takes the measured TCP path (``observation.tcp_pose``) and the commanded targets (``actions``) of
the selected frames, trims the rest before and after the movement, and reports

* SPARC, the spectral arc length of the speed profile (dimensionless; closer to 0 is smoother),
* LDLJ, the log dimensionless jerk (dimensionless; larger, i.e. less negative, is smoother),
* the RMS translational [m/s^3] and angular [rad/s^3] jerk,
* the chunk-boundary ratio: the mean second difference of the commanded position at action-chunk boundaries divided
  by that inside chunks (1 = a new chunk continues as smoothly as the chunk itself),
* gripper command reversals per second.

SPARC and LDLJ do not depend on the amplitude or the duration of the movement, so runs of different length compare;
RMS jerk does. Report task success next to them: a policy that barely moves is smooth. Compare methods on rollouts
of the same task, start poses and controller:

    .venv/bin/python scripts/ar3_smoothness.py --out picture/results_smoothness \
        --run "muscleVLTA=/path/to/rollouts_muscle_vlta" --run "LIFT=/path/to/rollouts_lift" \
        --run "Demonstrations=/data/jiang/ar3_eef_forceVisuoTactaile_plugIn"

--control selects the frames: policy (control_flag +1), human (-1), all, or auto (policy frames if the recording has
any, else all frames, as for demonstrations).
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from scripts import build_figures
    from scripts import svg_chart
else:
    import build_figures
    import svg_chart

POLICY_FLAG, HUMAN_FLAG = 1, -1
MIN_SAMPLES = 12
METRICS = {
    "sparc": ("SPARC", "{:.2f}"),
    "ldlj": ("LDLJ", "{:.1f}"),
    "rms_jerk": ("RMS jerk [m/s³]", "{:.2f}"),
    "boundary_ratio": ("Chunk-boundary ratio", "{:.2f}"),
}


def speed_profile(positions: np.ndarray, fps: float) -> np.ndarray:
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps


def jerk_profile(positions: np.ndarray, fps: float) -> np.ndarray:
    return np.linalg.norm(np.diff(positions, n=3, axis=0), axis=1) * fps**3


def sparc(speed: np.ndarray, fps: float, *, pad_level: int = 4, cutoff: float = 10.0, threshold: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al., 2015), up to min(cutoff, Nyquist)."""
    size = int(2 ** (np.ceil(np.log2(len(speed))) + pad_level))
    frequencies = np.arange(size) * fps / size
    spectrum = np.abs(np.fft.fft(speed, size))
    spectrum = spectrum / max(spectrum.max(), 1e-12)
    band = frequencies <= min(cutoff, fps / 2)
    frequencies, spectrum = frequencies[band], spectrum[band]
    above = np.nonzero(spectrum >= threshold)[0]
    frequencies, spectrum = frequencies[above[0] : above[-1] + 1], spectrum[above[0] : above[-1] + 1]
    if len(frequencies) < 2:
        return 0.0
    steps = np.diff(frequencies) / (frequencies[-1] - frequencies[0])
    return float(-np.sum(np.sqrt(steps**2 + np.diff(spectrum) ** 2)))


def log_dimensionless_jerk(positions: np.ndarray, fps: float) -> float:
    """-ln(T^3 / v_peak^2 * integral |jerk|^2 dt) of a path (Balasubramanian et al., 2015)."""
    velocity = np.diff(positions, axis=0) * fps
    jerk = np.diff(velocity, n=2, axis=0) * fps**2
    duration = len(velocity) / fps
    peak = np.linalg.norm(velocity, axis=1).max()
    integral = np.sum(jerk**2) / fps
    return float(-np.log(max(duration**3 / max(peak, 1e-12) ** 2 * integral, 1e-300)))


def rms_jerk(positions: np.ndarray, fps: float) -> float:
    return float(np.sqrt(np.mean(jerk_profile(positions, fps) ** 2)))


def rms_angular_jerk(rotation_vectors: np.ndarray, fps: float) -> float:
    rotations = Rotation.from_rotvec(rotation_vectors)
    velocity = (rotations[:-1].inv() * rotations[1:]).as_rotvec() * fps
    return float(np.sqrt(np.mean(np.sum(np.diff(velocity, n=2, axis=0) ** 2, axis=1))) * fps**2)


def boundary_ratio(targets: np.ndarray, horizon: int) -> float:
    """Mean second difference of the commanded positions that spans a chunk boundary, relative to the others. Frame 0
    starts a chunk; the step from frame k * horizon - 1 to k * horizon crosses a boundary."""
    second = np.linalg.norm(targets[2:] - 2 * targets[1:-1] + targets[:-2], axis=1)
    centre = np.arange(1, len(targets) - 1)
    crossing = np.isin(centre % horizon, (0, horizon - 1))
    if not crossing.any() or crossing.all():
        return float("nan")
    return float(second[crossing].mean() / max(second[~crossing].mean(), 1e-12))


def gripper_reversals(gripper: np.ndarray, fps: float, *, deadband: float = 1.0) -> float:
    """Direction changes of the gripper command per second (steps within the deadband are ignored)."""
    steps = np.diff(gripper)
    signs = np.sign(steps[np.abs(steps) > deadband])
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (len(gripper) / fps))


def episode_metrics(pose: np.ndarray, actions: np.ndarray, fps: float, horizon: int) -> dict[str, float] | None:
    """Metrics of one contiguous segment; None if the movement is too short. The rest before and after the movement
    (speed below 5% of the peak) is trimmed for the path metrics; the command metrics use the whole segment."""
    speed = speed_profile(pose[:, :3], fps)
    if len(speed) < MIN_SAMPLES or speed.max() <= 0:
        return None
    moving = np.nonzero(speed >= 0.05 * speed.max())[0]
    start, end = moving[0], moving[-1] + 2
    if end - start < MIN_SAMPLES:
        return None
    path = pose[start:end]
    return {
        "sparc": sparc(speed[start : end - 1], fps),
        "ldlj": log_dimensionless_jerk(path[:, :3], fps),
        "rms_jerk": rms_jerk(path[:, :3], fps),
        "rms_angular_jerk": rms_angular_jerk(path[:, 3:6], fps),
        "boundary_ratio": boundary_ratio(actions[:, :3], horizon),
        "gripper_reversals": gripper_reversals(actions[:, 6], fps),
        "duration": float((end - start) / fps),
        "peak_speed": float(speed.max()),
    }


@dataclasses.dataclass
class Episode:
    pose: np.ndarray  # [frames, 6] xyz [m] + rotation vector [rad]
    actions: np.ndarray  # [frames, 7] commanded xyz + rotation vector + gripper
    control: np.ndarray  # [frames]


def load_recording(root: str | pathlib.Path) -> tuple[list[Episode], float]:
    """The episodes of a LeRobot v3.0 recording (numeric columns only) and its frame rate."""
    import pandas as pd

    root = pathlib.Path(root)
    fps = float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    columns = ["episode_index", "frame_index", "observation.tcp_pose", "actions", "control_flag"]
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data/*/*.parquet under {root}")
    table = pd.concat([pd.read_parquet(path, columns=columns) for path in files])
    episodes = []
    for _, frames in table.sort_values(["episode_index", "frame_index"]).groupby("episode_index", sort=True):
        episodes.append(
            Episode(
                np.stack([np.asarray(value, dtype=np.float64) for value in frames["observation.tcp_pose"]]),
                np.stack([np.asarray(value, dtype=np.float64) for value in frames["actions"]]),
                np.asarray([int(np.ravel(value)[0]) for value in frames["control_flag"]]),
            )
        )
    return episodes, fps


def _longest_run(selected: np.ndarray) -> slice:
    best, start = slice(0, 0), None
    for index, value in enumerate([*selected, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best.stop - best.start:
                best = slice(start, index)
            start = None
    return best


def summarize_run(roots: Sequence[str], control: str, horizon: int, seed: int = 0) -> dict:
    """Per-episode metrics of the longest selected segment of every episode, their mean with a 95% bootstrap
    interval, median and standard deviation over episodes, and the speed and jerk curves of a typical episode."""
    rows, paths = [], []
    for root in roots:
        episodes, fps = load_recording(root)
        has_policy = any((episode.control == POLICY_FLAG).any() for episode in episodes)
        mode = ("policy" if has_policy else "all") if control == "auto" else control
        for episode in episodes:
            selected = {
                "policy": episode.control == POLICY_FLAG,
                "human": episode.control == HUMAN_FLAG,
                "all": np.ones(len(episode.control), dtype=bool),
            }[mode]
            segment = _longest_run(selected)
            metrics = episode_metrics(episode.pose[segment], episode.actions[segment], fps, horizon)
            if metrics is not None:
                rows.append(metrics)
                paths.append((episode.pose[segment, :3], fps))
    if not rows:
        raise ValueError(f"No episode of {list(roots)} has a movement of at least {MIN_SAMPLES} frames ({control}).")
    rng = np.random.default_rng(seed)
    summary = {}
    for name in rows[0]:
        values = np.array([row[name] for row in rows])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        resampled = rng.choice(values, size=(2000, len(values))).mean(axis=1)
        summary[name] = {
            "mean": float(values.mean()),
            "ci95": [float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))],
            "median": float(np.median(values)),
            "std": float(values.std()),
        }
    median_episode = int(np.argsort([row["sparc"] for row in rows])[len(rows) // 2])
    path, fps = paths[median_episode]
    speed, jerk = speed_profile(path, fps), jerk_profile(path, fps)
    return {
        "episodes": len(rows),
        "summary": summary,
        "per_episode": rows,
        "median_speed_profile": {"time": (np.arange(len(speed)) / fps).tolist(), "speed": speed.tolist()},
        "median_jerk_profile": {"time": ((np.arange(len(jerk)) + 1) / fps).tolist(), "jerk": jerk.tolist()},
    }


def draw(results: dict[str, dict]) -> svg_chart.Canvas:
    """Full-width figure: one panel of horizontal bars per metric, and the speed and jerk of a typical episode of
    each run."""
    rows = 18 * len(results)
    canvas = svg_chart.Canvas(700, 2 * (rows + 62) + 240)
    canvas.text(
        690,
        14,
        "shorter bars are smoother (LDLJ: log dimensionless jerk); whiskers: 95% bootstrap interval of the mean",
        size=10.5,
        anchor="end",
        fill=svg_chart.MUTED,
    )
    labels = [f"{label} (n = {result['episodes']})" for label, result in results.items()]
    # Room for the longest label (about 5.6 px per character) left of the bars, and for the value right of them.
    label_width = min(max(5.6 * max(len(label) for label in labels) + 8, 90.0), 220.0)
    colors = svg_chart.PALETTE
    for index, (name, (title, value_format)) in enumerate(METRICS.items()):
        entries = [result["summary"].get(name) for result in results.values()]
        offset = min([0.0, *(entry["ci95"][0] for entry in entries if entry)])
        # Bars start at zero: negative metrics (SPARC, LDLJ) are drawn by their magnitude.
        sign = -1.0 if offset < 0 else 1.0
        svg_chart.hbar_panel(
            canvas,
            (10 + label_width + 345 * (index % 2), 44 + (rows + 62) * (index // 2), 283 - label_width, rows),
            labels,
            [sign * entry["mean"] if entry else None for entry in entries],
            intervals=[tuple(sorted(sign * bound for bound in entry["ci95"])) if entry else None for entry in entries],
            colors=colors,
            title=f"|{title}|" if sign < 0 else title,
            notes=[value_format.format(entry["mean"]) if entry else None for entry in entries],
            label_width=label_width,
        )
    legend_y, legend_x = 2 * (rows + 62) + 34, 60.0
    for index, label in enumerate(results):
        canvas.line([(legend_x, legend_y), (legend_x + 18, legend_y)], colors[index % len(colors)], width=2.4)
        canvas.text(legend_x + 23, legend_y + 3.5, label, size=10.5, fill=svg_chart.MUTED)
        legend_x += 45 + 5.6 * len(label)
    top = legend_y + 40
    for column, (key, value, title, y_label) in enumerate(
        (
            ("median_speed_profile", "speed", "TCP speed of the median-SPARC episode", "speed [m/s]"),
            ("median_jerk_profile", "jerk", "TCP jerk of the same episode", "jerk [m/s³]"),
        )
    ):
        series = [
            (label, result[key]["time"], result[key][value], colors[i % len(colors)])
            for i, (label, result) in enumerate(results.items())
        ]
        svg_chart.line_panel(
            canvas,
            (60 + 345 * column, top, 275, 120),
            series,
            title=title,
            x_label="time [s]",
            y_label=y_label,
            legend=False,
        )
    return canvas


def format_table(results: dict[str, dict]) -> str:
    names = ["sparc", "ldlj", "rms_jerk", "rms_angular_jerk", "boundary_ratio", "gripper_reversals"]
    lines = [f"{'run':<24}{'episodes':>9}" + "".join(f"{name:>20}" for name in names)]
    for label, result in results.items():
        cells = []
        for name in names:
            entry = result["summary"].get(name)
            cells.append(f"{entry['mean']:.3f} ± {entry['std']:.3f}" if entry else "n/a")
        lines.append(f"{label:<24}{result['episodes']:>9}" + "".join(f"{cell:>20}" for cell in cells))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=RECORDING (several ':'-separated recordings are merged)."
    )
    parser.add_argument("--control", choices=("auto", "policy", "human", "all"), default="auto")
    parser.add_argument("--horizon", type=int, default=10, help="Action chunk length in frames.")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Write OUT.svg (+ .pdf, .png) and OUT.json.")
    args = parser.parse_args(argv)
    args.runs = {}
    for run in args.run:
        label, separator, paths = run.partition("=")
        if not separator or not label or not paths:
            parser.error(f"--run expects LABEL=RECORDING, got {run!r}")
        args.runs[label] = [path for path in paths.split(":") if path]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = {label: summarize_run(roots, args.control, args.horizon) for label, roots in args.runs.items()}
    print(format_table(results))
    if args.out is not None:
        figure = draw(results).save(args.out.with_suffix(".svg"))
        build_figures.render(figure)
        args.out.with_suffix(".json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"wrote {figure} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
