"""Draws the result figures of the paper from picture/src/results.json, the measured numbers only:

* picture/results_real_robot.{svg,pdf,png}: success rate of occluded insertion per method and start distance, with
  95% Wilson intervals from the success counts. Cells that have not been measured are drawn as empty "TBD" bars.
* picture/results_sensor_attribution.{svg,pdf,png}: what the policies read, from scripts/ar3_sensor_attribution.py.

Edit the counts in results.json when new trials are run and re-run:

    .venv/bin/python scripts/build_result_figures.py

It also prints the intervals and Fisher's exact tests between the measured cells of each condition.
"""

import itertools
import json
import math
import pathlib

from scipy import stats

if __package__:
    from scripts import build_figures
    from scripts import svg_chart
else:
    import build_figures
    import svg_chart

ROOT = pathlib.Path(__file__).resolve().parents[1] / "picture"
GREY, ORANGE, GREEN, PURPLE = svg_chart.PALETTE[:4]


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a success probability."""
    rate = successes / trials
    scale = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / scale
    half = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / scale
    return max(centre - half, 0.0), min(centre + half, 1.0)


def real_robot_figure(data: dict) -> svg_chart.Canvas:
    """Column-width figure: one panel of horizontal bars per start condition."""
    methods = data["methods"]
    rows = 17 * len(methods)
    canvas = svg_chart.Canvas(340, 2 * rows + 118)
    colors = [PURPLE if method["name"].startswith("muscleVLTA") else GREY for method in methods]
    for index, (key, title) in enumerate(data["conditions"].items()):
        cells = [method[key] for method in methods]
        svg_chart.hbar_panel(
            canvas,
            (120, 26 + index * (rows + 62), 150, rows),
            [method["name"] for method in methods],
            [100 * cell[0] / cell[1] if cell else None for cell in cells],
            intervals=[tuple(100 * bound for bound in wilson_interval(*cell)) if cell else None for cell in cells],
            colors=colors,
            title=title,
            x_label="success rate [%]",
            x_max=100.0,
            notes=[f"{cell[0]}/{cell[1]}" if cell else None for cell in cells],
            label_width=114,
        )
    return canvas


def attribution_figure(data: dict) -> svg_chart.Canvas:
    """Column-width figure: action change of the trained policy, and the held-out error increase of the pilots."""
    canvas = svg_chart.Canvas(340, 300)
    change = data["trained_policy"]["change"]
    rows = {
        "arm level swapped": "level swapped",
        "arm changes removed": "changes removed",
        "hand removed": "touch removed",
    }
    phases = {"free": "free", "contact": "contact"}
    svg_chart.hbar_panel(
        canvas,
        (138, 26, 130, 102),
        [f"{label} · {phase}" for label in rows.values() for phase in phases.values()],
        [change[row][phase] for row in rows for phase in phases],
        colors=[ORANGE, ORANGE, ORANGE, ORANGE, GREEN, GREEN],
        title="Action change under counterfactual readings",
        x_label="change [std of the recorded actions]",
        x_max=0.5,
        value_format="{:.3f}",
        label_width=132,
    )
    policies = data["pilots"]["policies"]
    names = {"shared fusion": "shared", "balanced fusion": "balanced"}
    svg_chart.hbar_panel(
        canvas,
        (138, 196, 130, 68),
        [f"{label} · {phase}" for label in names.values() for phase in phases.values()],
        [
            100 * (policies[name]["arm level swapped"][phase] / policies[name]["error"][phase] - 1)
            for name in names
            for phase in phases
        ],
        colors=[GREY, GREY, PURPLE, PURPLE],
        title="Held-out error increase, level swapped",
        x_label="increase of the action error [%]",
        x_max=75.0,
        value_format="+{:.0f}%",
        label_width=132,
    )
    return canvas


def report(data: dict) -> str:
    lines = []
    for key, title in data["conditions"].items():
        measured = [(method["name"], method[key]) for method in data["methods"] if method[key]]
        lines.append(title)
        for name, (successes, trials) in measured:
            low, high = wilson_interval(successes, trials)
            lines.append(
                f"  {name:<18} {successes}/{trials} = {100 * successes / trials:.2f}%  "
                f"95% Wilson [{100 * low:.1f}, {100 * high:.1f}]"
            )
        for (first, a), (second, b) in itertools.combinations(measured, 2):
            p_value = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
            lines.append(f"  Fisher exact, {first} vs {second}: p = {p_value:.2g}")
    return "\n".join(lines)


def main() -> None:
    data = json.loads((ROOT / "src" / "results.json").read_text(encoding="utf-8"))
    print(report(data["real_robot"]))
    for name, canvas in (
        ("results_real_robot", real_robot_figure(data["real_robot"])),
        ("results_sensor_attribution", attribution_figure(data["attribution"])),
    ):
        figure = canvas.save(ROOT / f"{name}.svg")
        rendered = build_figures.render(figure)
        print(f"wrote {', '.join(path.name for path in [figure, *rendered])}")


if __name__ == "__main__":
    main()
