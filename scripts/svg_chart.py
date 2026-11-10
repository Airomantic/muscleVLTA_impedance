"""Charts as plain SVG: only rect, path, circle and text with presentation attributes, the subset PowerPoint and Word
display (as in the exported architecture figures). matplotlib's SVG output relies on <use>, clip paths and CSS."""

from collections.abc import Sequence
import math
import pathlib
from xml.sax.saxutils import escape

FONT = "Arial, Helvetica, 'Liberation Sans', sans-serif"
INK, MUTED, GRID = "#102A43", "#52606D", "#DDE3EA"
PALETTE = ("#7C8DA6", "#E07B24", "#1E9E87", "#7A5BC2", "#3B6FB6", "#D7263D")


class Canvas:
    def __init__(self, width: float, height: float) -> None:
        self.width, self.height = width, height
        self._parts: list[str] = [f'<rect width="{width:g}" height="{height:g}" fill="#FFFFFF"/>']

    def rect(self, x, y, width, height, fill, *, stroke=None, stroke_width=1.0, rx=0.0, dash=None) -> None:
        extra = f' stroke="{stroke}" stroke-width="{stroke_width:g}"' if stroke else ""
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{rx:g}" fill="{fill}"{extra}/>'
        )

    def line(self, points: Sequence[tuple[float, float]], stroke, *, width=1.2, dash=None) -> None:
        path = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{width:g}" stroke-linejoin="round"{extra}/>'
        )

    def circle(self, x, y, radius, fill, *, stroke=None) -> None:
        extra = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self._parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:g}" fill="{fill}"{extra}/>')

    def text(self, x, y, text, *, size=12.0, anchor="start", weight=None, style=None, fill=INK, rotate=None) -> None:
        extra = f' font-weight="{weight}"' if weight else ""
        extra += f' font-style="{style}"' if style else ""
        extra += f' transform="rotate({rotate:g} {x:.2f} {y:.2f})"' if rotate else ""
        self._parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:g}" text-anchor="{anchor}" fill="{fill}"{extra}>'
            f"{escape(str(text))}</text>"
        )

    def svg(self) -> str:
        head = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:g}" height="{self.height:g}" '
            f'viewBox="0 0 {self.width:g} {self.height:g}" font-family="{FONT}">\n'
        )
        return head + "\n".join(self._parts) + "\n</svg>\n"

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.svg(), encoding="utf-8")
        return path


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """About ``count`` round tick values covering [low, high]."""
    if high <= low:
        high = low + 1.0
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda factor: abs(factor * magnitude - raw)) * magnitude
    first = math.floor(low / step) * step
    ticks = [first + index * step for index in range(math.ceil((high - first) / step) + 1)]
    return [round(tick, 10) for tick in ticks]


def _frame(canvas: Canvas, box, ticks, *, title, y_label, tick_format) -> tuple[float, float]:
    x, y, width, height = box
    low, high = ticks[0], ticks[-1]
    for tick in ticks:
        tick_y = y + height - (tick - low) / (high - low) * height
        canvas.line([(x, tick_y), (x + width, tick_y)], GRID, width=1)
        canvas.text(x - 6, tick_y + 3.5, tick_format.format(tick), size=10, anchor="end", fill=MUTED)
    canvas.line([(x, y), (x, y + height)], MUTED, width=1)
    if title:
        canvas.text(x, y - 12, title, size=12.5, weight="bold")
    if y_label:
        canvas.text(x - 40, y + height / 2, y_label, size=11, anchor="middle", fill=MUTED, rotate=-90)
    return low, high


def _wrap(label: str, width: float, size: float) -> list[str]:
    """Breaks a label at spaces into lines of at most ``width`` (about 0.52 em per character)."""
    limit = max(int(width / (0.52 * size)), 4)
    lines: list[str] = []
    for part in str(label).split("\n"):
        line = ""
        for word in part.split(" "):
            if line and len(line) + 1 + len(word) > limit:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
    return lines


def bar_panel(
    canvas: Canvas,
    box: tuple[float, float, float, float],
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    intervals: Sequence[tuple[float, float] | None] | None = None,
    colors: Sequence[str] | None = None,
    title: str = "",
    y_label: str = "",
    y_max: float | None = None,
    tick_format: str = "{:g}",
    value_format: str = "{:.2f}",
    notes: Sequence[str | None] | None = None,
) -> None:
    """Bars with optional whiskers (``intervals`` = (low, high)). A ``None`` value draws an empty dashed bar marked
    "TBD" (not measured). ``labels`` wrap at spaces and line breaks; ``notes`` replace the printed values (e.g.
    "13/60")."""
    x, y, width, height = box
    measured = [value for value in values if value is not None]
    tops = [interval[1] for interval in (intervals or []) if interval is not None]
    upper = y_max if y_max is not None else max([*measured, *tops, 1e-9]) * 1.12
    ticks = nice_ticks(0.0, upper)
    low, high = _frame(canvas, box, ticks, title=title, y_label=y_label, tick_format=tick_format)
    slot = width / max(len(values), 1)
    bar_width = min(0.62 * slot, 46.0)
    for index, value in enumerate(values):
        centre = x + (index + 0.5) * slot
        color = (colors or PALETTE)[index % len(colors or PALETTE)]
        if value is None:
            canvas.rect(
                centre - bar_width / 2, y + 0.55 * height, bar_width, 0.45 * height, "none", stroke=MUTED, dash="3 3"
            )
            canvas.text(centre, y + 0.55 * height - 5, "TBD", size=10, anchor="middle", fill=MUTED, style="italic")
        else:
            top = y + height - (value - low) / (high - low) * height
            canvas.rect(centre - bar_width / 2, top, bar_width, y + height - top, color)
            label_y = top - 5
            interval = intervals[index] if intervals else None
            if interval is not None:
                y_low = y + height - (interval[0] - low) / (high - low) * height
                y_high = y + height - (interval[1] - low) / (high - low) * height
                canvas.line([(centre, y_low), (centre, y_high)], INK, width=1.2)
                canvas.line([(centre - 4, y_high), (centre + 4, y_high)], INK, width=1.2)
                canvas.line([(centre - 4, y_low), (centre + 4, y_low)], INK, width=1.2)
                label_y = y_high - 5
            note = notes[index] if notes and notes[index] is not None else value_format.format(value)
            canvas.text(centre, label_y, note, size=10, anchor="middle")
        for line_index, line in enumerate(_wrap(labels[index], slot - 4, 10.5)):
            canvas.text(centre, y + height + 14 + 12 * line_index, line, size=10.5, anchor="middle", fill=MUTED)
    canvas.line([(x, y + height), (x + width, y + height)], MUTED, width=1)


def hbar_panel(
    canvas: Canvas,
    box: tuple[float, float, float, float],
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    intervals: Sequence[tuple[float, float] | None] | None = None,
    colors: Sequence[str] | None = None,
    title: str = "",
    x_label: str = "",
    x_max: float | None = None,
    tick_format: str = "{:g}",
    value_format: str = "{:.2f}",
    notes: Sequence[str | None] | None = None,
    label_width: float = 100.0,
) -> None:
    """Horizontal bars, one row per label, for many or long category names. ``box`` is the bar area; the labels are
    right-aligned in the ``label_width`` to its left. Whiskers, "TBD" rows and ``notes`` as in ``bar_panel``."""
    x, y, width, height = box
    measured = [value for value in values if value is not None]
    tops = [interval[1] for interval in (intervals or []) if interval is not None]
    ticks = nice_ticks(0.0, x_max if x_max is not None else max([*measured, *tops, 1e-9]) * 1.2, 4)
    high = ticks[-1]
    for tick in ticks:
        tick_x = x + tick / high * width
        canvas.line([(tick_x, y), (tick_x, y + height)], GRID, width=1)
        canvas.text(tick_x, y + height + 13, tick_format.format(tick), size=10, anchor="middle", fill=MUTED)
    canvas.line([(x, y), (x, y + height)], MUTED, width=1)
    if title:
        canvas.text(x - label_width, y - 9, title, size=12, weight="bold")
    if x_label:
        canvas.text(x + width / 2, y + height + 27, x_label, size=10.5, anchor="middle", fill=MUTED)
    slot = height / max(len(values), 1)
    thickness = min(0.62 * slot, 13.0)
    for index, value in enumerate(values):
        centre = y + (index + 0.5) * slot
        canvas.text(x - 7, centre + 3.6, labels[index], size=10.5, anchor="end", fill=INK)
        if value is None:
            canvas.rect(x, centre - thickness / 2, 0.22 * width, thickness, "none", stroke=MUTED, dash="3 3")
            canvas.text(x + 0.22 * width + 5, centre + 3.5, "TBD", size=10, fill=MUTED, style="italic")
            continue
        end = x + value / high * width
        canvas.rect(
            x, centre - thickness / 2, max(end - x, 0.0), thickness, (colors or PALETTE)[index % len(colors or PALETTE)]
        )
        interval = intervals[index] if intervals else None
        if interval is not None:
            x_low, x_high = x + interval[0] / high * width, x + interval[1] / high * width
            canvas.line([(x_low, centre), (x_high, centre)], INK, width=1.2)
            canvas.line([(x_low, centre - 3.5), (x_low, centre + 3.5)], INK, width=1.2)
            canvas.line([(x_high, centre - 3.5), (x_high, centre + 3.5)], INK, width=1.2)
            end = x_high
        note = notes[index] if notes and notes[index] is not None else value_format.format(value)
        canvas.text(end + 5, centre + 3.5, note, size=10)


def line_panel(
    canvas: Canvas,
    box: tuple[float, float, float, float],
    series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    tick_format: str = "{:g}",
    legend: bool = True,
) -> None:
    """Curves ``(label, xs, ys, color)`` over shared axes, with a legend in the top-right corner of the panel."""
    x, y, width, height = box
    x_low = min(min(xs) for _, xs, _, _ in series)
    x_high = max(max(xs) for _, xs, _, _ in series)
    lowest = min(min(ys) for _, _, ys, _ in series)
    ticks = nice_ticks(min(0.0, lowest), max(max(ys) for _, _, ys, _ in series))
    low, high = _frame(canvas, box, ticks, title=title, y_label=y_label, tick_format=tick_format)
    for index, (label, xs, ys, color) in enumerate(series):
        points = [
            (x + (vx - x_low) / max(x_high - x_low, 1e-9) * width, y + height - (vy - low) / (high - low) * height)
            for vx, vy in zip(xs, ys, strict=True)
        ]
        canvas.line(points, color, width=1.6)
        if legend:
            canvas.line(
                [(x + width - 118, y + 9 + 14 * index), (x + width - 100, y + 9 + 14 * index)], color, width=2.4
            )
            canvas.text(x + width - 95, y + 12.5 + 14 * index, label, size=10.5, fill=MUTED)
    canvas.line([(x, y + height), (x + width, y + height)], MUTED, width=1)
    for tick in nice_ticks(x_low, x_high):
        if x_low <= tick <= x_high:
            tick_x = x + (tick - x_low) / max(x_high - x_low, 1e-9) * width
            canvas.text(tick_x, y + height + 14, f"{tick:g}", size=10, anchor="middle", fill=MUTED)
    if x_label:
        canvas.text(x + width / 2, y + height + 30, x_label, size=11, anchor="middle", fill=MUTED)
