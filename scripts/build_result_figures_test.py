import json

import pytest

from scripts import build_result_figures as figures


def test_wilson_interval_matches_the_intervals_printed_in_the_paper():
    assert figures.wilson_interval(0, 60) == pytest.approx((0.0, 0.060), abs=1e-3)
    assert figures.wilson_interval(3, 60) == pytest.approx((0.017, 0.137), abs=1e-3)
    assert figures.wilson_interval(13, 60) == pytest.approx((0.131, 0.336), abs=1e-3)
    assert figures.wilson_interval(55, 60) == pytest.approx((0.819, 0.964), abs=1e-3)


def test_report_and_chart_follow_the_counts_of_the_data_file():
    data = json.loads((figures.ROOT / "src" / "results.json").read_text(encoding="utf-8"))["real_robot"]
    report = figures.report(data)
    # Fisher's exact test of the far-start counts: 13/60 against 3/60 and against 0/60.
    assert "p = 0.014" in report
    assert "p = 0.00012" in report

    methods = {method["name"]: method for method in data["methods"]}
    methods["LIFT"]["far"] = [6, 60]
    chart = figures.real_robot_figure(data).svg()
    # Measured cells show their counts; cells without counts stay marked.
    assert "13/60" in chart
    assert "6/60" in chart
    assert "TBD" in chart
