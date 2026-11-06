"""Builds the frame-aligned Tac G2 indentation-map cache of AR3 LeRobot v3.0 datasets (needed for tactile training).

For every frame it takes the newest ``tactile/episode-*/tactile_depth.sensor_*.parquet`` map received at or before the
frame time, area-pools it to 48x64 and writes ``<dataset>/tactile_cache/depth_48x64/``. Run it once per dataset
(about a minute per 20 GB of sidecars with 16 workers); re-running skips up-to-date caches. Example:

    python scripts/build_ar3_tactile_cache.py --dataset /data/jiang/ar3_eef_forceVisuoTactaile_plugIn --workers 16
"""

import argparse
from collections.abc import Sequence
import logging

from openpi.data import tactile_cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", required=True, help="LeRobot v3.0 dataset roots.")
    parser.add_argument("--out", default=None, help="Cache directory (only with one dataset; default: inside it).")
    parser.add_argument(
        "--max-age",
        type=float,
        default=None,
        help="Maps older than this many seconds count as missing (default: meta/ar3_capture.json robot.max_tactile_age_s).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Episodes processed in parallel.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing caches.")
    args = parser.parse_args(argv)
    if args.out is not None and len(args.dataset) != 1:
        parser.error("--out needs exactly one --dataset")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for dataset in args.dataset:
        out = tactile_cache.build_cache(
            dataset, args.out, max_age_s=args.max_age, workers=args.workers, overwrite=args.overwrite
        )
        print(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
