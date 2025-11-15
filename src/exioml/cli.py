"""Command line interface exposed through ``python -m exioml``."""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import pandas as pd

from .factors import list_regions, list_years, load_factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exioml",
        description="Inspect and query ExioML emission-factor tables",
    )
    parser.add_argument(
        "--schema",
        default="PxP",
        help="Schema to query (PxP or IxI)",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List available regions for the selected schema",
    )
    parser.add_argument(
        "--list-years",
        action="store_true",
        help="List available years for the selected schema",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Region codes to filter (space separated)",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="Years to filter (space separated integers)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional extra columns to display",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Rows to display when querying data",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_regions:
        for region in list_regions(args.schema):
            print(region)
        return 0
    if args.list_years:
        for year in list_years(args.schema):
            print(year)
        return 0

    years = _coerce_ints(args.years)
    df = load_factor(
        schema=args.schema,
        years=years,
        regions=args.regions,
        columns=args.columns,
    )
    if args.limit:
        df = df.head(args.limit)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))
    return 0


def _coerce_ints(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not values:
        return None
    return [int(value) for value in values]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
