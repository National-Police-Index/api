"""Command-line entry points for the entity-resolution pipeline.

Two ways to run the SAME PostMatcher pipeline:
  from-csv   batch-resolve a CSV of mentions, write the result files
  from-name  resolve a single mention typed on the command line (candidates from the API)

    python -m resolve.cli from-name --first Scott --last Lunger --state CA \
        --year 2015 --source-agency "Hayward Police Department" --api http://localhost:8001

    python -m resolve.cli from-csv --input data/input/involved_officers.csv \
        --api http://localhost:8001 --default-state CA
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from resolve.io import build_mention, read_mentions, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resolve", description="NPI entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api", default=None,
                        help="API base URL (else NPI_API_URL env, else localhost:8000)")
    common.add_argument("--threshold", type=float, default=0.5)
    common.add_argument("--require-state", dest="require_state", action="store_true", default=True)
    common.add_argument("--no-require-state", dest="require_state", action="store_false")

    p_name = sub.add_parser("from-name", parents=[common],
                            help="resolve a single mention from CLI args")
    p_name.add_argument("--first", required=True)
    p_name.add_argument("--last", required=True)
    p_name.add_argument("--state", default=None)
    p_name.add_argument("--year", type=int, required=True)
    p_name.add_argument("--source-agency", dest="source_agency", default="")
    p_name.add_argument("--mentioned", dest="mentioned_agencies", default="")
    p_name.add_argument("--middle", dest="middle_name", default="")

    p_csv = sub.add_parser("from-csv", parents=[common], help="batch-resolve a mentions CSV")
    p_csv.add_argument("--input", required=True)
    p_csv.add_argument("--output-dir", dest="output_dir", default="resolve/data/output/all_states")
    p_csv.add_argument("--default-state", dest="default_state", default=None)
    p_csv.add_argument("--sample-n", dest="sample_n", type=int, default=None)
    p_csv.add_argument("--sample-seed", dest="sample_seed", type=int, default=None)

    return parser


def _make_matcher(args):
    from resolve.pipeline import PostMatcher
    return PostMatcher(api_url=args.api, require_state=args.require_state,
                       threshold=args.threshold)


def cmd_from_name(args, matcher=None):
    matcher = matcher or _make_matcher(args)
    mention = build_mention({
        "first_name": args.first,
        "last_name": args.last,
        "middle_name": args.middle_name,
        "source_agency": args.source_agency,
        "mentioned_agencies": args.mentioned_agencies,
        "incident_year": args.year,
        "state": args.state,
    })
    result = matcher.resolve_one(mention)
    _print_verdict(result)
    return result


def cmd_from_csv(args, matcher=None):
    matcher = matcher or _make_matcher(args)
    mentions = read_mentions(args.input, default_state=args.default_state,
                             sample_n=args.sample_n, sample_seed=args.sample_seed)
    results = matcher.resolve_batch(mentions)
    paths = write_outputs(results, args.output_dir)
    auto = sum(1 for r in results if r.status == "auto_matched")
    print(f"Resolved {len(results)} mentions: {auto} auto-matched, "
          f"{len(results) - auto} routed to review.")
    print(f"Outputs written to {args.output_dir}")
    return paths


def _print_verdict(result):
    if result.status == "auto_matched":
        m = result.match or {}
        print(f"AUTO-MATCHED -> {m.get('post_person_nbr', '?')} "
              f"({m.get('post_agency_name', '')})")
    else:
        print(f"REVIEW: {result.reason}")
        if result.candidates:
            print(f"  ({len(result.candidates)} candidate(s) considered)")


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "from-name":
        cmd_from_name(args)
    elif args.command == "from-csv":
        cmd_from_csv(args)


if __name__ == "__main__":
    main()
