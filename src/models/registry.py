from __future__ import annotations

import argparse

from src.models.workflow import promote, rollback


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote or roll back a versioned fraud model.")
    parser.add_argument("action", choices=["promote", "rollback"])
    parser.add_argument("version", help="Registry version such as v1")
    args = parser.parse_args()

    if args.action == "promote":
        promote(args.version)
    else:
        rollback(args.version)
    print(f"{args.action.title()}d {args.version} to models/production")


if __name__ == "__main__":
    main()
