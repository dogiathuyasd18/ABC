"""Top-level commands for the 9th ABC head-pose project."""

from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Data pipeline (see python -m src.prep.pipeline -h)")
    train = sub.add_parser("train")
    eval_p = sub.add_parser("evaluate")
    infer = sub.add_parser("infer")

    args, rest = parser.parse_known_args(argv)
    if args.command == "prepare":
        from src.prep.pipeline import main as prepare_main
        prepare_main()
    elif args.command == "train":
        from src.training.train import parse_args, train
        train(parse_args(rest))
    elif args.command == "evaluate":
        from src.evaluation.evaluate import evaluate, parse_args
        evaluate(parse_args(rest))
    elif args.command == "infer":
        from src.inference.predict import main as infer_main
        infer_main(rest)


if __name__ == "__main__":
    main()
