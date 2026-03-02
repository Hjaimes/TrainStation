"""CLI entry point: python run.py config.yaml [--override key=value] [--mode train]"""
from __future__ import annotations
import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="TrainStation CLI")
    parser.add_argument("config", nargs="?", help="Path to config file (YAML or JSON)")
    parser.add_argument("--override", "-o", action="append", default=[],
                        help="Config override key=value (repeatable)")
    parser.add_argument("--mode", default="train",
                        choices=["train", "cache-latents", "cache-text", "cache-all"])
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_models:
        from trainer.registry import list_models
        for name in list_models():
            print(f"  - {name}")
        return

    if not args.config:
        parser.error("Config file is required.")

    try:
        from trainer.config.io import load_config, apply_overrides
        config = load_config(args.config)
        if args.override:
            config = apply_overrides(config, args.override)

        if args.validate_only:
            from trainer.config.validation import validate_config
            result = validate_config(config)
            for issue in result.all_issues():
                prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[issue.level]
                print(f"  [{prefix}] {issue.message}")
            sys.exit(1 if result.has_errors else 0)

        from trainer.training.session import TrainingSession
        from trainer.callbacks import CLIProgressCallback
        session = TrainingSession()
        session.start(config, callbacks=[CLIProgressCallback()], mode=args.mode)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).error(str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
