import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(description="Run reinforcement learning tasks.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---------------------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------------------

    train_parser = subparsers.add_parser("train", help="Train a policy.")

    train_parser.add_argument(
        "--task-section", type=str, help="Name of the task section in tasks.yaml."
    )
    train_parser.add_argument(
        "--start-task", type=int, default=0, help="Index of the first task."
    )
    train_parser.add_argument(
        "--end-task", type=int, default=None, help="Index of the last task."
    )
    train_parser.add_argument(
        "--init-task", type=int, default=None, help="Index of the initialization task."
    )
    train_parser.add_argument(
        "--n-trained-agents", type=int, default=1, help="Number of agents to train."
    )
    train_parser.add_argument(
        "--sequential-curriculum",
        action="store_true",
        help="Activate a sequential curriculum.",
    )
    train_parser.add_argument(
        "--policy-name", type=str, default=None, help="Name of the trained policy."
    )

    train_group = train_parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "--max-steps", type=int, default=None, help="Maximum number of training steps."
    )
    train_group.add_argument(
        "--no-max-steps",
        action="store_true",
        help="Train without a maximal number of steps.",
    )

    # ---------------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------------

    validation_parser = subparsers.add_parser(
        "validate", help="Validate a trained policy."
    )

    validation_parser.add_argument(
        "--task-section", type=str, help="Name of the task section in tasks.yaml."
    )
    validation_parser.add_argument(
        "--policy-name", type=str, help="Name of the trained policy."
    )
    validation_parser.add_argument(
        "--start-task", type=int, default=0, help="Index of the first task."
    )
    validation_parser.add_argument(
        "--end-task", type=int, default=None, help="Index of the last task."
    )
    validation_parser.add_argument(
        "--n-episodes", type=int, default=500, help="Number of episodes to run."
    )
    validation_parser.add_argument(
        "--n-renders",
        type=int,
        default=5,
        help="Number of episodes to render per task.",
    )

    # ---------------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------------

    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained policy."
    )

    evaluation_parser.add_argument(
        "--task-section",
        type=str,
        help="Name of the task section in tasks.yaml.",
    )
    evaluation_parser.add_argument(
        "--start-task", type=int, default=0, help="Index of the first task."
    )
    evaluation_parser.add_argument(
        "--end-task", type=int, default=None, help="Index of the last task."
    )
    evaluation_parser.add_argument(
        "--policy-name",
        type=str,
        help="Name of the trained policy.",
    )
    evaluation_parser.add_argument(
        "--n-episodes", type=int, default=500, help="Number of episodes to run."
    )
    evaluation_parser.add_argument(
        "--n-renders",
        type=int,
        default=5,
        help="Number of episodes to render per task.",
    )

    # ---------------------------------------------------------------------------------
    # Animation
    # ---------------------------------------------------------------------------------

    animation_parser = subparsers.add_parser(
        "animate", help="Generate renders and animations."
    )

    animation_parser.add_argument(
        "--task-section",
        type=str,
        help="Name of task section in tasks.yaml.",
    )
    animation_parser.add_argument(
        "--policy-name", type=str, help="Name of the trained policy."
    )
    animation_parser.add_argument(
        "--path", type=str, default="figures/demo", help="Path to save the animation."
    )
    animation_parser.add_argument(
        "--fps", type=int, default=10, help="Number of fps in the animation/"
    )

    # ---------------------------------------------------------------------------------
    # Demo
    # ---------------------------------------------------------------------------------

    demo_parser = subparsers.add_parser("demo", help="Run a demonstration.")

    demo_parser.add_argument(
        "--task-section", type=str, help="Name of the task section in tasks.yaml."
    )
    demo_parser.add_argument(
        "--policy-name", type=str, help="Name of the trained policy."
    )
    demo_parser.add_argument(
        "--path", type=str, default="figures/demo", help="Path to the animation."
    )

    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    return build_parser().parse_args(argv)
