import argparse

from runner import run_animation, run_demo, run_evaluation, run_train, run_validation
from utils.config_loader import load_experiments
from utils.plotting import plot_renders


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL experiments.")

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    # Train arguments
    train_parser = subparsers.add_parser(
        "train",
        help="Train the agent.",
    )
    train_parser.add_argument(
        "--start-exp",
        type=int,
        default=0,
        help="Index of the first training experiment.",
    )
    train_parser.add_argument(
        "--end-exp",
        type=int,
        default=None,
        help="Index after the last training experiment.",
    )
    train_parser.add_argument(
        "--previous-exp",
        type=int,
        default=None,
        help="Index of the experiment used to initialize the training.",
    )
    train_parser.add_argument(
        "--trained-agent-id",
        type=str,
        default="agent_1",
        help="Identifiant of the agent that has to be trained.",
    )

    # Validation arguments
    validation_parser = subparsers.add_parser(
        "validate",
        help="Validate a trained policy.",
    )
    validation_parser.add_argument(
        "--start-exp",
        type=int,
        default=0,
        help="Index of the first validation experiment.",
    )
    validation_parser.add_argument(
        "--end-exp",
        type=int,
        default=None,
        help="Index of the last validation experiment.",
    )
    validation_parser.add_argument(
        "--n-episodes", type=int, default=100, help="Number of episodes to validate"
    )
    validation_parser.add_argument(
        "--n-render",
        type=int,
        default=5,
        help="Number of episodes to render per experiment.",
    )
    validation_parser.add_argument(
        "--trained-agent-id",
        type=str,
        default="agent_1",
        help="Identifiant of the agent that has been trained.",
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained policy.",
    )
    evaluation_parser.add_argument(
        "--start-exp",
        type=int,
        default=0,
        help="Index of the first evaluation experiment.",
    )
    evaluation_parser.add_argument(
        "--end-exp",
        type=int,
        default=None,
        help="Index of the last evaluation experiment.",
    )
    evaluation_parser.add_argument(
        "--policy",
        type=int,
        default=-1,
        help="Index of the training experiment used as policy.",
    )
    evaluation_parser.add_argument(
        "--n-episodes", type=int, default=500, help="Number of episodes to evaluate"
    )
    evaluation_parser.add_argument(
        "--n-render",
        type=int,
        default=5,
        help="Number of episodes to render per experiment.",
    )
    evaluation_parser.add_argument(
        "--trained-agent-id",
        type=str,
        default="agent_1",
        help="Identifiant of the agent that has been trained.",
    )

    # ------------------------------------------------------------------
    # Demo
    # ------------------------------------------------------------------
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a demonstration.",
    )
    demo_parser.add_argument(
        "--policy",
        type=int,
        default=-1,
        help="Index of the training experiment used as policy.",
    )
    demo_parser.add_argument(
        "--trained-agent-id",
        type=str,
        default="agent_1",
        help="Identifiant of the agent that has been trained.",
    )

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    animation_parser = subparsers.add_parser(
        "animate",
        help="Generate renders and animations.",
    )
    animation_parser.add_argument(
        "--policy",
        type=int,
        default=-1,
        help="Index of the training experiment used as policy.",
    )
    animation_parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Number of frames per second in the animation",
    )
    animation_parser.add_argument(
        "--path",
        type=str,
        default="figures/demo",
        help="Directory to save the files in",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_experiments = load_experiments()

    if args.mode == "train":
        experiments = train_experiments[args.start_exp : args.end_exp]
        previous_exp = (
            train_experiments[args.previous_exp]
            if args.previous_exp is not None
            else None
        )
        trained_agent_id = args.trained_agent_id

        run_train(
            experiments=experiments,
            previous_exp=previous_exp,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "validation":
        experiments = train_experiments[args.start_exp : args.end_exp]
        n_episodes = args.n_episodes
        n_render = args.n_render
        trained_agent_id = args.trained_agent_id

        run_validation(
            experiments=experiments,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "evaluation":
        eval_experiments = load_experiments(obj="eval")
        experiments = eval_experiments[args.start_exp : args.end_exp]
        policy = train_experiments[args.policy]
        n_episodes = args.n_episodes
        n_render = args.n_render
        trained_agent_id = args.trained_agent_id

        run_evaluation(
            experiments=experiments,
            policy=policy,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "demo":
        eval_experiments = load_experiments(obj="eval")
        policy = train_experiments[args.policy]
        trained_agent_id = args.trained_agent_id

        run_demo(
            experiments=train_experiments + eval_experiments,
            policy=policy,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "animation":
        eval_experiments = load_experiments(obj="eval")
        policy = train_experiments[args.policy]
        fps = args.fps
        path = args.path

        plot_renders(
            experiments=train_experiments,
            path=path,
            file_name="train",
        )
        plot_renders(
            experiments=eval_experiments,
            path=path,
            file_name="eval",
        )

        run_animation(
            experiments=train_experiments,
            policy=policy,
            path=path,
            file_name="train",
            fps=fps,
        )
        run_animation(
            experiments=eval_experiments,
            policy=policy,
            path=path,
            file_name="eval",
            fps=fps,
        )


if __name__ == "__main__":
    main()
