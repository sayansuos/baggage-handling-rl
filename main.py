import argparse

from runner import run_animation, run_demo, run_evaluation, run_train, run_validation
from utils.config_loader import load_tasks
from utils.plotting import plot_renders


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL tasks.")

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train the agent.",
    )
    train_parser.add_argument(
        "--start-task",
        type=int,
        default=0,
        help="Index of the first training task.",
    )
    train_parser.add_argument(
        "--end-task",
        type=int,
        default=None,
        help="Index after the last training task.",
    )
    train_parser.add_argument(
        "--previous-task",
        type=int,
        default=None,
        help="Index of the task used to initialize the training.",
    )
    train_parser.add_argument(
        "--trained-agent-id",
        type=str,
        default="agent_1",
        help="Identifiant of the agent that has to be trained.",
    )
    train_parser.add_argument(
        "--probabilistic-curriculum",
        action="store_true",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    validation_parser = subparsers.add_parser(
        "validate",
        help="Validate a trained policy.",
    )
    validation_parser.add_argument(
        "--start-task",
        type=int,
        default=0,
        help="Index of the first validation task.",
    )
    validation_parser.add_argument(
        "--end-task",
        type=int,
        default=None,
        help="Index of the last validation task.",
    )
    validation_parser.add_argument(
        "--n-episodes", type=int, default=100, help="Number of episodes to validate"
    )
    validation_parser.add_argument(
        "--n-render",
        type=int,
        default=5,
        help="Number of episodes to render per task.",
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
        "--start-task",
        type=int,
        default=0,
        help="Index of the first evaluation task.",
    )
    evaluation_parser.add_argument(
        "--end-task",
        type=int,
        default=None,
        help="Index of the last evaluation task.",
    )
    evaluation_parser.add_argument(
        "--policy-name",
        type=str,
        default=-1,
        help="Name of the policy that has to be evaluated.",
    )
    evaluation_parser.add_argument(
        "--n-episodes", type=int, default=500, help="Number of episodes to evaluate"
    )
    evaluation_parser.add_argument(
        "--n-render",
        type=int,
        default=5,
        help="Number of episodes to render per task.",
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
        "--policy-name",
        type=str,
        default="curriculum",
        help="Name of the policy that has to be demonstrated.",
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
        "--policy-name",
        type=str,
        default="curriculum",
        help="Name of the policy that has to be animated.",
    )
    animation_parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Number of frames per second in the animation",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_tasks = load_tasks()

    if args.mode == "train":
        tasks = train_tasks[args.start_task : args.end_task]
        previous_task = (
            train_tasks[args.previous_task] if args.previous_task is not None else None
        )
        trained_agent_id = args.trained_agent_id
        fixed_curriculum = not args.probabilistic_curriculum

        run_train(
            tasks=tasks,
            previous_task=previous_task,
            trained_agent_id=trained_agent_id,
            fixed_curriculum=fixed_curriculum,
        )

    elif args.mode == "validate":
        tasks = train_tasks[args.start_task : args.end_task]
        n_episodes = args.n_episodes
        n_render = args.n_render
        trained_agent_id = args.trained_agent_id

        run_validation(
            tasks=tasks,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "evaluate":
        eval_tasks = load_tasks(obj="eval")
        tasks = eval_tasks[args.start_task : args.end_task]
        policy_name = args.policy_name
        n_episodes = args.n_episodes
        n_render = args.n_render
        trained_agent_id = args.trained_agent_id

        run_evaluation(
            tasks=tasks,
            policy_name=policy_name,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "demo":
        eval_tasks = load_tasks(obj="eval")
        policy_name = args.policy_name
        trained_agent_id = args.trained_agent_id

        run_demo(
            tasks=train_tasks + eval_tasks,
            policy_name=policy_name,
            trained_agent_id=trained_agent_id,
        )

    elif args.mode == "animate":
        eval_tasks = load_tasks(obj="eval")
        policy_name = args.policy_name
        fps = args.fps
        path = f"figures/demo/{policy_name}"

        plot_renders(
            tasks=train_tasks,
            path=path,
            file_name="train",
        )
        plot_renders(
            tasks=eval_tasks,
            path=path,
            file_name="eval",
        )

        run_animation(
            tasks=train_tasks,
            policy_name=policy_name,
            path=path,
            file_name="train",
            fps=fps,
        )
        run_animation(
            tasks=eval_tasks,
            policy_name=policy_name,
            path=path,
            file_name="eval",
            fps=fps,
        )


if __name__ == "__main__":
    main()
