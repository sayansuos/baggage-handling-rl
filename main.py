from pathlib import Path

import numpy as np

from cli import parse_args
from runner import run_animation, run_demo, run_evaluation, run_train, run_validation
from utils.config_loader import load_tasks
from utils.plotting import plot_renders


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------------------

    if args.mode == "train":
        end_task = args.end_task + 1 if args.end_task is not None else None
        tasks = load_tasks(section=args.task_section)[args.start_task : end_task]

        if args.no_max_steps:
            max_steps = np.inf
        elif args.max_steps is None:
            max_steps = sum(task.n_steps for task in tasks)
        else:
            max_steps = args.max_steps

        run_train(
            tasks=tasks,
            policy_name=args.policy_name,
            n_trained_agents=args.n_trained_agents,
            init_policy_name=args.init_policy_name,
            init_checkpoint_name=args.init_checkpoint_name,
            max_steps=max_steps,
            sequential_curriculum=args.sequential_curriculum,
        )

    # ---------------------------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------------------------

    elif args.mode == "validate":
        end_task = args.end_task + 1 if args.end_task is not None else None
        tasks = load_tasks(section=args.task_section)[args.start_task : end_task]

        run_validation(
            tasks=tasks,
            policy_name=args.policy_name,
            checkpoint_strategy=args.checkpoint_strategy,
            n_episodes=args.n_episodes,
            n_renders=args.n_renders,
        )

    # ---------------------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------------------

    elif args.mode == "evaluate":
        end_task = args.end_task + 1 if args.end_task is not None else None
        tasks = load_tasks(section=args.task_section)[args.start_task : end_task]

        run_evaluation(
            tasks=tasks,
            policy_name=args.policy_name,
            checkpoint_name=args.checkpoint_name,
            n_episodes=args.n_episodes,
            n_renders=args.n_renders,
        )

    # ---------------------------------------------------------------------------------
    # Animate
    # ---------------------------------------------------------------------------------

    elif args.mode == "animate":
        tasks = load_tasks(section=args.task_section)

        plot_renders(
            tasks=tasks,
            path=Path("figures/demo"),
            file_name=args.task_section,
            max_ncols=4,
        )

        run_animation(
            tasks=tasks,
            policy_name=args.policy_name,
            checkpoint_name=args.checkpoint_name,
            file_name=args.task_section,
            fps=args.fps,
        )

    # ---------------------------------------------------------------------------------
    # Demo
    # ---------------------------------------------------------------------------------

    elif args.mode == "demo":
        run_demo(
            policy_name=args.policy_name,
            checkpoint_name=args.checkpoint_name,
            file_name=args.task_section,
        )


if __name__ == "__main__":
    main()
