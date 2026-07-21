from runner import run_animation, run_demo, run_evaluation, run_train, run_validation
from utils.config_loader import load_experiments
from utils.plotting import plot_renders

if __name__ == "__main__":
    MODE = "train"
    train_experiments = load_experiments()
    eval_experiments = load_experiments(obj="eval")

    if MODE == "demo":
        run_demo(
            experiments=train_experiments + eval_experiments,
            policy=train_experiments[-1],
            trained_agent_id="agent_1",
        )

    elif MODE == "animation":
        plot_renders(
            experiments=train_experiments, path="figures/demo", file_name="train"
        )
        plot_renders(
            experiments=eval_experiments, path="figures/demo", file_name="eval"
        )
        run_animation(
            experiments=train_experiments,
            policy=train_experiments[-1],
            path="figures/demo",
            file_name="train",
            fps=10,
        )
        run_animation(
            experiments=eval_experiments,
            policy=train_experiments[-1],
            path="figures/demo",
            file_name="eval",
            fps=10,
        )

    elif MODE == "train":
        run_train(experiments=train_experiments[1:2], previous_exp=train_experiments[0])
        run_validation(experiments=train_experiments[1:2], n_episodes=100, n_render=5)

    elif MODE == "validation":
        run_validation(experiments=train_experiments[:1], n_episodes=100, n_render=5)

    elif MODE == "evaluation":
        run_evaluation(
            experiments=eval_experiments,
            policy=train_experiments[-1],
            n_episodes=500,
            n_render=5,
        )
