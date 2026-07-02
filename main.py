from runner import (
    run_animation,
    run_evaluation,
    run_test,
    run_train,
    run_validation,
)
from utils.config_loader import load_experiments
from utils.plotting import plot_renders

if __name__ == "__main__":

    MODE = "test"
    train_experiments = load_experiments()
    eval_experiments = load_experiments(obj="eval")

    if MODE == "test":
        plot_renders(
            experiments=train_experiments, path="figures/test", file_name="train"
        )
        plot_renders(
            experiments=eval_experiments, path="figures/test", file_name="eval"
        )
        # run_test(
        #     experiments=train_experiments + eval_experiments,
        #     best_exp=train_experiments[-1],
        #     render=True,
        # )
    elif MODE == "animation":
        run_animation(experiments=train_experiments)
    elif MODE == "train":
        run_train(experiments=train_experiments)
        run_validation(experiments=train_experiments, n_episodes=100, n_render=5)
    elif MODE == "validation":
        run_validation(experiments=train_experiments[8:], n_episodes=500, n_render=5)
    elif MODE == "evaluation":
        run_evaluation(
            experiments=eval_experiments,
            policy=train_experiments[-1],
            n_episodes=500,
            n_render=5,
        )
