from runner import run_evaluation, run_save, run_simulation, run_test, run_train
from utils.config_loader import load_experiments

if __name__ == "__main__":

    MODE = "train"
    experiments = load_experiments()

    if MODE == "simulation":
        run_simulation(experiments=experiments)
    elif MODE == "test":
        run_test(experiments=experiments, render=True)
    elif MODE == "save":
        run_save(experiments=experiments)
    elif MODE == "train":
        run_train(experiments=experiments[:2], n_steps=[10_000, 10_000])
        run_evaluation(experiments=experiments[:2], n_episodes=100, n_render=5)
