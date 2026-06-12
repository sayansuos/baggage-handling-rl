from runner import run_save, run_simulation, run_test, run_train
from utils.config_loader import load_experiments

if __name__ == "__main__":

    MODE = "train"
    experiments = load_experiments()

    if MODE == "simulation":
        run_simulation(experiments=experiments)
    elif MODE == "test":
        run_test(experiments=experiments, render=False)
    elif MODE == "save":
        run_save(experiments=experiments)
    elif MODE == "train":
        run_train(experiments=experiments[:1], n_episodes=1000)
