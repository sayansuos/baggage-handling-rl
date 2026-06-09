from simulator.runner import run_simulation, run_test, run_save, run_train
from simulator.utils.load_config import load_experiments

if __name__ == "__main__":

    MODE = "save"
    experiments = load_experiments()

    if MODE == "simulation":
        run_simulation(experiments=experiments)
    elif MODE == "test":
        run_test(experiments=experiments, render=True)
    elif MODE == "save":
        run_save(experiments=experiments)
    elif MODE == "train":
        run_train(experiments=experiments[:1])
