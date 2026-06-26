from runner import run_animation, run_evaluation, run_simulation, run_test, run_train
from utils.config_loader import load_experiments

if __name__ == "__main__":

    MODE = "other"
    experiments = load_experiments()

    if MODE == "simulation":
        run_simulation(experiments=experiments)
    elif MODE == "test":
        run_test(experiments=experiments, render=True)
    elif MODE == "animation":
        run_animation(experiments=experiments)
    elif MODE == "train":
        run_train(experiments=experiments)
        run_evaluation(experiments=experiments, n_episodes=100, n_render=5)
    elif MODE == "other":
        run_train(experiments=experiments[3:], previous_exp=experiments[2])
        run_evaluation(experiments=experiments, n_episodes=100, n_render=5)
