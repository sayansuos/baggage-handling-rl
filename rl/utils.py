import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_running_avg(returns, env):
    """
    Plot the running average of returns with a window of 100 games.

    This function calculates the running average of a list of returns and
    plots the result using matplotlib. The running average is calculated
    over a window of 100 games, providing a smooth plot of return trends over time.

    Parameters
    ----------
    returns : list or numpy.ndarray
        A list or numpy array containing the returns from consecutive games.

    Notes
    -----
    This function assumes that `returns` is a list or array of numerical values
    that represent the returns obtained in each game or episode. The running
    average is computed and plotted, which is useful for visualizing performance
    trends in tasks such as games or simulations.

    Examples
    --------
    >>> returns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> plot_running_avg(returns)
    This will plot a graph showing the running average of the returns over a window of 10 games.
    """
    avg = np.zeros_like(returns)
    for i in range(len(returns)):
        avg[i] = np.mean(returns[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_running_avg.png")


def save_training_metrics(
    metrics: list[dict],
    file_name: str,
    path: str = "logs/training",
):
    """
    Save SAC training metrics and plot return evolution.
    """

    df = pd.DataFrame(metrics)
    df.to_csv(f"{path}/{file_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["episode"], df["return"], label="Score", alpha=0.4)
    ax.plot(df["episode"], df["average_return"], label="Average return")
    ax.plot(df["episode"], df["best_return"], label="Best return", linestyle="--")

    ax.set_title(f"SAC training - {file_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"{path}/{file_name}.png", dpi=300)
    plt.close(fig)

    return df
