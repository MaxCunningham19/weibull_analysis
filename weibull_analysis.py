import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import pandas as pd


def fit_weibull_to_wind_data(wind_speeds):
    """
    Fit a Weibull distribution to wind speed data and return the shape (k) and scale (gamma) parameters.

    Parameters:
    wind_speeds (array-like): Array of wind speed measurements

    Returns:
    tuple: Shape parameter k and scale parameter gamma of the fitted Weibull distribution
    """

    k, loc, gamma = weibull_min.fit(wind_speeds, floc=0)  # Fix location parameter to 0,  uses maximum likelihood estimate

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()

    ax1.hist(wind_speeds, bins="auto", density=True, alpha=0.7, color="skyblue", label="Wind Speed Data")

    x = np.linspace(0, max(wind_speeds), 1000)
    pdf = weibull_min.pdf(x, k, loc=0, scale=gamma)
    cdf = weibull_min.cdf(x, k, loc=0, scale=gamma)

    ax1.plot(x, pdf, "r-", lw=2, label=f"Fitted Weibull PDF\n(k={k:.2f}, γ={gamma:.2f})")

    ax2.plot(x, cdf, "g-", lw=2, label="Fitted Weibull CDF")

    ax1.set_xlabel("Wind Speed")
    ax1.set_ylabel("Probability Density", color="r")
    ax2.set_ylabel("Cumulative Probability", color="g")
    plt.title("Wind Speed Distribution with Fitted Weibull")

    ax1.grid(True, alpha=0.3)

    ax1.tick_params(axis="y", labelcolor="r")
    ax2.tick_params(axis="y", labelcolor="g")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.show()

    return k, gamma


def read_wind_data(file_path):
    """
    Read wind speed data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing wind speed data

    Returns:
    numpy.ndarray: Array of wind speed values
    """
    try:
        df = pd.read_csv(file_path)

        wind_speeds = df["wdsp"].values

        wind_speeds = wind_speeds[~np.isnan(wind_speeds)]

        return wind_speeds

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


if __name__ == "__main__":

    wind_speeds = read_wind_data("./hly532/hly532.csv")
    if wind_speeds is None:
        print("Failed to read wind speed data")
        exit()
    k, gamma = fit_weibull_to_wind_data(wind_speeds)
    print(f"Shape parameter k: {k}")
    print(f"Scale parameter gamma: {gamma}")
