import csv
import decimal
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, kstest, probplot
from scipy.optimize import curve_fit
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--drop_zeros", action="store_true", default=False, help="Drop zero valued data points from analysis.")
args = parser.parse_args()


K_INIT, GAMMA_INIT = 2, 10  # selected initial fit parameters based on intuition


def weibull_pdf(x, c, scale):
    """Weibull probability density function."""
    return weibull_min.pdf(x, c, scale=scale)


def compute_fit_statistics(wind_speeds, k, gamma, method_name):
    """Compute AIC, BIC, and KS test statistic/p-value for Weibull fit."""
    if k is None or gamma is None:
        return None, None, None, None

    n = len(wind_speeds)
    log_likelihood = np.sum(weibull_min.logpdf(wind_speeds, k, loc=0, scale=gamma))
    aic = 2 * 2 - 2 * log_likelihood
    bic = 2 * np.log(n) - 2 * log_likelihood

    # Perform KS test (note: loc is 0 for all our fits)
    ks_stat, ks_pval = kstest(wind_speeds, "weibull_min", args=(k, 0, gamma), method="asymp")

    return aic, bic, ks_stat, ks_pval


def plot_qq_plots(monthly_wind_speeds, monthly_params, fit_methods, month_names, input_file_name):
    """
    Generate QQ plots for each month and fitting method (MLE, MM, LS) on the same subplot.

    Parameters:
    monthly_wind_speeds (dict): Dictionary of wind speeds grouped by month.
    monthly_params (dict): Fitting parameters (k, gamma) for each month and method.
    fit_methods (list): List of fitting methods ("mle", "mm", "ls").
    month_names (list): List of month names.
    input_file_name (str): Name of the input file for labeling.
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.ravel()

    colors = {"mle": "red", "mm": "green", "ls": "blue"}

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]

        ax.set_title(f"{month_names[month-1]}")

        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                continue

            res = probplot(wind_speeds, dist="weibull_min", sparams=(k, 0, gamma), plot=None)
            ax.plot(res[0][1], res[0][1], color="black", markersize=1)
            ax.plot(res[0][0], res[0][1], "o", color=colors[method], label=f"{method.upper()}", markersize=2, alpha=0.4)

        ax.grid(True)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"./images/qq_plots/{input_file_name}.png", dpi=300)
    plt.close()


def plot_pp_plots(monthly_wind_speeds, monthly_params, fit_methods, month_names, input_file_name):
    """
    Generate PP plots for each month and fitting method (MLE, MM, LS) on the same subplot.

    Parameters:
    monthly_wind_speeds (dict): Dictionary of wind speeds grouped by month.
    monthly_params (dict): Fitting parameters (k, gamma) for each month and method.
    fit_methods (list): List of fitting methods ("mle", "mm", "ls").
    month_names (list): List of month names.
    input_file_name (str): Name of the input file for labeling.
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.ravel()

    colors = {"mle": "red", "mm": "green", "ls": "blue"}

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]

        ax.set_title(f"{month_names[month-1]}")

        has_theoretical_line = False
        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                continue

            # Sort the wind speed data and calculate the observed CDF
            sorted_data = np.sort(wind_speeds)
            observed_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            # Calculate the theoretical CDF using the Weibull distribution
            theoretical_cdf = weibull_min.cdf(sorted_data, k, scale=gamma)

            # Plot the PP plot (observed CDF vs theoretical CDF)
            if not has_theoretical_line:
                ax.plot(theoretical_cdf, theoretical_cdf, color="black", markersize=1)
                has_theoretical_line = True
            ax.plot(observed_cdf, theoretical_cdf, "o", color=colors[method], label=f"{method.upper()}", markersize=3, alpha=0.4)

        ax.set_xlabel("Observed CDF")
        ax.set_ylabel("Theoretical CDF")
        ax.grid(True)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"./images/pp_plots/{input_file_name}.png", dpi=300)
    plt.close()


def fit_weibull_to_monthly_wind_data(monthly_wind_speeds, input_file):
    """
    Fit Weibull distributions to monthly wind speed data using multiple fitting methods and create visualization.

    Parameters:
    monthly_wind_speeds (dict): Dictionary with months as keys and wind speed arrays as values

    Returns:
    dict: Dictionary with months as keys  and dict of fitting method key and parameters (k, gamma) as values
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fit_methods = ["mle", "mm", "ls"]  # Maximum likelihood, method of moments, and least squares
    colors = {"mle": "red", "mm": "green", "ls": "blue"}
    line_styles = {"mle": "-", "mm": "--", "ls": ":"}

    # Create a 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.ravel()

    params = {}

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]
        params[month] = {}

        # Plot histogram of data points
        bins = np.arange(0, np.ceil(max(wind_speeds)) + 1, 1)
        ax.hist(wind_speeds, bins=bins, density=True, alpha=0.7, color="skyblue", label="Wind Speed Data")
        for method in fit_methods:
            try:
                if method in ["mle", "mm"]:
                    k, loc, gamma = weibull_min.fit(wind_speeds, floc=0, method=method)
                    params[month][method] = (k, gamma)
                elif method == "ls":
                    # Scipy doesnt have a built-in method for least squares fitting so do it using curve_fit
                    try:
                        hist, bin_edges = np.histogram(wind_speeds, bins="auto", density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        fitted_params, _ = curve_fit(
                            weibull_pdf, bin_centers, hist, p0=[K_INIT, GAMMA_INIT], bounds=([1.0, 1e-10], [np.inf, np.inf])
                        )  # prevent k from being < 1
                        k, gamma = fitted_params
                        params[month][method] = (k, gamma)

                    except Exception as e:
                        print(f"Detailed LS fitting error for month {month}:")
                        print(f"  Error type: {type(e).__name__}")
                        print(f"  Error message: {str(e)}")
                        params[month][method] = (None, None)

                x = np.linspace(0, max(wind_speeds), 1000)
                pdf = weibull_min.pdf(x, k, loc=0, scale=gamma)
                ax.plot(x, pdf, color=colors[method], linestyle=line_styles[method], lw=2, label=f"{method.upper()}\n(k={k:.2f}, γ={gamma:.2f})")

            except Exception as e:
                print(f"Warning: {method} fitting failed for month {month}: {str(e)}")
                params[month][method] = (None, None)

        ax.set_title(f"{month_names[month-1]}")
        ax.grid(True, alpha=0.3)
        if month in [10, 11, 12]:  # Bottom row
            ax.set_xlabel("Wind Speed")
        if month in [1, 4, 7, 10]:  # Left column
            ax.set_ylabel("Probability Density")
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"./images/weibull_distributions/{input_file}.png", dpi=300)
    plt.close()

    return params


def read_wind_data(file_path, drop_zeros=False):
    """
    Read wind speed data from a CSV file and group by month, ignoring invalid or missing data.
    Optionally drops zero wind speed values.
    """
    try:
        df = pd.read_csv(file_path, usecols=["wdsp", "date"], low_memory=False)

        df["wdsp"] = pd.to_numeric(df["wdsp"], errors="coerce")

        df = df.dropna(subset=["wdsp", "date"])

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["date"])

        # Optionally drop zero wind speed values
        if drop_zeros:
            df = df[df["wdsp"] != 0]

        df["month"] = df["date"].dt.month

        monthly_speeds = {month: group["wdsp"].values for month, group in df.groupby("month")}
        monthly_dates = {month: group["date"].values for month, group in df.groupby("month")}

        if not drop_zeros:
            for month, speeds in monthly_speeds.items():
                zero_count = np.sum(speeds == 0)
                print(f"Month {month}: {zero_count} zero wind speed measurements")

        return monthly_speeds, monthly_dates

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None


if __name__ == "__main__":

    input_files = [
        ("hly532", "dublin_airport"),
        ("hly1075", "cork_roches_point"),
        ("hly3904", "cork_airport"),
        ("hly875", "kildare_mullingar"),
        ("hly1875", "galway_athenry"),
        ("hly2075", "donegal_finner"),
        ("hly2275", "kerry_valentia_observatory"),
    ]
    file_to_save_to = "./data/weibull_parameters.csv"
    stats_file = "./data/weibull_fitting_stats.csv"

    os.makedirs("./images/weibull_distributions")
    os.makedirs("./images/qq_plots")
    os.makedirs("./images/pp_plots")

    with open(stats_file, "w", newline="") as stats_csvfile:
        writer = csv.writer(stats_csvfile)
        writer.writerow(
            [
                "Dataset",
                "Month",
                "Method",
                "Shape Parameter (k)",
                "Scale Parameter (gamma)",
                "Num Data Points",
                "AIC",
                "BIC",
                "KS Statistic",
                "KS p-value",
            ]
        )

    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    for input_file, input_file_name in input_files:
        monthly_wind_speeds, monthly_dates = read_wind_data(f"./data/{input_file}.csv", args.drop_zeros)
        if monthly_wind_speeds is None:
            print(f"Failed to read wind speed data for {input_file_name} from ./data/{input_file}.csv")
            continue

        monthly_params = fit_weibull_to_monthly_wind_data(monthly_wind_speeds, input_file_name)

        plot_qq_plots(monthly_wind_speeds, monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)
        plot_pp_plots(monthly_wind_speeds, monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)

        with open(file_to_save_to, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for month, methods in monthly_params.items():
                for method, (k, gamma) in methods.items():
                    n_points = len(monthly_wind_speeds[month])
                    writer.writerow([input_file_name, month_names[month - 1], f"{k:.4f}", f"{gamma:.4f}", method, n_points])

        # Compute and save goodness-of-fit statistics
        with open(stats_file, "a", newline="") as stats_csvfile:
            writer = csv.writer(stats_csvfile)
            for month, methods in monthly_params.items():
                wind_speeds = monthly_wind_speeds[month]
                for method, (k, gamma) in methods.items():
                    aic, bic, ks_stat, ks_pval = compute_fit_statistics(np.array(wind_speeds), k, gamma, method)
                    n_points = len(wind_speeds)
                    writer.writerow(
                        [
                            input_file_name,
                            month_names[month - 1],
                            method,
                            f"{k:.4f}" if k is not None else None,
                            f"{gamma:.5f}" if gamma is not None else None,
                            n_points,
                            f"{aic:.5f}" if aic is not None else None,
                            f"{bic:.5f}" if bic is not None else None,
                            f"{ks_stat:.5f}" if ks_stat is not None else None,
                            f"{ks_pval:.5f}" if ks_pval is not None else 0.00000,
                        ]
                    )

        print(f"Weibull parameters and stats for {input_file_name} saved.")
