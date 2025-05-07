from collections import defaultdict
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, kstest, probplot, ks_2samp
from scipy.optimize import curve_fit
import pandas as pd
import argparse
import seaborn as sns
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument("--drop_zeros", action="store_true", default=False, help="Drop zero valued data points from analysis.")
args = parser.parse_args()


K_INIT, GAMMA_INIT = 2, 10  # selected initial fit parameters based on intuition

colors = {"mle": "red", "mm": "green", "ls": "blue"}
line_styles = {"mle": ":", "mm": "--", "ls": "-"}
alphas = {"mle": 0.9, "mm": 0.7, "ls": 0.5}


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


def compute_wind_farm_statistics(k, gamma):
    """
    Computes summary statistics from a Weibull distribution with parameters k and gamma.

    Returns a tuple: (mean, CI_95_Low, CI_95_High, Pct_10_15, Pct_GT_25)
    """
    dist = weibull_min(c=k, scale=gamma)

    mean = dist.mean()
    ci_low = dist.ppf(0.025)
    ci_high = dist.ppf(0.975)
    pct_10_15 = dist.cdf(15) - dist.cdf(10)
    pct_gt_25 = 1 - dist.cdf(25)

    return mean, ci_low, ci_high, pct_10_15, pct_gt_25


def plot_percentile_stats(monthly_params, fit_methods, month_names, input_file_name):
    """
    Generate plots of Pct_10_15 and Pct_GT_25 for each month and method on the same plot.

    Parameters:
    monthly_params (dict): Dictionary of fitting parameters (k, gamma) for each month and method.
    fit_methods (list): List of fitting methods ("mle", "mm", "ls").
    month_names (list): List of month names.
    input_file_name (str): Name of the input file for labeling and saving.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 17))
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for month in range(1, 13):
        ax = axes[month - 1]
        ax.set_title(f"{month_names[month - 1]}")

        pct_10_15_vals = []
        pct_gt_25_vals = []

        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                pct_10_15_vals.append(0)
                pct_gt_25_vals.append(0)
            else:
                _, _, _, pct_10_15, pct_gt_25 = compute_wind_farm_statistics(k, gamma)
                pct_10_15_vals.append(pct_10_15)
                pct_gt_25_vals.append(pct_gt_25)

        (line1,) = ax.plot(fit_methods, pct_10_15_vals, label="Pct_10_15", color="skyblue", marker="o")
        (line2,) = ax.plot(fit_methods, pct_gt_25_vals, label="Pct_GT_25", color="salmon", marker="s")

        if month == 1:
            legend_handles.extend([line1, line2])
            legend_labels.extend(["Pct_10_15", "Pct_GT_25"])

        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Fit Method")
        ax.grid(True, alpha=0.3)

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"./images/percentile_plots/{input_file_name}.png", dpi=500)
    plt.close()


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
    fig, axes = plt.subplots(3, 4, figsize=(20, 17))
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]

        min_q, max_q = float("inf"), float("-inf")

        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                continue

            res = probplot(wind_speeds, dist="weibull_min", sparams=(k, 0, gamma), plot=None)
            theoretical_q = res[0][0]
            empirical_q = res[0][1]

            (plot_line,) = ax.plot(
                theoretical_q, empirical_q, color=colors[method], linestyle=line_styles[method], lw=2, label=f"{method.upper()}", alpha=alphas[method]
            )

            if month == 1:
                legend_handles.append(plot_line)
                legend_labels.append(f"{method.upper()}")

            min_q = min(min(theoretical_q), min(empirical_q), min_q)
            max_q = max(max(theoretical_q), max(empirical_q), max_q)

        ax.plot([min_q, max_q], [min_q, max_q], "k", lw=2, alpha=0.6)

        ax.set_title(f"{month_names[month - 1]}")
        ax.grid(True, alpha=0.3)
        if month in [9, 10, 11, 12]:  # Bottom row
            ax.set_xlabel("Theoretical Quantiles")
        if month in [1, 5, 9]:  # Left column
            ax.set_ylabel("Observed Quantiles")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(f"./images/qq_plots/{input_file_name}.png", dpi=500)
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
    fig, axes = plt.subplots(3, 4, figsize=(20, 17))
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]

        sorted_data = np.sort(wind_speeds)
        observed_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(observed_cdf, observed_cdf, "k", lw=2, alpha=1.0)

        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                continue

            theoretical_cdf = weibull_min.cdf(sorted_data, k, scale=gamma)
            (plot_line,) = ax.plot(
                theoretical_cdf,
                observed_cdf,
                color=colors[method],
                linestyle=line_styles[method],
                lw=2,
                label=f"{method.upper()}",
                alpha=alphas[method],
            )

            if month == 1:
                legend_handles.append(plot_line)
                legend_labels.append(f"{method.upper()}")

        ax.set_title(f"{month_names[month - 1]}")
        ax.grid(True, alpha=0.3)
        if month in [9, 10, 11, 12]:  # Bottom row
            ax.set_xlabel("Theoretical CDF")
        if month in [1, 5, 9]:  # Left column
            ax.set_ylabel("Observed CDF")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), fontsize=18)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(f"./images/pp_plots/{input_file_name}.png", dpi=500)
    plt.close()


def plot_cdf(monthly_wind_speeds, monthly_params, fit_methods, month_names, input_file_name):
    """
    Generate CDF plots (Wind Speed vs CDF) for each month and fitting method (MLE, MM, LS) on the same subplot.

    Parameters:
    monthly_wind_speeds (dict): Dictionary of wind speeds grouped by month.
    monthly_params (dict): Fitting parameters (k, gamma) for each month and method.
    fit_methods (list): List of fitting methods ("mle", "mm", "ls").
    month_names (list): List of month names.
    input_file_name (str): Name of the input file for labeling.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 17))
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for month in range(1, 13):
        wind_speeds = monthly_wind_speeds[month]
        ax = axes[month - 1]

        sorted_data = np.sort(wind_speeds)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        (empirical_plot,) = ax.plot(sorted_data, empirical_cdf, "k", label="Empirical", markersize=3)

        if month == 1:
            legend_handles.append(empirical_plot)
            legend_labels.append("Empirical")

        for method in fit_methods:
            k, gamma = monthly_params[month].get(method, (None, None))
            if k is None or gamma is None:
                continue

            theoretical_cdf = weibull_min.cdf(sorted_data, k, scale=gamma)
            (plot_line,) = ax.plot(
                sorted_data,
                theoretical_cdf,
                color=colors[method],
                linestyle=line_styles[method],
                lw=2,
                label=f"{method.upper()}",
                alpha=alphas[method],
            )

            if month == 1:
                legend_handles.append(plot_line)
                legend_labels.append(f"{method.upper()}")

        ax.set_title(f"{month_names[month-1]}")
        ax.grid(True, alpha=0.3)
        if month in [9, 10, 11, 12]:  # Bottom row
            ax.set_xlabel("Wind Speed")
        if month in [1, 5, 9]:  # Left column
            ax.set_ylabel("Probability Density")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the legend

    plt.savefig(f"./images/cdf_plots/{input_file_name}.png", dpi=500)
    plt.close()


def plot_ks_statistics(monthly_wind_speeds, month_names, input_file_name):
    ks_matrix = np.zeros((12, 12))

    for i in range(12):
        for j in range(12):
            data_i = monthly_wind_speeds[i + 1]
            data_j = monthly_wind_speeds[j + 1]

            if len(data_i) == 0 or len(data_j) == 0:
                ks_matrix[i, j] = np.nan
                continue

            ks_stat, _ = ks_2samp(data_i, data_j)
            ks_matrix[i, j] = ks_stat

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        ks_matrix, xticklabels=month_names, yticklabels=month_names, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "KS Statistic"}
    )

    plt.title(f"Pairwise KS Statistics Between Months\n{input_file_name.replace('_', ' ').title()}")
    plt.xlabel("Month")
    plt.ylabel("Month")
    plt.tight_layout()
    os.makedirs("./images/ks_heatmaps", exist_ok=True)
    plt.savefig(f"./images/ks_heatmaps/{input_file_name}_pairwise_ks.png", dpi=500)
    plt.close()
    return ks_matrix


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

    # Create a 4x3 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 17))
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
                ax.plot(x, pdf, color=colors[method], linestyle=line_styles[method], lw=2, label=f"{method.upper()}", alpha=alphas[method])

            except Exception as e:
                print(f"Warning: {method} fitting failed for month {month}: {str(e)}")
                params[month][method] = (None, None)

        ax.set_title(f"{month_names[month-1]}")
        ax.grid(True, alpha=0.3)
        if month in [9, 10, 11, 12]:  # Bottom row
            ax.set_xlabel("Wind Speed")
        if month in [1, 5, 9]:  # Left column
            ax.set_ylabel("Probability Density")
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"./images/weibull_distributions/{input_file}.png", dpi=500)
    plt.close()

    return params


def plot_avg_ks_per_method(averages_file_path, output_path="./images/model_comparison/avg_ks_by_method.png"):
    """
    Create a bar chart showing average KS statistic for each fitting method across all stations.
    """
    df = pd.read_csv(averages_file_path)

    df["Method"] = df["Method"].str.lower()

    avg_ks = df.groupby("Method")["KS Statistic"].mean().sort_values()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(avg_ks.index.str.upper(), avg_ks.values, color=[colors[m] for m in avg_ks.index])

    plt.title("Average KS Statistic per Fitting Method", fontsize=14)
    plt.ylabel("Average KS Statistic", fontsize=12)
    plt.xlabel("Fitting Method", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f"{yval:.3f}", ha="center", va="bottom", fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def plot_avg_params_per_station(averages_file_path, output_path="./images/model_comparison/avg_params_by_station.png"):
    """
    Create bar charts for average shape (k) and scale (gamma) per station for the chosen method (MLE).
    """
    df = pd.read_csv(averages_file_path)
    df_mle = df[df["Method"].str.lower() == "mle"]

    stations = df_mle["Dataset"]
    shape_k = df_mle["Shape Parameter (k)"].astype(float)
    scale_gamma = df_mle["Scale Parameter (gamma)"].astype(float)

    x = np.arange(len(stations))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, shape_k, width, label="Shape (k)", color="skyblue")
    plt.bar(x + width / 2, scale_gamma, width, label="Scale (γ)", color="lightcoral")

    plt.xticks(x, stations, rotation=45, ha="right")
    plt.ylabel("Parameter Value")
    plt.title("Average Weibull Parameters per Station (MLE)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=500)
    plt.close()


def get_station_colors(stations):
    cmap = plt.get_cmap("tab20", len(stations))
    base_colors = {station: cmap(i) for i, station in enumerate(stations)}
    light_dark_colors = {}

    for station, color in base_colors.items():
        r, g, b, _ = color
        lighter = (r * 0.8 + 0.2, g * 0.8 + 0.2, b * 0.8 + 0.2)
        darker = (r * 0.6, g * 0.6, b * 0.6)
        light_dark_colors[station] = {"Pct_10_15": lighter, "Pct_GT_25": darker, "middle": color}

    return light_dark_colors


def plot_monthly_wind_stats(stats_file_path, output_dir="./images/model_comparison"):
    df = pd.read_csv(stats_file_path)
    methods = df["Method"].unique()
    stations = df["Dataset"].unique()
    months = df["Month"].unique()
    station_colors = get_station_colors(stations)

    for method in methods:
        df_method = df[df["Method"] == method]
        fig, axes = plt.subplots(3, 4, figsize=(20, 17))
        axes = axes.ravel()

        for i, month in enumerate(months):
            ax = axes[i]
            filtered_data = df_method[df_method["Month"] == month]
            x = np.arange(len(stations))
            width = 0.34
            pct_10_15 = filtered_data["Pct_10_15"].astype(float)
            pct_gt_25 = filtered_data["Pct_GT_25"].astype(float)

            for j, station in enumerate(stations):
                color_set = station_colors[station]
                ax.bar(j - width / 2, pct_10_15.iloc[j], width, color=color_set["Pct_10_15"])
                ax.bar(j + width / 2, pct_gt_25.iloc[j], width, color=color_set["Pct_GT_25"])

            ax.set_xticks(x)
            ax.set_xticklabels(stations, rotation=35, ha="right", fontsize=5)
            if month in [1, 5, 9]:
                ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_title(f"{month}", fontsize=12)
            ax.grid(axis="y", alpha=0.3)

        sample_colors = list(station_colors.values())[0]
        legend_patches = [
            Patch(color=sample_colors["Pct_10_15"], label="Pct_10_15 (Lighter)"),
            Patch(color=sample_colors["Pct_GT_25"], label="Pct_GT_25 (Darker)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"wind_stats_method_{method.lower()}.png")
        plt.savefig(output_file_path, dpi=500, bbox_inches="tight")
        plt.close()


def plot_avg_wind_stats(stats_file_path, output_dir="./images/model_comparison"):
    df = pd.read_csv(stats_file_path)
    methods = df["Method"].unique()
    stations = df["Dataset"].unique()
    station_colors = get_station_colors(stations)

    for method in methods:
        df_method = df[df["Method"] == method]
        pct_10_15 = df_method["Pct_10_15"].astype(float)
        pct_gt_25 = df_method["Pct_GT_25"].astype(float)
        x = np.arange(len(stations))
        width = 0.4

        plt.figure(figsize=(12, 8))

        for i, station in enumerate(stations):
            color_set = station_colors[station]
            plt.bar(i - width / 2, pct_10_15.iloc[i], width, color=color_set["Pct_10_15"])
            plt.bar(i + width / 2, pct_gt_25.iloc[i], width, color=color_set["Pct_GT_25"])

        sample_colors = list(station_colors.values())[0]
        legend_patches = [
            Patch(color=sample_colors["Pct_10_15"], label="Pct_10_15 (Lighter)"),
            Patch(color=sample_colors["Pct_GT_25"], label="Pct_GT_25 (Darker)"),
        ]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=12)

        plt.xticks(x, stations, rotation=45, ha="right")
        plt.ylabel("Percentage (%)")
        plt.title(f"Average Wind Statistics per Station - Method: {method.upper()}", fontsize=14)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"avg_wind_stats_{method.lower()}.png")
        plt.savefig(output_file_path, dpi=500, bbox_inches="tight")
        plt.close()


def plot_mean_with_ci_by_method(csv_path, output_dir="./images/model_comparison"):
    df = pd.read_csv(csv_path)

    methods = df["Method"].unique()
    datasets = df["Dataset"].unique()
    month_labels = df["Month"].unique().tolist()

    station_colors = get_station_colors(datasets)
    os.makedirs(output_dir, exist_ok=True)

    for method in methods:
        df_method = df[df["Method"] == method]

        plt.figure(figsize=(12, 8))

        for dataset in datasets:
            df_subset = df_method[df_method["Dataset"] == dataset]
            if df_subset.empty:
                continue

            x = np.arange(len(df_subset))
            mean = df_subset["Mean"]
            ci_low = df_subset["CI_95_Low"]
            ci_high = df_subset["CI_95_High"]

            color = station_colors[dataset]
            plt.plot(x, mean, label=dataset, color=color["middle"], marker="o", linestyle="-", linewidth=2, markersize=8, alpha=0.8)
            plt.fill_between(x, ci_low, ci_high, color=color["middle"], alpha=0.25)
            plt.plot(x, ci_low, color=color["middle"], linestyle="--", linewidth=1, alpha=0.9)
            plt.plot(x, ci_high, color=color["middle"], linestyle="--", linewidth=1, alpha=0.9)

        plt.xticks(ticks=np.arange(len(month_labels)), labels=month_labels, rotation=45, fontsize=10)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Mean Wind Speed", fontsize=12)
        plt.title(f"Monthly Mean Wind Speed with 95% CI - Method: {method.upper()}", fontsize=14)
        plt.legend(title="Location", fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"mean_ci_{method.lower()}.png")
        plt.savefig(output_path, dpi=500)
        plt.close()


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
        ("hly875", "westmeath_mullingar"),
        ("hly1875", "galway_athenry"),
        ("hly2075", "donegal_finner"),
        ("hly2275", "kerry_valentia_observatory"),
    ]
    stats_file = "./data/weibull_fitting_stats.csv"
    averages_file = "./data/weibull_average_stats.csv"
    ks_files = "./data/monthly_comparisons.csv"

    os.makedirs("./images/weibull_distributions", exist_ok=True)
    os.makedirs("./images/qq_plots", exist_ok=True)
    os.makedirs("./images/pp_plots", exist_ok=True)
    os.makedirs("./images/cdf_plots", exist_ok=True)
    os.makedirs("./images/percentile_plots", exist_ok=True)
    os.makedirs("./images/model_comparison", exist_ok=True)
    os.makedirs("./images/ks_heatmaps", exist_ok=True)

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
                "Mean",
                "CI_95_Low",
                "CI_95_High",
                "Pct_10_15",
                "Pct_GT_25",
            ]
        )

    with open(averages_file, "w", newline="") as avg_stats_csvfile:
        writer = csv.writer(avg_stats_csvfile)
        writer.writerow(
            [
                "Dataset",
                "Method",
                "Shape Parameter (k)",
                "Scale Parameter (gamma)",
                "Num Data Points",
                "AIC",
                "BIC",
                "KS Statistic",
                "KS p-value",
                "Mean",
                "CI_95_Low",
                "CI_95_High",
                "Pct_10_15",
                "Pct_GT_25",
            ]
        )

    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    with open(ks_files, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Input File", "Month"] + month_names)

    for input_file, input_file_name in input_files:
        monthly_wind_speeds, monthly_dates = read_wind_data(f"./data/{input_file}.csv", args.drop_zeros)
        if monthly_wind_speeds is None:
            print(f"Failed to read wind speed data for {input_file_name} from ./data/{input_file}.csv")
            continue

        monthly_params = fit_weibull_to_monthly_wind_data(monthly_wind_speeds, input_file_name)

        plot_qq_plots(monthly_wind_speeds, monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)
        plot_pp_plots(monthly_wind_speeds, monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)
        plot_cdf(monthly_wind_speeds, monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)
        plot_percentile_stats(monthly_params, ["mle", "mm", "ls"], month_names, input_file_name)
        ks_matrix = plot_ks_statistics(monthly_wind_speeds, month_names, input_file_name)

        with open(ks_files, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for i, row_name in enumerate(month_names):
                row = [input_file_name, row_name]
                for j in range(len(month_names)):
                    row.append(f"{ks_matrix[i, j]:.4f}")
                writer.writerow(row)

        with open(stats_file, "a", newline="") as stats_csvfile:
            writer = csv.writer(stats_csvfile)
            for month, methods in monthly_params.items():
                wind_speeds = monthly_wind_speeds[month]
                for method, (k, gamma) in methods.items():
                    aic, bic, ks_stat, ks_pval = compute_fit_statistics(np.array(wind_speeds), k, gamma, method)
                    mean, CI_95_Low, CI_95_High, Pct_10_15, Pct_GT_25 = compute_wind_farm_statistics(k, gamma)
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
                            f"{mean:.5f}" if mean is not None else None,
                            f"{CI_95_Low:.5f}" if CI_95_Low is not None else None,
                            f"{CI_95_High:.5f}" if CI_95_High is not None else None,
                            f"{Pct_10_15:.5f}" if Pct_10_15 is not None else None,
                            f"{Pct_GT_25:.5f}" if Pct_GT_25 is not None else None,
                        ]
                    )

        method_totals = defaultdict(
            lambda: {
                "k_sum": 0.0,
                "gamma_sum": 0.0,
                "aic_sum": 0.0,
                "bic_sum": 0.0,
                "ks_stat_sum": 0.0,
                "ks_pval_sum": 0.0,
                "n_points": 0,
                "mean_sum": 0.0,
                "CI_95_Low_sum": 0.0,
                "CI_95_High_sum": 0.0,
                "Pct_10_15_sum": 0.0,
                "Pct_GT_25_sum": 0.0,
            }
        )

        for month, methods in monthly_params.items():
            wind_speeds = monthly_wind_speeds[month]
            for method, (k, gamma) in methods.items():
                aic, bic, ks_stat, ks_pval = compute_fit_statistics(np.array(wind_speeds), k, gamma, method)
                mean, CI_95_Low, CI_95_High, Pct_10_15, Pct_GT_25 = compute_wind_farm_statistics(k, gamma)
                n_points = len(wind_speeds)

                stats = method_totals[method]
                stats["k_sum"] += k if k is not None else 0
                stats["gamma_sum"] += gamma if gamma is not None else 0
                stats["aic_sum"] += aic if aic is not None else 0
                stats["bic_sum"] += bic if bic is not None else 0
                stats["ks_stat_sum"] += ks_stat if ks_stat is not None else 0
                stats["ks_pval_sum"] += ks_pval if ks_pval is not None else 0
                stats["n_points"] += n_points
                stats["mean_sum"] += mean
                stats["CI_95_Low_sum"] += CI_95_Low
                stats["CI_95_High_sum"] += CI_95_High
                stats["Pct_10_15_sum"] += Pct_10_15
                stats["Pct_GT_25_sum"] += Pct_GT_25

        with open(averages_file, "a", newline="") as avg_stats_csvfile:
            writer = csv.writer(avg_stats_csvfile)
            for method, stats in method_totals.items():
                count = 12
                writer.writerow(
                    [
                        input_file_name,
                        method,
                        f"{stats['k_sum']/count:.5f}" if count else None,
                        f"{stats['gamma_sum']/count:.5f}" if count else None,
                        stats["n_points"],
                        f"{stats['aic_sum']/count:.5f}" if count else None,
                        f"{stats['bic_sum']/count:.5f}" if count else None,
                        f"{stats['ks_stat_sum']/count:.5f}" if count else None,
                        f"{stats['ks_pval_sum']/count:.5f}" if count else "0.00000",
                        f"{stats['mean_sum']/count:.5f}" if count else None,
                        f"{stats['CI_95_Low_sum']/count:.5f}" if count else None,
                        f"{stats['CI_95_High_sum']/count:.5f}" if count else None,
                        f"{stats['Pct_10_15_sum']/count:.5f}" if count else None,
                        f"{stats['Pct_GT_25_sum']/count:.5f}" if count else None,
                    ]
                )

        print(f"Weibull parameters and stats for {input_file_name} saved.")
    plot_avg_ks_per_method(averages_file)
    plot_avg_params_per_station(averages_file)
    plot_monthly_wind_stats(stats_file)
    plot_avg_wind_stats(averages_file)
    plot_mean_with_ci_by_method(stats_file)
