import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import pandas as pd


K_INIT, GAMMA_INIT = 2, 10  # selected inital fit parameters based on intuition


def weibull_pdf(x, c, scale):
    """Weibull probability density function."""
    return weibull_min.pdf(x, c, scale=scale)


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
        ax.hist(wind_speeds, bins="auto", density=True, alpha=0.7, color="skyblue", label="Wind Speed Data")
        for method in fit_methods:
            try:
                if method in ["mle", "mm"]:
                    k, loc, gamma = weibull_min.fit(wind_speeds, floc=0, method=method)
                    params[month][method] = (k, gamma)
                elif method == "ls":
                    # Scipy doesnt have a built in method for least squares fitting so do it using curve_fi
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
    plt.savefig(f"./images/weibull_distributions_{input_file}.png", dpi=300)
    plt.show(block=False)

    return params


def read_wind_data(file_path):
    """
    Read wind speed data from a CSV file and group by month.
    """
    try:
        # Read CSV with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(file_path, usecols=["wdsp", "date"], low_memory=False)

        # Convert wind speeds to numeric, invalid values become NaN
        df["wdsp"] = pd.to_numeric(df["wdsp"], errors="coerce")

        # Drop any rows where wind speed is NaN
        df = df.dropna(subset=["wdsp", "date"])

        # Convert dates
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M")
        df["month"] = df["date"].dt.month

        # Group data by month
        monthly_speeds = {month: group["wdsp"].values for month, group in df.groupby("month")}
        monthly_dates = {month: group["date"].values for month, group in df.groupby("month")}
        # Count and print number of zero wind speeds
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

    with open(file_to_save_to, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Month", "Shape Parameter (k)", "Scale Parameter (gamma)", "Method", "Number of Data Points"])

    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    for input_file, input_file_name in input_files:
        monthly_wind_speeds, monthly_dates = read_wind_data(f"./data/{input_file}.csv")
        if monthly_wind_speeds is None:
            print(f"Failed to read wind speed data for {input_file_name} from ./data/{input_file}.csv")
            continue

        monthly_params = fit_weibull_to_monthly_wind_data(monthly_wind_speeds, input_file_name)

        # Append results for this input file
        with open(file_to_save_to, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for month, methods in monthly_params.items():
                for method, (k, gamma) in methods.items():
                    n_points = len(monthly_wind_speeds[month])
                    writer.writerow([input_file_name, month_names[month - 1], f"{k:.4f}", f"{gamma:.4f}", method, n_points])
        print(f"Weibull parameters for {input_file_name} saved to {file_to_save_to}")
