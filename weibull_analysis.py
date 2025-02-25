import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import pandas as pd


def fit_weibull_to_monthly_wind_data(monthly_wind_speeds):
    """
    Fit Weibull distributions to monthly wind speed data using multiple fitting methods and create visualization.

    Parameters:
    monthly_wind_speeds (dict): Dictionary with months as keys and wind speed arrays as values

    Returns:
    dict: Dictionary with months as keys and dict of fitting method parameters (k, gamma) as values
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fit_methods = ["mle", "mm"]  # Maximum likelihood and method of moments
    colors = {"mle": "red", "mm": "green"}
    line_styles = {"mle": "-", "mm": "--"}

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

        # Fit and plot for each method
        for method in fit_methods:
            try:
                k, loc, gamma = weibull_min.fit(wind_speeds, floc=0, method=method)
                params[month][method] = (k, gamma)

                # Plot Weibull PDF
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
    plt.show()

    return params


def read_wind_data(file_path):
    """
    Read wind speed data from a CSV file and group by month.

    Parameters:
    file_path (str): Path to the CSV file containing wind speed data

    Returns:
    dict: Dictionary with months (1-12) as keys and arrays of wind speeds as values
    dict: Dictionary with months (1-12) as keys and arrays of dates as values
    """
    try:
        df = pd.read_csv(file_path, usecols=["wdsp", "date"])
        df = df.dropna(subset=["wdsp", "date"])

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M")
        df["month"] = df["date"].dt.month

        monthly_speeds = {month: group["wdsp"].values for month, group in df.groupby("month")}
        monthly_dates = {month: group["date"].values for month, group in df.groupby("month")}

        return monthly_speeds, monthly_dates

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None


if __name__ == "__main__":
    monthly_wind_speeds, monthly_dates = read_wind_data("./hly532/hly532.csv")
    if monthly_wind_speeds is None:
        print("Failed to read wind speed data")
        exit()

    monthly_params = fit_weibull_to_monthly_wind_data(monthly_wind_speeds)

    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    with open("weibull_parameters.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Month", "Shape Parameter (k)", "Scale Parameter (gamma)", "Method", "Number of Data Points"])
        for month, methods in monthly_params.items():
            for method, (k, gamma) in methods.items():
                n_points = len(monthly_wind_speeds[month])
                writer.writerow([month_names[month - 1], f"{k:.4f}", f"{gamma:.4f}", method, n_points])
    print("Weibull parameters saved to weibull_parameters.csv")
