import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from sklearn.metrics import r2_score

sigmoid_model_code = """
data {
    int<lower=0> N;
    array[N] real log_epsilon;
    array[N] real accuracy;
}
parameters {
    real<lower=0, upper=1> L;
    real<lower=0> k;
    real c;
    real<lower=0> baseline;
    real<lower=0> sigma;
}
model {
    array[N] real mu;
    for (i in 1:N) {
        mu[i] = baseline + (L - baseline) / (1 + exp(-k * (log_epsilon[i] - c)));
    }
    L ~ beta(40, 2);
    k ~ lognormal(log(5), 1);
    c ~ normal(log(4), 2);
    baseline ~ beta(2, 10);
    sigma ~ gamma(2, 70);
    accuracy ~ normal(mu, sigma);
}
"""


def build_stan_model(log_epsilon, accuracy):
    data = {"N": len(log_epsilon), "log_epsilon": log_epsilon, "accuracy": accuracy}
    return stan.build(sigmoid_model_code, data=data, random_seed=42)


def sample_from_model(model, num_samples=6000, num_chains=4):
    fit = model.sample(
        num_chains=num_chains, num_samples=num_samples, num_warmup=num_samples // 2
    )
    return fit


def calculate_r_squared(
    L_samples, k_samples, c_samples, baseline_samples, log_epsilon, accuracy
):
    L_samples = np.squeeze(L_samples)  # (24000,)
    k_samples = np.squeeze(k_samples)  # (24000,)
    c_samples = np.squeeze(c_samples)  # (24000,)
    baseline_samples = np.squeeze(baseline_samples)  # (24000,)

    L_samples = L_samples[:, None]  # (24000, 1)
    k_samples = k_samples[:, None]  # (24000, 1)
    c_samples = c_samples[:, None]  # (24000, 1)
    baseline_samples = baseline_samples[:, None]  # (24000, 1)
    log_epsilon = log_epsilon[None, :]  # (1, num_points)

    # Calculate predictions
    predicted = baseline_samples + (L_samples - baseline_samples) / (
        1 + np.exp(-k_samples * (log_epsilon - c_samples))
    )
    predicted_mean = np.mean(predicted, axis=0)  # Shape: (num_points,)

    r_squared = r2_score(accuracy, predicted_mean)
    return r_squared


def plot_sigmoid_posterior(
    L_samples,
    k_samples,
    c_samples,
    baseline_samples,
    log_epsilon_observed,
    accuracy_observed,
    dataset_label,
    subset_label,
    output_dir,
):
    x_plot_original = np.logspace(np.log10(0.01), np.log10(16), 500)
    tick_values = [0.01, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    if dataset_label == "sun397":
        x_plot_continue = np.logspace(np.log10(16), np.log10(64), 100)[1:]
        x_plot_original = np.concatenate((x_plot_original, x_plot_continue))
        tick_values.append(32.0)
        tick_values.append(64.0)

    x_plot_log = np.log(x_plot_original)
    mean_vals = []
    lower_vals = []
    upper_vals = []

    for x in x_plot_log:
        curve = baseline_samples + (L_samples - baseline_samples) / (
            1 + np.exp(-k_samples * (x - c_samples))
        )
        mean_vals.append(np.mean(curve))
        lower_vals.append(np.percentile(curve, 2.5))
        upper_vals.append(np.percentile(curve, 97.5))

    r_squared = calculate_r_squared(
        L_samples,
        k_samples,
        c_samples,
        baseline_samples,
        log_epsilon_observed,
        accuracy_observed,
    )

    plt.figure()
    plt.scatter(
        np.exp(log_epsilon_observed), accuracy_observed, color="red", label="Observed"
    )
    plt.plot(x_plot_original, mean_vals, color="blue", label="Mean")
    plt.fill_between(
        x_plot_original, lower_vals, upper_vals, color="blue", alpha=0.2, label="95% CI"
    )
    plt.xscale("log")

    plt.xticks(tick_values, labels=[f"{tick:.3f}" for tick in tick_values], rotation=45)

    plt.ylim([0, 1])
    plt.xlabel("Epsilon (log scale)")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.title(f"{dataset_label} - {subset_label} (R² = {r_squared:.3f})")
    plt.grid()

    subset_dir = os.path.join(output_dir, subset_label)
    os.makedirs(subset_dir, exist_ok=True)

    plt.savefig(os.path.join(subset_dir, f"plot_{dataset_label}_{subset_label}.png"))
    plt.close()

    print(f"R^2 for {dataset_label} - {subset_label}: {r_squared:.3f}")


def get_every_nth_epsilon(log_eps, acc, step):
    indices = list(range(0, len(log_eps), step))

    if indices[-1] != len(log_eps) - 1:
        indices.append(len(log_eps) - 1)

    return log_eps[indices], acc[indices]


def main(output_dir="plots", csv_path="epsilon-accuracy.csv"):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    dataset_names = df["dataset_name"].unique()

    for ds in dataset_names:
        # Clean dataset name for filenames
        ds_label = ds.split("/")[-1]
        df_sub = df[df["dataset_name"] == ds].copy()
        df_sub.sort_values("target_epsilon", inplace=True)

        # Full data
        eps_all = df_sub["target_epsilon"].values
        log_eps_all = np.log(eps_all)
        acc_all = df_sub["accuracy"].values

        # Subsample every 10th
        log_eps_10, acc_10 = get_every_nth_epsilon(log_eps_all, acc_all, 10)

        # Subsample every 20th
        log_eps_20, acc_20 = get_every_nth_epsilon(log_eps_all, acc_all, 20)

        for subset_label, log_eps_array, acc_array in [
            ("all", log_eps_all, acc_all),
            ("every10th", log_eps_10, acc_10),
            ("every20th", log_eps_20, acc_20),
        ]:
            if len(log_eps_array) < 2:
                continue

            model = build_stan_model(log_eps_array, acc_array)
            fit = sample_from_model(model, num_samples=6000, num_chains=4)
            idata = az.from_pystan(posterior=fit)

            L_samples = fit["L"]
            k_samples = fit["k"]
            c_samples = fit["c"]
            baseline_samples = fit["baseline"]

            plot_sigmoid_posterior(
                L_samples,
                k_samples,
                c_samples,
                baseline_samples,
                log_eps_array,
                acc_array,
                ds_label,
                subset_label,
                output_dir,
            )

            subset_dir = os.path.join(output_dir, subset_label)
            os.makedirs(subset_dir, exist_ok=True)

            az_summary = az.summary(idata)
            txt_file = os.path.join(
                subset_dir, f"diagnostics_{ds_label}_{subset_label}.txt"
            )
            az_summary.to_csv(txt_file)


if __name__ == "__main__":
    main()
