"""
Just testing what the calculated standard deviation is when sampling from the 95th percentile.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def sample_from_95th_percentile(data, n_samples, print_info=False):
    """
    Sample n_samples from the 95th percentile of the data.
    """
    # Calculate the 95th percentile
    p975 = np.percentile(data, 97.5)
    p025 = np.percentile(data, 2.5)
    if print_info:
        print(f"p975: {p975}")
        print(f"p025: {p025}")

    tail_data_p975 = data[data >= p975]
    tail_data_p025 = data[data <= p025]
    tail_data = np.concatenate((tail_data_p975, tail_data_p025))

    if print_info:
        print(f"tail_data: {tail_data}")
    # Sample from the tail data
    samples = np.random.choice(tail_data, size=n_samples, replace=True)

    return samples, p975, p025


# data = np.random.normal(loc=0, scale=1, size=10000)
# n_samples = 10
# samples, p975, p025 = sample_from_95th_percentile(data, n_samples, print_info=True)
# print(f"Samples: {samples}")


# %%
def evaluate_sample(mu, sigma, n_samples):
    """
    Evaluate the sample by calculating the mean and standard deviation.
    """
    samples = np.random.normal(loc=mu, scale=sigma, size=n_samples * 100)
    tail_sample, p975, p025 = sample_from_95th_percentile(samples, n_samples)
    # print(tail_sample)
    mean = np.mean(tail_sample)
    std = np.std(tail_sample)
    # print(f"tail_smaple mean: {mean} vs actual mean: {mu}")
    # print(f"tail_smaple std: {std} vs actual std: {sigma}")

    return mean, std


if __name__ == "__main__":
    # Test the function with different parameters
    evaluate_sample(mu=0, sigma=1, n_samples=1000)
    evaluate_sample(mu=0, sigma=2, n_samples=1000)

    test_stds = np.linspace(0.01, 1000, 100)
    result_stds = []
    result_means = []
    for std in tqdm(test_stds):
        mean, std_m = evaluate_sample(mu=0, sigma=std, n_samples=100)
        result_stds.append(std_m)
        result_means.append(mean)
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(test_stds, result_stds, label="Sampled Std", marker="o")
    plt.plot(test_stds, result_means, label="Sampled Mean", marker="o")
    plt.axhline(y=0, color="r", linestyle="--", label="True Mean")
    plt.plot(test_stds, test_stds, label="True Std", ls="--", color="orange")
    plt.legend()
    plt.title("Sampled Mean and Std vs True Values")

    multiple_measured = [
        res_std / test_std for res_std, test_std in zip(result_stds, test_stds)
    ]
    print(multiple_measured)
    print(
        f"the measured std from the tailsample is {np.mean(multiple_measured)} times the true std"
    )
    # %%
    # test the same relationship with the number of samples
    test_samples = np.linspace(1, 1000, 100)
    result_stds = []
    result_means = []
    for n_samples in tqdm(test_samples):
        stds = []
        means = []
        for i in range(1):
            mean, std_m = evaluate_sample(mu=0, sigma=1, n_samples=int(n_samples))
            stds.append(std_m)
            means.append(mean)
        result_stds.append(np.mean(stds))
        result_means.append(np.mean(means))

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(test_samples, result_stds, label="Sampled Std", marker="o")
    plt.plot(test_samples, result_means, label="Sampled Mean", marker="o")
    plt.axhline(y=0, color="r", linestyle="--", label="True Mean")
    plt.plot(
        test_samples,
        np.ones_like(test_samples),
        label="True Std",
        ls="--",
        color="orange",
    )
    plt.legend()
    plt.title("Sampled Mean and Std vs True Values")
