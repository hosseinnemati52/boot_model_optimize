#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:01:37 2025

@author: hossein
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, poisson, chi2, gamma


# Function to fit an exponential distribution and plot the data
def fit_and_plot_exponential(data, file_name):
    # Remove any potential outliers (optional step, can be skipped if not needed)
    # data = remove_outliers(data)

    # Fit an exponential distribution to the data
    loc, scale = expon.fit(data)  # 'loc' is the offset, 'scale' is 1/lambda for exponential distribution

    # Print the fitted parameters
    print(f"Fitted Exponential Distribution Parameters:\nLocation (shift): {loc}, Scale (1/lambda): {scale}")

    # Plot the histogram for the data
    plt.hist(data, bins=30, alpha=0.6, color='b', edgecolor='black', density=True, label='Histogram')

    # Generate x values for plotting the exponential distribution
    x_values = np.linspace(min(data), max(data), 1000)

    # Plot the fitted exponential distribution
    plt.plot(x_values, expon.pdf(x_values, loc=loc, scale=scale), color='r', label='Fitted Exponential Distribution')

    # Show the plot
    plt.title('Data Distribution with Fitted Exponential Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(file_name, dpi=300)
    plt.show()

    return loc, scale

def fit_and_plot_chi_square(data, file_name):
    # Fit a Chi-Square distribution to the data
    df, loc, scale = chi2.fit(data)  # 'df' is degrees of freedom, 'loc' is the offset, 'scale' is the scale parameter

    # Print the fitted parameters
    print(f"Fitted Chi-Square Distribution Parameters:\nDegrees of Freedom (df): {df}, Location (shift): {loc}, Scale: {scale}")

    # Plot the histogram for the data
    plt.hist(data, bins=30, alpha=0.6, color='b', edgecolor='black', density=True, label='Histogram')

    # Generate x values for plotting the Chi-Square distribution
    x_values = np.linspace(min(data), max(data), 1000)

    # Plot the fitted Chi-Square distribution
    plt.plot(x_values, chi2.pdf(x_values, df, loc=loc, scale=scale), color='r', label='Fitted Chi-Square Distribution')

    # Show the plot
    plt.title('Data Distribution with Fitted Chi-Square Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(file_name, dpi=300)
    plt.show()

    return df, loc, scale

def fit_and_plot_gamma(data, file_name):
    # Fit a Gamma distribution to the data
    shape, loc, scale = gamma.fit(data)  # 'shape' is the shape parameter, 'loc' is the offset, 'scale' is the scale parameter

    # Print the fitted parameters
    print(f"Fitted Gamma Distribution Parameters:\nShape: {shape}, Location (shift): {loc}, Scale: {scale}")

    # Plot the histogram for the data
    plt.hist(data, bins=30, alpha=0.6, color='b', edgecolor='black', density=True, label='Histogram')

    # Generate x values for plotting the Gamma distribution
    x_values = np.linspace(min(data), max(data), 1000)

    # Plot the fitted Gamma distribution
    plt.plot(x_values, gamma.pdf(x_values, shape, loc=loc, scale=scale), color='r', label='Fitted Gamma Distribution')

    # Show the plot
    plt.title('Data Distribution with Fitted Gamma Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(file_name, dpi=300)
    plt.show()

    return shape, loc, scale

def fit_and_plot_poisson(data):
    data = np.round(data)  # Poisson distribution is discrete, round the data
    mean_val = np.mean(data)  # Mean is used as the parameter for Poisson (`lambda`)
    print(f"Poisson Fit - Lambda: {mean_val}")

    plt.hist(data, bins=30, alpha=0.6, color='b', edgecolor='black', density=True, label='Histogram')
    x_values = np.arange(min(data), max(data) + 1)
    plt.plot(x_values, poisson.pmf(x_values, mu=mean_val), color='g', label='Poisson Fit')
    plt.title('Data with Fitted Poisson Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Mass')
    plt.legend()
    plt.show()

    return mean_val

# Function to generate samples from the fitted exponential distribution
def generate_exponential_samples(loc, scale, sample_size, data, file_name):
    """
    Generates integer samples from the exponential distribution with the given parameters.

    Parameters:
    loc (float): Location parameter of the exponential distribution (minimum value).
    scale (float): Scale parameter of the exponential distribution (1/lambda).
    sample_size (int): Number of samples to generate.

    Returns:
    np.array: Array of integer samples from the exponential distribution.
    """
    # Generate samples from the exponential distribution
    samples = expon.rvs(loc=loc, scale=scale, size=sample_size)
    samples = samples[(samples <= np.max(data)) & (samples >= np.min(data))]
    # Round the samples to the nearest integer
    integer_samples = np.round(samples).astype(int)
    # integer_samples = samples
    
    # random_choices = np.random.rand(len(samples)) < 0.5
    # # Apply floor or ceil based on the random choices
    # integer_samples = np.where(random_choices, np.floor(samples), np.ceil(samples)).astype(int)
    
    bins = np.linspace(np.min(data), np.max(data), 31)
    
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.7, color='b', density=True, label='Real'+' ('+str(len(data))+')')
    plt.hist(integer_samples, bins=bins, alpha=0.4, color='r', edgecolor='k', density=True, label='Synthetic'+' ('+str(len(integer_samples))+')')
    
    x_values = np.linspace(min(data), max(data), 1000)
    # Plot the fitted exponential distribution
    plt.plot(x_values, expon.pdf(x_values, loc=loc, scale=scale), color='r', label='Fitted Exponential Distribution')

    # Show the plot
    plt.title('Data Distribution with Fitted Exponential Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(file_name, dpi=300)
    plt.show()
        
    return integer_samples

def generate_gamma_samples(shape, loc, scale, sample_size, data, file_name):
    """
    Generates integer samples from the exponential distribution with the given parameters.

    Parameters:
    loc (float): Location parameter of the exponential distribution (minimum value).
    scale (float): Scale parameter of the exponential distribution (1/lambda).
    sample_size (int): Number of samples to generate.

    Returns:
    np.array: Array of integer samples from the exponential distribution.
    """
    # Generate samples from the exponential distribution
    samples = gamma.rvs(a=shape, loc=loc, scale=scale, size=sample_size)
    
    samples = samples[(samples <= np.max(data)) & (samples >= np.min(data))]
    # Round the samples to the nearest integer
    integer_samples = np.round(samples).astype(int)
    # integer_samples = samples
    
    # random_choices = np.random.rand(len(samples)) < 0.5
    # # Apply floor or ceil based on the random choices
    # integer_samples = np.where(random_choices, np.floor(samples), np.ceil(samples)).astype(int)
    
    bins = np.linspace(np.min(data), np.max(data), 31)
    
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.7, color='b', density=True, label='Real'+' ('+str(len(data))+')')
    plt.hist(integer_samples, bins=bins, alpha=0.4, color='r', edgecolor='k', density=True, label='Synthetic'+' ('+str(len(integer_samples))+')')
    
    x_values = np.linspace(min(data), max(data), 1000)
    # Plot the fitted exponential distribution
    plt.plot(x_values, gamma.pdf(x_values, shape, loc=loc, scale=scale), color='r', label='Fitted Gamma Distribution')

    # Show the plot
    plt.title('Data Distribution with Fitted Gamma Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.savefig(file_name, dpi=300)
    plt.show()
        
    return integer_samples

# Function to remove outliers using the IQR method (optional)
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def remove_outliers_2d(x_data, y_data):
    """
    Removes outliers from 2D data using the IQR method.

    Parameters:
    x_data (array-like): Data for the x-axis.
    y_data (array-like): Data for the y-axis.

    Returns:
    tuple: Filtered x_data and y_data without outliers.
    """
    def filter_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        return (data >= lower_bound) & (data <= upper_bound)

    # Apply the filter to both x_data and y_data
    x_filter = filter_outliers(x_data)
    y_filter = filter_outliers(y_data)
    combined_filter = x_filter & y_filter

    return x_data[combined_filter], y_data[combined_filter]

def plot_2d_histogram(x_data, y_data, bins=30, cmap='viridis'):
    """
    Plots a 2D histogram with color mapping after removing outliers.

    Parameters:
    x_data (array-like): Data for the x-axis.
    y_data (array-like): Data for the y-axis.
    bins (int or tuple): Number of bins for the histogram. Can be an integer or a tuple for x and y bins.
    cmap (str): Colormap to be used for the 2D histogram.

    Returns:
    None
    """
    # Remove outliers from the data
    x_data_filtered, y_data_filtered = remove_outliers_2d(x_data, y_data)

    # Create the 2D histogram
    plt.hist2d(x_data_filtered, y_data_filtered, bins=bins, cmap=cmap)

    # Add a colorbar to show the intensity of the bins
    plt.colorbar(label='Counts')

    # Label the axes
    plt.xlabel('X-axis Data')
    plt.ylabel('Y-axis Data')

    # Set a title for the plot
    plt.title('2D Histogram with Color Mapping (Outliers Removed)')

    # Display the plot
    plt.show()
    
# Example dataset
# Corrected data (comma-separated)

N_size_max = 100000

data_pure_WT = np.array([14, 52, 9, 9, 8, 16, 13, 25, 61, 18, 54, 25, 22, 19, 21, 38, 10, 8, 14, 21, 9, 21, 41, 29, 27, 13, 15, 57, 11, 19, 97, 129, 97, 169, 141, 57, 204, 30, 186, 45, 14, 21, 53, 18, 33, 82, 55, 127, 77, 136, 14, 30, 37, 41, 52, 39, 32, 91, 47, 22, 123, 24, 90, 35, 74, 44])
loc, scale = fit_and_plot_exponential(data_pure_WT, "WT_pure.PNG")
generated_samples = generate_exponential_samples(loc, scale, N_size_max, data_pure_WT, "WT_pure_synthetic.PNG")
np.savetxt("WT_sample_bank.csv", X=generated_samples, fmt="%d", delimiter=',')



data_pure_C = np.array([60, 146, 15, 112, 36, 48, 42, 100, 27, 144, 21, 34, 72, 25, 85, 42, 28, 77, 31, 22, 116, 10, 39, 73, 86, 93, 13, 62, 65, 56, 33, 21, 46, 49, 71, 20, 115, 18, 20, 88, 66, 22, 56, 26, 54])
shape, loc, scale = fit_and_plot_gamma(data_pure_C, "C_pure.PNG")
generated_samples = generate_gamma_samples(shape, loc, scale, N_size_max, data_pure_C, "C_pure_synthetic.PNG")
np.savetxt("C_sample_bank.csv", X=generated_samples, fmt="%d", delimiter=',')






# data_mix_WT = np.array([52, 61, 48, 24, 40, 22, 20, 192, 32, 44, 56, 44, 29, 38, 272, 15, 33, 38, 28, 17, 23, 130, 17, 311, 223, 141, 56, 50, 74, 167, 74, 132, 14, 37, 15, 43, 31, 79, 105, 94, 85, 36, 70, 27, 60, 90])
data_mix_WT = np.array([52, 61, 48, 24, 40, 22, 20, 192, 32, 44, 56, 44, 29, 38, 15, 33, 38, 28, 17, 23, 130, 17, 311, 223, 141, 56, 50, 74, 167, 74, 132, 14, 37, 15, 43, 31, 79, 105, 94, 85, 36, 70, 27, 60, 90]) # 272 was deleted (liked to 2516 in C)
loc, scale = fit_and_plot_exponential(data_mix_WT, "WT_mix.PNG")
generated_samples_WT = generate_exponential_samples(loc, scale, N_size_max, data_mix_WT, "WT_mix_synthetic.PNG")


# data_mix_C = np.array([8, 9, 2, 71, 19, 10, 13, 125, 148, 3, 12, 33, 11, 25, 2516, 9, 34, 22, 7, 2, 23, 8, 18, 184, 98, 17, 13, 97, 14, 241, 56, 61, 25, 56, 26, 13, 8, 33, 8, 85, 78, 8, 11, 26, 18, 20])
data_mix_C = np.array([8, 9, 2, 71, 19, 10, 13, 125, 148, 3, 12, 33, 11, 25, 9, 34, 22, 7, 2, 23, 8, 18, 184, 98, 17, 13, 97, 14, 241, 56, 61, 25, 56, 26, 13, 8, 33, 8, 85, 78, 8, 11, 26, 18, 20])
 # 2516 was deleted (liked to 272 in WT)
loc, scale = fit_and_plot_exponential(data_mix_C, "C_mix.PNG")
generated_samples_C = generate_exponential_samples(loc, scale, N_size_max, data_mix_C, "C_mix_synthetic.PNG")


length = min(len(generated_samples_WT), len(generated_samples_C))
sample_bank = np.zeros((length,2))
sample_bank[:,0] = generated_samples_WT[:length]
sample_bank[:,1] = generated_samples_C[:length]
np.savetxt("mixed_sample_bank.csv", X=sample_bank, fmt="%d", delimiter=',')

kjhskjhdsfbgjh


plt.scatter(data_mix_WT, data_mix_C)



# Calculate the ratio of WT to C in the mixed data
mix_WT_to_C_ratio = data_mix_WT / data_mix_C
plt.figure()
plt.hist(mix_WT_to_C_ratio, bins=30)


data = data_mix_WT
min_val = 20
max_val = 200

# Fit and plot the exponential distribution
loc, scale = fit_and_plot_exponential(data)
# loc, scale = fit_and_plot_poisson(data)

plot_2d_histogram(data_mix_WT, data_mix_C)
# Generate new samples with the same distribution
sample_size = 10000  # Adjust sample size as needed
generated_samples = generate_exponential_samples(loc, scale, sample_size)

sample_bank_mix_WT = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]
# sample_bank = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]

# sample_bank = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]
# np.savetxt("mixed_sample_bank.csv", X=sample_bank, fmt="%d", delimiter=',')


data = data_mix_C
min_val = 5
max_val = 200

# Fit and plot the exponential distribution
loc, scale = fit_and_plot_exponential(data)
# loc, scale = fit_and_plot_poisson(data)

plot_2d_histogram(data_mix_WT, data_mix_C)
# Generate new samples with the same distribution
sample_size = 10000  # Adjust sample size as needed
generated_samples = generate_exponential_samples(loc, scale, sample_size)

sample_bank_mix_C = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]
# sample_bank = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]

# sample_bank = generated_samples[(generated_samples <= max_val) & (generated_samples >= min_val)]
# np.savetxt("mixed_sample_bank.csv", X=sample_bank, fmt="%d", delimiter=',')

length = min(len(sample_bank_mix_WT), len(sample_bank_mix_C))
sample_bank = np.zeros((length,2))
sample_bank[:,0] = sample_bank_mix_WT[:length]
sample_bank[:,1] = sample_bank_mix_C[:length]
np.savetxt("mixed_sample_bank.csv", X=sample_bank, fmt="%d", delimiter=',')


plot_2d_histogram(sample_bank[:,0], sample_bank[:,1])


# Plot the generated samples
plt.hist(generated_samples, bins=30, alpha=0.6, color='g', edgecolor='black', density=True)
plt.title(f'Generated Data Distribution (Sample Size = {sample_size})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()





