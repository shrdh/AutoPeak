import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
import os
import matplotlib.cm as cm
import pandas as pd
from utils import preprocess_data
from itertools import groupby
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
import math
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
import peakutils

from scipy.optimize import curve_fit

tf.get_logger().setLevel('ERROR')

# folder_cycle1 = r'D:\Downloads\Sensor_1_ESR_cycling-20240213T204739Z-001\Sensor_1_ESR_cycling\sensor1_esr_temp_cycle_1'
folder_cycle1= r'D:\Downloads\sensor1_esr_temp_cycle_1'

# folder_cycle2 =  r'D:\Downloads\sensor_1_ESR_cycle_2'

#folder_cycle1= r'D:\Downloads\sensor1_esr_temp_cycle_1'
# folder_cycle2 =  r'D:\Downloads\sensor_1_ESR_cycle_2'
folder_cycle2= r'D:\Downloads\sensor_1_ESR_cycle_2-20240811T005415Z-001\sensor_1_ESR_cycle_2'




# Get a list of all the files in the folders (excluding the PARAMS file)
cycle1_files = os.listdir(folder_cycle1)
cycle1_files = [f for f in cycle1_files if "PARAMS" not in f]
cycle2_files = os.listdir(folder_cycle2)
cycle2_files = [f for f in cycle2_files if "PARAMS" not in f]

# Defining Parameters
s1 = np.array([[0.0,1.0,0.0],
    [1.0,0.0,1.0],
    [0.0,1.0,0.0]])

s2 = np.array([[0.0,-1.0j,0.0],
    [1.0j,0.0,-1.0j],
    [0.0,1.0j,0.0]])

s3 = np.array([[1.0,0.0,0.0],
    [0.0,0.0,0.0],
    [0.0,0.0,-1.0]])

spin1 = (1.0/np.sqrt(2.0))*s1
spin2 = (1.0/np.sqrt(2.0))*s2
spin3=s3


spin1 = tf.constant(spin1, dtype = 'complex128')
spin2 = tf.constant(spin2, dtype = 'complex128')
spin3 = tf.constant(spin3, dtype = 'complex128')

# a=tf.constant(-7.86851953723355e-05,dtype='float64')# Linear Regression # cycle 1
# b= tf.constant(2.870665858002803,dtype='float64') # Linear Regression

b= tf.constant( 2.87068615576284,dtype='float64') # Grad search cycle 2 
a=tf.constant(-7.723607188481802e-05, dtype='float64') # Grad search cycle 2 
c=tf.constant( -4.3478260869566193e-07,dtype='float64')
d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value

# a=tf.constant(-7.723606710665643e-05,dtype='float64') # (MLE cycle 1)
# b=tf.constant(2.8707465740453,dtype='float64') # (MLE cycle 1)



v = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_v)
w = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_w)

P_0 = tf.constant(1e-4, dtype = 'float64')
P = tf.constant(0.18, dtype = 'float64')
alpha= tf.constant(14.52e-3, dtype = 'float64')
I = tf.eye(3,dtype = 'complex128')
    
    
    
def getD(T, P):
    D = a * T + b + alpha * (P_0 - P_0)
    E = c * T + d + w
    return D, E

def H(D, E):
    Ham = tf.complex(D * (tf.math.real(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.real(spin1 @ spin1 - spin2 @ spin2)),
                    D * (tf.math.imag(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.imag(spin1 @ spin1 - spin2 @ spin2)))
    return Ham


@tf.autograph.experimental.do_not_convert
@tf.function
def getP_k(T, P):
    D, E = getD(T, P)
    Ham = H(D, E)
    eigenvalues = tf.linalg.eigvals(Ham)
    return eigenvalues


# @tf.function
# def bilorentzian(x, T, P):
#     eigenvalues = getP_k(T, P)
#     x0 = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
#     x01 = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)
#     x = tf.cast(x, tf.float64)
#     a=tf.cast(    47.95674298146268,tf.float64) # avg
#     gamma = tf.cast( 0.004293972996039488, tf.float64) # avg
#     return a * gamma**2 / ((x - x0)**2 + gamma**2) + a * gamma**2 / ((x - x01)**2 + gamma**2)


# def _get_vals(T, P):
#     timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
#     timespace = tf.cast(timespace, 'float64')
#     vals = bilorentzian(timespace, T, P)
#     return tf.reshape(vals, [N, 1])


# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
temperatures_cycle2 = [-30.0, -20.0, -10.0,  0.0, 10.0,  20.0, 30.0,  40.0, 50.0,40.0, 30.0,  20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
num_files_per_temp_cycle2 = 2
Frequency_cycle2 = None

peak1_locations = []
peak2_locations = [] 

# Process each group of 20 files
for i in range(0, len(cycle2_files), num_files_per_temp_cycle2):
    files_group_cycle2 = cycle2_files[i:i+num_files_per_temp_cycle2]
    temp_cycle2 = temperatures_cycle2[i//num_files_per_temp_cycle2]  # Get the corresponding temperature for this group
    T = tf.constant(temp_cycle2, dtype=tf.float64)
    ratios_cycle2 = np.array([])

   # Initialize a dictionary to store the peak locations
    
    for file in files_group_cycle2:
        data_cycle2 = pd.read_csv(os.path.join(folder_cycle2, file), delimiter=delimiter, header=None, names=variable_names)

        ratio_cycle2 = np.divide(data_cycle2['Intensity2'], data_cycle2['Intensity1'])
        if ratios_cycle2.size == 0:
            ratios_cycle2 = np.array([ratio_cycle2])
        else:
            ratios_cycle2 = np.vstack((ratios_cycle2, [ratio_cycle2]))  # Add ratio to the numpy array

    avg_intensity_cycle2 = np.mean(ratios_cycle2, axis=0)
    if Frequency_cycle2 is None:
        Frequency_cycle2 = data_cycle2['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz_cycle2 = Frequency_cycle2 / 1e9
        start_frequency_cycle2 = np.min(Frequency_cycle2)/1e9

    end_frequency_cycle2 = np.max(Frequency_cycle2)/1e9

    N = Frequency_cycle2.shape[0]
    dt = np.round((end_frequency_cycle2 - start_frequency_cycle2) / N, 4)

    timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
    # sim_val = _get_vals(T, P)
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean_cycle2 = np.mean(noise_sample_cycle2)
    avg_intensity_cycle2 = avg_intensity_cycle2 - noise_mean_cycle2
    # avg_intensity_cycle2 = np.max(sim_val)*( avg_intensity_cycle2)/(np.max(avg_intensity_cycle2))
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]]
    std_noise_cycle2=np.std(noise_sample_cycle2)


    y = avg_intensity_cycle2
    x = Frequency_GHz_cycle2
    original_x = x.copy()
    original_y = y.copy()

    
# Define a quadratic function
 
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

# Find the indices of the peaks
    peak_indices = peakutils.indexes(y, thres=0.5, min_dist=3)

    # Sort the peak indices by the y-values of the peaks
    peak_indices = sorted(peak_indices, key=lambda i: y[i], reverse=True)

    # Get the highest peak
    peak1_index = peak_indices[0]

    # Get the 3 points around the peak
    x_points1 = x[peak1_index-1:peak1_index+2]
    y_points1 = y[peak1_index-1:peak1_index+2]

    # Fit a quadratic to the points
    popt1 = np.polyfit(x_points1, y_points1, 2)

    # Get the maximum of the quadratic
    x_max1 = -popt1[1] / (2 * popt1[0])
    # print(f'Peak 1 is at {x_max1}')
    print(f'Peak 2 for group {i//num_files_per_temp_cycle2} is at {x_max1}')
    # Zero out points around the first peak
    #y[peak1_index-1:peak1_index+2] = 0
    # y[peak1_index-2:peak1_index+3] = 0 # zero out 5 points around the peak
    y[peak1_index-2:peak1_index+3] = 0 #zero out 4 points around the peak

    # Find the indices of the peaks again
    peak_indices = peakutils.indexes(y, thres=0.5, min_dist=3)

    # Sort the peak indices by the y-values of the peaks
    peak_indices = sorted(peak_indices, key=lambda i: y[i], reverse=True)

    # Get the highest peak
    peak2_index = peak_indices[0]

    # Get the 3 points around the second peak
    x_points2 = x[peak2_index-1:peak2_index+2]
    y_points2 = y[peak2_index-1:peak2_index+2]

    # Fit a quadratic to the points
    popt2 = np.polyfit(x_points2, y_points2, 2)

    # Get the maximum of the quadratic
    x_max2 = -popt2[1] / (2 * popt2[0])
    peak1_locations.append(x_max1)
    peak2_locations.append(x_max2)
    # print(f'Peak 2 is at {x_max2}')
    print(f'Peak 2 for group {i//num_files_per_temp_cycle2} is at {x_max2}')
    # plt.figure()
    # plt.plot(original_x, original_y, label='Data')
    # plt.plot([x_max1, x_max2], [np.polyval(popt1, x_max1), np.polyval(popt2, x_max2)], 'ro', label='Peak centers')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
    
peak1_locations = np.array(peak1_locations)
peak2_locations = np.array(peak2_locations)
training_data = np.stack((peak1_locations, peak2_locations), axis=-1)

all_temperatures = []
all_roots = []
# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
Frequency = None 
num_files_per_temp = 20
temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10]

peak1_locations_cycle1 = []
peak2_locations_cycle1 = []


temperatures = temperatures[1:]


# Process each group of 20 files in cycle 1
for i in range(num_files_per_temp, len(cycle1_files), num_files_per_temp):
    files_group = cycle1_files[i:i+num_files_per_temp]
    temp = temperatures[(i//num_files_per_temp)-1]   # Get the corresponding temperature for this group
    T = tf.constant(temp, dtype=tf.float64)
    ratios = np.array([])

    for file in files_group:
        data = pd.read_csv(os.path.join(folder_cycle1, file), delimiter=delimiter, header=None, names=variable_names)

        ratio = np.divide(data['Intensity2'], data['Intensity1'])
        if ratios.size == 0:
            ratios = np.array([ratio])
        else:
            ratios = np.vstack((ratios, [ratio]))  # Add ratio to the numpy array

    avg_intensity = np.mean(ratios, axis=0)
    if Frequency is None:
        Frequency = data['Frequency'].values
        Frequency_GHz = Frequency / 1e9
        start_frequency = np.min(Frequency)/1e9

    end_frequency = np.max(Frequency)/1e9
    N = Frequency.shape[0]
    dt = np.round((end_frequency - start_frequency) / N, 4)
    timespace = np.linspace(start_frequency, end_frequency, num=N)
    # sim_val = _get_vals(T, P)
    noise_sample = avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean = np.mean(noise_sample)
    avg_intensity = avg_intensity - noise_mean
    noise_sample = avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
    std_noise = np.std(noise_sample)

    y_cycle1 = avg_intensity
    X_cycle1 = Frequency_GHz
    original_x_cycle1 = X_cycle1.copy()
    original_y_cycle1 = y_cycle1.copy()
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c


# Find the indices of the peaks
    peak_indices_cycle1 = peakutils.indexes(y_cycle1, thres=0.5, min_dist=3)

    # Sort the peak indices by the y-values of the peaks
    peak_indices_cycle1 = sorted(peak_indices_cycle1, key=lambda i: y_cycle1[i], reverse=True)

    # Get the highest peak
    peak1_index_cycle1 = peak_indices_cycle1[0]

    # Get the 3 points around the peak
    x_points1_cycle1 = X_cycle1[peak1_index_cycle1-1:peak1_index_cycle1+2]
    y_points1_cycle1 = y_cycle1[peak1_index_cycle1-1:peak1_index_cycle1+2]

    # Fit a quadratic to the points
    popt1_cycle1 = np.polyfit(x_points1_cycle1, y_points1_cycle1, 2)

    # Get the maximum of the quadratic
    x_max1_cycle1 = -popt1_cycle1[1] / (2 * popt1_cycle1[0])
    print(f'Peak 2 for group {i//num_files_per_temp} is at {x_max1_cycle1}')
    # Zero out points around the first peak
    # y_cycle1[peak1_index-1:peak1_index+2] = 0 # zero out 3 points around the peak
    
    # y_cycle1[peak1_index-2:peak1_index+3] = 0
    y_cycle1[peak1_index_cycle1-2:peak1_index_cycle1+3] = 0 # zero in 4 points around the peak

    # Find the indices of the peaks again
    peak_indices_cycle1 = peakutils.indexes(y_cycle1, thres=0.5, min_dist=3)

    # Sort the peak indices by the y-values of the peaks
    peak_indices_cycle1 = sorted(peak_indices_cycle1, key=lambda i: y_cycle1[i], reverse=True)

    # Get the highest peak
    peak2_index_cycle1 = peak_indices_cycle1[0]

    # Get the 3 points around the second peak
    x_points2_cycle1 = X_cycle1[peak2_index_cycle1-1:peak2_index_cycle1+2]
    y_points2_cycle1 = y_cycle1[peak2_index_cycle1-1:peak2_index_cycle1+2]

    # Fit a quadratic to the points
    popt2_cycle1 = np.polyfit(x_points2_cycle1, y_points2_cycle1, 2)

    # Get the maximum of the quadratic
    x_max2_cycle1 = -popt2_cycle1[1] / (2 * popt2_cycle1[0])
    peak1_locations_cycle1.append(x_max1_cycle1)
    peak2_locations_cycle1.append(x_max2_cycle1)

    print(f'Peak 2 for group {i//num_files_per_temp} is at {x_max2_cycle1}')
    # plt.figure(figsize=(6,4 ))
    # plt.plot(original_x_cycle1, original_y_cycle1, label='Data')
    # plt.plot([x_max1_cycle1, x_max2_cycle1], [np.polyval(popt1_cycle1, x_max1_cycle1), np.polyval(popt2_cycle1, x_max2_cycle1)], 'ro', label='Peaks')
    # plt.xticks(fontsize=12)  # Set the font size of x values
    # plt.yticks(fontsize=12)
    # plt.xlabel('Frequency (GHz)', fontsize=12)
    # plt.ylabel('Intensity(arb. Units)', fontsize=12)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\Peak\\Peakcenters.png',dpi=300)
peak1_locations_cycle1 = np.array(peak1_locations_cycle1)
peak2_locations_cycle1 = np.array(peak2_locations_cycle1)
testing_data = np.stack((peak1_locations_cycle1, peak2_locations_cycle1), axis=-1)
    
    

# Stack the x0_values and x01_values horizontally to create the feature matrix for cycle 2


# Create a LinearRegression object
reg = LinearRegression()

# Fit the model to the data
reg.fit(training_data, temperatures_cycle2) # calculates the coefficients that minimize the squared difference between the model's predictions and the actual target values

# Stack the x0_values_cycle1 and x01_values_cycle1 horizontally to create the feature matrix for cycle 1


# Predict the temperatures for cycle 1
temperatures_cycle1_predicted = reg.predict(testing_data)

print(f"Predicted temperatures for cycle 1: {temperatures_cycle1_predicted}")

testing_error = root_mean_squared_error(temperatures, temperatures_cycle1_predicted)

print(f'Testing error: {testing_error}')
# Predict the temperatures for the training data
temperatures_cycle2_predicted = reg.predict(training_data)

# Calculate the training error
training_error =root_mean_squared_error(temperatures_cycle2, temperatures_cycle2_predicted)

print(f'Training error: {training_error}')


# # Testing
kelvin_temperatures = [temp + 273.15 for temp in temperatures]

temperatures_cycle1_predicted_kelvin = [root + 273.15 for root in  temperatures_cycle1_predicted]
coefficients = np.polyfit(kelvin_temperatures, temperatures_cycle1_predicted_kelvin, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(kelvin_temperatures), max(kelvin_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(kelvin_temperatures)
# Your predicted values
predicted =temperatures_cycle1_predicted_kelvin

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted,)
# Plot the data points
plt.plot(kelvin_temperatures, temperatures_cycle1_predicted_kelvin , 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(kelvin_temperatures), max(temperatures_cycle1_predicted_kelvin) * 0.5, 'R-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()

# Save the figure
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\peakutilis_testing.png', dpi=300) 
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()




# Create a figure
plt.figure(figsize=(6,4 ))

# Fit a linear polynomial
kelvin_temperatures = [temp + 273.15 for temp in temperatures]

temperatures_cycle1_predicted_kelvin = [root + 273.15 for root in  temperatures_cycle1_predicted]
coefficients1 = np.polyfit(kelvin_temperatures, temperatures_cycle1_predicted_kelvin, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(kelvin_temperatures), max(kelvin_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(kelvin_temperatures)
# Your predicted values
predicted =temperatures_cycle1_predicted_kelvin

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(kelvin_temperatures, temperatures_cycle1_predicted_kelvin , 'o',color='green')
plt.plot(x_values, y_values, '-',color='red')

# # Plot the linear fit in red
# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(kelvin_temperatures, temperatures_cycle1_predicted_kelvin, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(kelvin_temperatures)
# predicted =temperatures_cycle1_predicted_kelvin
# r_squared_poly2 = r2_score(true_poly2, predicted)
# from sklearn.metrics import r2_score, mean_squared_error
# rmse_poly2 = root_mean_squared_error(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# # Add text boxes for the linear and quadratic fits
# plt.text(min(temperatures), max(predicted) * 0.2, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(temperatures)*0.489, max(predicted)*0.1, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))


plt.text(min(kelvin_temperatures) , max(temperatures_cycle1_predicted_kelvin) -20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(kelvin_temperatures)-15, max(temperatures_cycle1_predicted_kelvin)-30, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# # Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\Peak\\testing_cycle_1_linear.png',dpi=300)

# Show the plot
plt.show()



# Training
# Create a figure
plt.figure(figsize=(6,4 ))


temperatures_cycle2_kelvin = [temp + 273.15 for temp in  temperatures_cycle2]
temperatures_cycle2_predicted_kelvin = [root + 273.15 for root in  temperatures_cycle2_predicted]
coefficients = np.polyfit(temperatures_cycle2_kelvin, temperatures_cycle2_predicted_kelvin, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2_kelvin), max(temperatures_cycle2_kelvin), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2_kelvin)
# Your predicted values
predicted = temperatures_cycle2_predicted_kelvin
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
plt.plot(temperatures_cycle2_kelvin, temperatures_cycle2_predicted_kelvin , 'o',color='green')
plt.plot(x_values, y_values, '-',color='red')
# Plot the linear fit in red
# Fit a quadratic polynomial
# coefficients2 = np.polyfit(temperatures_cycle2_kelvin, temperatures_cycle2_predicted_kelvin, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(temperatures_cycle2_kelvin)
# predicted =temperatures_cycle2_predicted_kelvin
# r_squared_poly2 = r2_score(true_poly2, predicted)
# from sklearn.metrics import r2_score, mean_squared_error
# rmse_poly2 = root_mean_squared_error(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# # Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# # Add text boxes for the linear and quadratic fits
# plt.text(min(temperatures), max(predicted) * 0.2, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(temperatures)*0.489, max(predicted)*0.1, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))


plt.text(min(temperatures_cycle2_kelvin) , max(temperatures_cycle2_predicted_kelvin) -40, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(temperatures_cycle2_kelvin)-30, max(temperatures_cycle2_predicted_kelvin)-55, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\Peak\\training_cycle_2_linear.png',dpi=300)
# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\training_2_peak_utilis.png', dpi=300) 

# Show the plot
plt.show()
# # # For Training


temperatures_cycle2_kelvin = [temp + 273.15 for temp in  temperatures_cycle2]
temperatures_cycle2_predicted_kelvin = [root + 273.15 for root in  temperatures_cycle2_predicted]
coefficients = np.polyfit(temperatures_cycle2_kelvin, temperatures_cycle2_predicted_kelvin, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2_kelvin), max(temperatures_cycle2_kelvin), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2_kelvin)
# Your predicted values
predicted = temperatures_cycle2_predicted_kelvin
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures_cycle2_kelvin, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.text(min(temperatures_cycle2_kelvin), max(predicted) * 0.5, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()



# # # QUADRATIC FIT for training
coefficients = np.polyfit(temperatures_cycle2, temperatures_cycle2_predicted , 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted =temperatures_cycle2_predicted
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures_cycle2, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.text(min(temperatures_cycle2), max(predicted) * 0.5, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # Quadratic Fit for testing
coefficients = np.polyfit(temperatures, temperatures_cycle1_predicted, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures), max(temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures)
# Your predicted values
predicted = temperatures_cycle1_predicted
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(temperatures, temperatures_cycle1_predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')

plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()


# N