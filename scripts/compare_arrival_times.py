import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lmfit import Model

from plotting_prefs import *

# assume flight length
L = 0.69 # 69 cm

line_arrival_times = pd.read_csv('../figures/line_positions/new/positions_new.csv',
                                sep=',', skiprows=1)

print(line_arrival_times.head())

#print(line_arrival_times.columns)
global fig, ax
fig, ax = plt.subplots(figsize=(6,6))


def linear_func(x, slope, delay):
    return x*slope + delay

"""
Idea for estimating m/q

if y = E_i,
and x = 1/2 (L/t_f)^2
then the slope of this line should be (m/q)

"""

'''for line in line_arrival_times.columns[1:] :# first column is energy, ignore that. Rest are line positions
    # for some lines, there isn't a measured value, which pandas fills with "NaN"s.
    # drop rows containing those to avoid numerical problems
    data = line_arrival_times[~line_arrival_times[line].isna()]

    x_vals = (1/2)*(L/(data[line]*1E-6))**2
    y_vals = 0.64*data['Energy']*(1.66E-19) # eV -> J

    coeffs = np.polyfit(x=x_vals, y=y_vals, deg=1) # fit a straight line
    mass_to_charge = coeffs[0]/(1.67E-27)
    ax.plot(x_vals, y_vals, marker='o', mec='dimgray', mew=0.5, lw=1.,
            label=f'$m/q = {mass_to_charge:.2f}$')#marker='D', ls='-', c='dimgray', mfc='deeppink')'''


all_energies = [500, 600, 700, 800, 900, 1000]
for line in line_arrival_times.columns[1:] :# first column is energy, ignore that. Rest are line positions
    # for some lines, there isn't a measured value, which pandas fills with "NaN"s.
    # drop rows containing those to avoid numerical problems
    data = line_arrival_times[~line_arrival_times[line].isna()]
    # the energies for which this line has been measured 
    energies = np.array([all_energies[i] for i in data[line].index])
    x_vals = L**2/(2*energies)
    arrival_times_sq = (data[line]*1E-6)**2
    
    # provide an initial guess for the fit, assume there's no delay
    mq_guess = arrival_times_sq/x_vals
    # the returned value is a list, one val for each energy, take the average
    mass_guess = np.average(mq_guess)
    # standard deviation among these
    mass_guess_std = np.std(mq_guess)
    #print('Guess mq,', mq_guess*1E8)
    print(f'===Average m/q: {mass_guess*1E8}, sigma:{mass_guess_std*1E8}')

    #print(arrival_times_sq)
    #print(x_vals)
    lin_model = Model(linear_func)
    params = lin_model.make_params(slope=mass_guess, delay=0)
    result = lin_model.fit(arrival_times_sq, params, x=x_vals)
    #print(result.fit_report())
    #fit_coeff, cov_matrix = curve_fit(linear_func, x_vals, arrival_times_sq)
    #coeffs = np.polyfit(x=x_vals, y=arrival_times_sq, deg=1)
    mass_to_charge = result.params['slope']*1E8 # 
    error = result.params['slope'].stderr*1E8
    print(f"*** Fit result: m/q is {mass_to_charge:.2f} +/- {error:.2f}")
    print(f"***Delay time estimate: {np.sqrt(result.params['delay'])*1E6:.2f} us, with error" +
          f"+/- {np.sqrt(result.params['delay'].stderr)*1E6:.2f}")
    #del_t2 = result.eval_uncertainty(sigma=1)
    #print(coeffs)
    ax.plot(x_vals, arrival_times_sq, marker='o', mec='dimgray', mew=0.5, lw=1.,
           label=f'$m/q = {mass_to_charge:.2f}\pm{error:.2f}$')
    #print(data[line].index)
#for line in line_arrival_times

ax.tick_params(axis='both', labelsize=13)
ax.set_xlabel(r"$L^2/2U$ [m$^2$ J C$^{-1}$]", fontsize=13)
ax.set_ylabel('$t^2$ [s$^2$]', fontsize=13)
ax.legend(loc='upper right', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/mass_to_charge_slope_estimates.png', dpi=300)
plt.show()