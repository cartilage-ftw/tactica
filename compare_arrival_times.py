import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

# assume flight length
L = 0.69 # 69 cm

line_arrival_times = pd.read_csv('../figures/line_positions/positions_summary.tsv',
                                sep='\t', skiprows=1)

print(line_arrival_times.head())

#print(line_arrival_times.columns)

fig, ax = plt.subplots(figsize=(6,6))

"""
Idea for estimating m/q

if y = E_i,
and x = 1/2 (L/t_f)^2
then the slope of this line should be (m/q)

"""

for line in line_arrival_times.columns[1:] :# first column is energy, ignore that. Rest are line positions
    # for some lines, there isn't a measured value, which pandas fills with "NaN"s.
    # drop rows containing those to avoid numerical problems
    data = line_arrival_times[~line_arrival_times[line].isna()]

    x_vals = (1/2)*(L/(data[line]*1E-6))**2
    y_vals = 0.64*data['Energy']*(1.66E-19) # eV -> J

    coeffs = np.polyfit(x=x_vals, y=y_vals, deg=1) # fit a straight line
    mass_to_charge = coeffs[0]/(1.67E-27)
    ax.plot(x_vals, y_vals, marker='o', label=f'$m/q = {mass_to_charge:.2f}$')#marker='D', ls='-', c='dimgray', mfc='deeppink')

ax.set_xlabel(r"$L^2/2t_f^2$ [m$^2$ s$^{-2}$]")
ax.set_ylabel('Energy [J]')
ax.legend()
plt.show()