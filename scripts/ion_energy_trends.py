"""
trends with ionization energy
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

contents = os.listdir('../data/mar16/')

file_names = []
other_names = []
for f in contents:
    if '.5kV' in f and ".csv" in f:
        file_names.append(f)
    elif "bg_sub.csv" in f:
        other_names.append(f)

other_names = sorted(other_names, reverse=True)
dataframes = []
for f in other_names + [file_names[-1]]:
    df = pd.read_csv('../data/mar16/' + f, skiprows=2, names=['Arrival Time', 'Signal', 'Trigger'])
    dataframes.append(df)

energies = [float(name.split('kV')[0])*1000 for name in other_names+[file_names[-1]]]

extraction_voltages = [float(name.split('_')[1][:-1]) for name in other_names+[file_names[-1]]]
print('extraction voltages are', extraction_voltages)
flight_length = 0.69 # m, or 69 cm

def time_to_mq(df, E_i):
    e = 1.6e-19 # C
    m_p = 1.67e-27 # kg
    global flight_length # use the flight length that was fixed
    m_by_q = 2*E_i*(df['Arrival Time'])**2/(flight_length**2)
    mq_natural = m_by_q*(e/m_p)# convert into natural units (not sure if the right word)
    return mq_natural

colors = ['crimson', 'gold', 'olive', 'tab:green', 'cornflowerblue', 'tab:pink']
fig, axes = plt.subplots(len(dataframes), 1, figsize=(6,8), sharex=True)

print(file_names)
to_plot = True
if to_plot == True:
    for df, energy, ax, file_name, color in zip(
                    dataframes, extraction_voltages, axes, other_names + [file_names[0]], colors):
        mass_to_charge = time_to_mq(df, energy)
        ax.plot(mass_to_charge, df['Signal BGS'], label=f"{round(1E3*float(file_name.split('kV')[0]))} V",
                    c=color, lw=1.0)
        ax.legend()
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=6, direction='out')
        ax.tick_params(axis='both', which='minor', length=3, direction='in')
        ax.set_xlim(left=0, right=900)

    axes[5].set_xlabel('Reconstructed $m/q$')
    axes[2].set_ylabel('Signal (V)')
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('../figures/ion_energy_trends.png')
plt.show()
