import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from lmfit.models import GaussianModel, Model
from lmfit.parameter import Parameters

from plotting_prefs import *

plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.Set1(np.linspace(0, 1, 9)))
# my choice of color palette :)

plt.ion()


"""
Decide whether or not to subtract background
"""

to_subtract = True
already_subtracted = True

# the column that will be used for plotting purposes will be adjusted accordingly
signal_col = ''
if to_subtract == False and already_subtracted == False:
	signal_col = 'Signal' # raw
else:
	signal_col = 'Signal_BGS'

"""
Load the data file(s)
"""
'''contents = os.listdir('../data/mar16/')
print(contents)
for f in contents:
	if ".csv" in f:
		print(f)'''

names_mar16 = ['0.5kV_475V_10.csv',
				'0.9kV_869V_2.csv',
				'0.7kV_671V_2.csv',
				'0.8kV_775V_2.csv',
				'0.6kV_573V_2.csv',
				'0.5kV_475V_2.csv',
				'0.5kV_475V_4.csv',
				'1kV_966V_2.csv',
				'0.5kV_475V_8.csv',
				'0.5kV_475V_6.csv']

data_cols = ['Arrival Time', 'Signal', 'Laser']
data_dir = '../data/mar16/'

file_name = names_mar16[9]


print('Reading:', file_name)
dataframe = pd.read_csv(data_dir + file_name #'mock_data.csv'
			, skiprows=2, names=data_cols)
dataframe['Arrival Time (mu s)'] = 1E6*dataframe['Arrival Time']
#print(dataframe.head())


def gauss_func(x, y0, A, mu, sigma):
	"""
	Define a Gaussian function, with a vertical offset y0 as an additional
	degree of freedom

	y0 is a vertical, zero point offset
	"""
	return y0 + A*np.exp(-(x-mu)**2/(sigma**2))


"""
Useful machinery for interactive Gaussian fitting
"""

class Region:
	"""
	Just to define an (x_min, x_max) range of the area of interest in which 
	all of the 'n' fitted Gaussians extend.
	"""
	def __init__(self, left=None, right=None):
		self.left = left
		self.right = right
		
	def is_ready(self):
		"""
		We're ready to fit in this region if both self.left and self.right
		are defined
		"""
		return (self.left != None) and (self.right != None)


def index_for_xval(x_val, df):
	# Idea in mind: subtract 'x_val' from the column containing x-coordinates
	# and get the index corresponding to that value.

	df['distance'] = np.abs(df['Arrival Time (mu s)'] - x_val)
	# Assuming that 'Arrival Time' is going to be the name of the column
	# we're searching 
	index = df['distance'].idxmin()

	#print(f"Value {x_val} has closest {df['Arrival Time (mu s)'][index]} index {index}")
	return index

def guess_sigma(x_mean, df):
	# so I want to find the dx for which the value becomes 1/2 of its
	# value at the "mean" location
	y_val = yval_for_x(x_mean, df)
	# index this position in the dataframe
	index = index_for_xval(x_mean, df)

	# if a y-value falls within this range (yes I chose it to be generously large)
	epsilon = y_val/20 # but still 10 times smaller than y_val/2
	i = index
	fwhm = 99.9
	# search for values on the right
	while (len(df) - i) > 0:
		x = df.iloc[i]['Arrival Time (mu s)']
		y = df.iloc[i][signal_col]
		# if we've hit "half maximum"
		if abs(y-(y_val/2)) < epsilon:
			fwhm = 2*abs(x-x_mean)
			break
		i += 1
	# now search to the left of the mean, in case a smaller FWHM is found
	# (this can be nice to do when there are overlapping peaks
	i = index
	while i >= 0:
		x = df.iloc[i]['Arrival Time (mu s)']
		y = df.iloc[i][signal_col]
		# if we've hit "half maximum"
		width = abs(y-y_val/2)
		if abs(y-(y_val/2)) < epsilon:
			width = 2*abs(x-x_mean)
			# if this is smaller, set this as FWHM
			fwhm = min(width, fwhm)
			break
		i -= 1
	# sigma is just
	sigma = fwhm/2.355
	
	return sigma


def yval_for_x(x_val, df):
	index = index_for_xval(x_val, dataframe)
	return df[signal_col][index]


def plot_fitted_result(xvals, yvals, final_result):
	global fig, axes
	axes[0].plot(xvals, final_result.best_fit, ls='-', lw=1,# c='red',
	      			zorder=2, label='composite model')
	residuals = 100*np.abs(yvals-final_result.best_fit)/yvals
	axes[1].plot(xvals, residuals, lw=1, c='tab:pink', label='Residuals')
	# also show the components separately
	components = final_result.eval_components(x=xvals)
	for i, comp_name in enumerate(components.keys()):
		#if comp_name != cont_offset:
		mu = final_result.params[comp_name + 'center'].value
		sigma = final_result.params[comp_name + 'sigma'].value
		print('data type', type(mu))
		axes[0].plot(xvals, components[comp_name], ls='--', lw=0.75,
				label=rf'$t_f={mu:.2f}\mu s,\ \sigma={sigma:.2f}$')
		#else:
		#	print('Continuum offset, y0 =', final_result.params['cont_offset'].value)
	axes[0].legend()
	fig.canvas.draw()


def cont_offset(x, y_offset):
	return x + y_offset

def perform_fit(df, init_models, region):
	models = []
	pars = Parameters()

	region_width = abs(region.right-region.left)# useful to know
	print(f'Selected region width: {region_width} micro sec')
	for i in range(len(init_models)):
		amplitude, mean, std = init_models[i]
		gauss = GaussianModel(prefix=f'g{i+1}_')
		pars.update(gauss.make_params())
		# initialize guess parameters of this model, bounding some of these
		# from above and below (min, max) is important to avoid non-sensical fits
		pars[f'g{i+1}_sigma'].set(value=std, min=std/5, max=std*1.5) # shouldn't be too off the initial guess
		pars[f'g{i+1}_center'].set(value=mean, min=mean-0.1*std, max=mean+0.1*std)#min=region.left, max=region.right)
		pars[f'g{i+1}_amplitude'].set(value=amplitude, min=0, # avoid absorption profiles
					max=10*amplitude) 
		# add this Gaussian to the total list of Gaussians
		models.append(gauss)
	# The following is a cheap solution to create a composite model
	# one does mod = g1 + g2 + .. + gn

	#print('**** Offset', offset_pars)
	#print("params", pars)
	composite_model = models[0]
	for i in range(1, len(models)):
		composite_model = composite_model + models[i]

	# IMPORTANT: continuum should be inferred from the neighbourhood of where you're plotting
	#y_offset = Model(cont_offset)
	#pars.update(y_offset.make_params())
	#pars['y_offset'].set(value=0) # assume y_offset is at 0

	composite_model = composite_model #+ y_offset

	# keeping track of this is useful for debugging
	# now fit the composite model
	left = index_for_xval(active_region.left, df)
	right = index_for_xval(active_region.right, df)
	region_yvals = df[signal_col][left:right]
	region_xvals = df['Arrival Time (mu s)'][left:right]
	initial_model = composite_model.eval(pars, x=region_xvals)
	

	result = composite_model.fit(region_yvals, pars,
			       x=region_xvals)
	print(result.fit_report())
	#print(len(result.best_fit))

	plot_fitted_result(region_xvals, region_yvals, result)
	#return region_xvals, initial_model, result


def fit_continuum(ext_regions, df, degree=0):
	"""
	ext_regions: an iterable/list of regions for which the continuum is to be set using
	"""
	# turn region (x_min, x_max) values to indices
	region_slices = []
	for region in ext_regions:
		min_i = index_for_xval(region.left, df)
		max_i = index_for_xval(region.right, df)
		region_slices.append(df.iloc[min_i:max_i])
	# gives a single dataframe
	full_ext_region = pd.concat(region_slices)

	coeffs = np.polyfit(full_ext_region['Arrival Time (mu s)'], full_ext_region['Signal'], deg=degree)
	# return the results of the fit
	return coeffs


def subtract_background(regions, dataframe, degree=0, show_preview=False):

	# fit a polynomial to the continuum region and obtain the coefficients
	poly_coeffs = fit_continuum(regions, dataframe, degree)
	print('fitted polynomial coeffs', poly_coeffs)
	# estimate the y-value that needs to be subtracted for each point based on the fit
	# initialize a diff column
	dataframe['Subtracted BG'] = 0.
	
	for i in range(len(poly_coeffs)):
		dataframe['Subtracted BG'] += poly_coeffs[i]* (dataframe['Arrival Time (mu s)']**i)
	dataframe['Signal_BGS'] = dataframe['Signal'] - dataframe['Subtracted BG']

	if show_preview == True:
		fig, ax = plt.subplots(figsize=(6,5))
		ax.plot(dataframe['Arrival Time (mu s)'], dataframe['Signal'], label='Raw Signal')
		ax.plot(dataframe['Arrival Time (mu s)'], dataframe['Signal_BGS'], label='After Subtraction')
		ax.plot(dataframe['Arrival Time (mu s)'], dataframe['Subtracted BG'], label='Background Offset')
		#ax.plot(dataframe['Arrival Time (mu s)'], (dataframe['Laser']-2)/2E3, c='dimgray',
		#				label='Laser Trigger', zorder=-1)
		ax.axhline(y=0, xmin=0, xmax=1, ls=':', c='dimgray', lw=0.75)
		ax.legend()
		#ax.set_xticklabels([])
		#ax.set_yticklabels([])
		ax.set_xlabel('Arrival time ($\mu$s)', fontsize=13)
		ax.set_ylabel('Signal (V)', fontsize=13)
		ax.tick_params(which='major', axis='both', direction='out', length=6)
		ax.minorticks_on()
		ax.tick_params(which='minor', axis='both', direction='out', length=4)

		plt.tight_layout()
		plt.subplots_adjust(wspace=0)
		#plt.savefig('bg_subtract.png')



# make a 2-panel plot (actual spectrum, residuals)
fig, axes = plt.subplots(2, 1, figsize=(6,8), sharex=True, height_ratios=[3,1])
print('did this get executed?')

# Region to use for background subtraction
regions = [Region(65.0, 86.2)]#Region(27.3, 36.9)]

if to_subtract == True:
	# use a zero/first degree polynomial for the background subtraction.
	subtract_background(regions, dataframe, degree=0, show_preview=False)

def plot_data():
	global fig, axes

	# plot signal
	axes[0].plot(dataframe['Arrival Time (mu s)'], dataframe[signal_col], c='darkgray',
			  				label=file_name)
	# plot
	#axes[0].plot(dataframe['Arrival Time (mu s)'], (dataframe['Laser']-2)/1E3, c='cornflowerblue',
	#					label='Laser Trigger', zorder=-1)

	axes[0].set_ylabel('Signal (V)')
	axes[1].set_xlabel('Arrival Time ($\mu$s)')
	axes[1].set_ylabel('Residuals [$\%$]')

	# compute residuals
	x = dataframe['Arrival Time (mu s)'][982:1030]
	#residual = dataframe[signal_col][982:1030] - gauss_func(x, *guess_bigp)

	# plot residuals
	#axes[1].plot(x, residual)
	for ax in axes:
		ax.axhline(y=0, xmin=0, xmax=1, c='dimgray', lw=0.75, ls='--', zorder=-1)

	# all the fancy things to adjust plot appearance/aesthetics.
	axes[0].legend()

plot_data()

def on_mouse_click(event):
	print(f'Hello, I know you {event.button} clicked at ({event.x}, {event.y}), i.e. ({event.xdata}, {event.ydata})')
	index_for_xval(event.xdata, dataframe)


# left and right limits of the "region" of the x-axis the fit should happen in
active_region = Region()

# need a list that holds
models_to_fit = []

def on_key_press(event):
	"""
	The action of pressing a certain key is defined here.
	Summary:
		Place your mouse cursor and press 'a' on the keybaord to define 
			the left and right limits of the area/region in which the fit is to be done.
		Press 'e'
	"""
	if event.key not in "advxy": # the three keys that should affect anything
		return
	else:
		global active_region
		if event.key == 'a':
			if active_region.left == None:
				active_region.left = event.xdata
				print('Lower limit set to', event.xdata)
			elif active_region.right == None:
				active_region.right = event.xdata
				print('Upper limit set to', event.xdata)
			else:
				active_region = Region()
				print('Cleared area selection. Press \'a\' again to start afresh')
		elif event.key == 'd':
			if active_region.is_ready():
				global models_to_fit
				# prepare to fit a Gaussian here
				print(f'Fitting Gaussian at {event.xdata}, choosing A={yval_for_x(event.xdata, dataframe)}')
				# initial guess: y0, A, \mu, \sigma
				amp = yval_for_x(event.xdata, dataframe)
				sigma = guess_sigma(event.xdata, dataframe)
				# amplitude, mean, std dev
				guess_param = (amp, event.xdata, sigma)
				models_to_fit.append(guess_param)
				print(models_to_fit)
			else: 
				print('No region defined. Please make sure you have set min and max values' +
		  			 	'of the region to fit in')
		elif event.key == 'v':
			x_pos = event.xdata
			global fig, axes
			print(f'Drawing a vertical line at x={x_pos}')
			axes[0].axvline(x=x_pos, ymin=0, ymax=1, ls=':', label=f'$t={x_pos:.2f}$ $\mu$s')
			axes[0].legend()
		elif event.key == 'x':
			print(f'Clearing fitted model near ({event.xdata}, {event.ydata})')
		elif event.key == 'y':
			if active_region.left == None or active_region.right == None:
				print("No region to fit defined, please press 'a' to specify left and right limits")
			if len(models_to_fit) > 0:
				print(f'Fitting {len(models_to_fit)} Gaussians')
				# performs a fit, plots it, and redraws the figure
				perform_fit(dataframe, models_to_fit, active_region)
			else:
				print('No models to fit')
			# clear "active" region
			active_region = Region()
			print('Cleared selected region')
			models_to_fit = [] # clear this now
		print(f'hihi, you pressed {event.key}')


def on_press_key(event):
	print(f'hello, did you just press \'{event.key}\' while your cursor was at ({event.xdata}, {event.ydata})')

fig.canvas.mpl_connect('button_press_event', on_mouse_click)
fig.canvas.mpl_connect('key_press_event', on_key_press)
#plt.savefig('test.png')
fig.tight_layout()
fig.subplots_adjust(hspace=0, bottom=0.09)
#plt.tight_layout()
plt.show(block=True)


cont_sub_file = data_dir + file_name[:-4] + '_bg_sub.csv'
dataframe.to_csv(cont_sub_file, sep=',', index=False)
print('Successfully dumped', cont_sub_file)