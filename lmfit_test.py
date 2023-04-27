import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExponentialModel, GaussianModel


dat = np.loadtxt('../data/test/Gauss2.dat', skiprows=61)
x = dat[:, 1]
y = dat[:, 0]

exp_mod = ExponentialModel(prefix='exp_')
pars = exp_mod.guess(y, x=x)

gauss1 = GaussianModel(prefix='g1_')
pars.update(gauss1.make_params())

print('\n**->now with updates**\n')
for p in pars:
    print(p)

'''
pars['g1_sigma'].set(value=15, min=3)'''
pars['g1_center'].set(value=105, min=75, max=125)
pars['g1_amplitude'].set(value=2000, min=10)

print('**\nwhat did those manual changes do?\n')
for p in pars:
    print(pars['g1_amplitude'])
    print(p)
    #print(dir(p))
    #print(type(p))
    break

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())

pars['g2_center'].set(value=155)#, min=125, max=175)
pars['g2_sigma'].set(value=15)#, min=3)
pars['g2_amplitude'].set(value=2000)#, min=10)

mod = gauss1 + gauss2 + exp_mod
print('type of mod', type(mod))
init = mod.eval(pars, x=x)
out = mod.fit(y, pars, x=x)

#print(dir(out.params))
print(out.params)
print(type(out.params))
#print(out.fit_report(min_correl=0.5))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
axes[0].plot(x, y)
axes[0].plot(x, init, '--', label='initial fit')
axes[0].plot(x, out.best_fit, '-', label='best fit')
axes[0].legend()

comps = out.eval_components(x=x)
axes[1].plot(x, y)
axes[1].plot(x, comps['g1_'], '--', label='Gaussian component 1')
axes[1].plot(x, comps['g2_'], '--', label='Gaussian component 2')
axes[1].plot(x, comps['exp_'], '--', label='Exponential component')
axes[1].legend()

plt.show()