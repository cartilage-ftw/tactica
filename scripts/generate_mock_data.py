
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.linspace(0, 40, 500)
t_secs = t/1E6

y = 0.05*np.random.uniform(-0.5, 0.5, size=500) # some noise

def gauss_func(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma))

y += gauss_func(t, 1, 16.6, 1)
y += gauss_func(t, 0.3, 19.8, 1)

plt.plot(t, y)

trigger = np.zeros(500) # doesn't matter

df = pd.DataFrame(data={'Arrival Time':t_secs, 'Signal':y, 'Trigger':trigger})
df.to_csv('mock_data.csv', index=False)
plt.show()