import numpy as np
import scipy
import matplotlib.pyplot as plt

S = 2.0
L = 10.0
sfreq = 100

data = np.sin(2 * np.pi * np.arange(0, L * sfreq))
data += np.cos(2 * np.pi * np.arange(0, L * sfreq))
data += np.sin(2/3 * np.pi * np.arange(0, L * sfreq))
data += np.sin(2/4 * np.pi * np.arange(0, L * sfreq))

a = np.abs(scipy.signal.hilbert(data))
b = scipy.ndimage.uniform_filter1d(a, size=int(S * sfreq))
data = (a-b)/b

plt.plot(data)
plt.show()