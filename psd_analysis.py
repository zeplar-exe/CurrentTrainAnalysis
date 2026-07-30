from analyze import get_subject_data, RECORDS
import mne
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


raw, events = get_subject_data(RECORDS[0])
sfreq = int(raw.info["sfreq"])
nyq = int(sfreq / 2)
ch_names = raw.ch_names
raw.notch_filter(60)
max_freq = min(nyq, 100) - 1
raw.filter(1, max_freq)

sec, ts = raw.get_data(tmin=0.0, tmax=15.0, picks=[ch_names[0]], return_times=True)
sec = sec.flatten()

stft = ShortTimeFFT(hann(sfreq), 1, raw.info["sfreq"], scale_to="psd")
spec = stft.spectrogram(sec)

power_magnitude = np.median([-np.floor(np.log10(x)) for x in spec.flatten() if x != 0.0])
spec = spec * 10**power_magnitude

freq_means = np.mean(spec, axis=1, keepdims=True)
freq_stds = np.std(spec, axis=1, keepdims=True)

spec_z = (spec - freq_means) / freq_stds

plt.imshow(spec_z, origin='lower', aspect='auto', cmap='viridis')
plt.ylabel('Frequency bin')
plt.xlabel('Time slice')
plt.show()

# for this one... best bet is to use a convolutional network because of the sheer amount of features we're working with
    # input: 1 second of data, 160 (320?) samples, 64 channels, up to 100 frequency bins, maybe with attention baked in?
    # actually, no, what??? only take the seconds of data that are either mostly or entirely events, so we can do this supervised
        #  hwo does the model handle multiclass probabilities exatly?
