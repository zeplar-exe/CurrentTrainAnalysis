from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
import mne
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam 
import scipy
import matplotlib.pyplot as plt

mne.set_log_level("error")

# on motor imagery
# basic cnn with colony-selection vs deep cnn+L2
# csp + lda baseline

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.b1 = nn.BatchNorm1d(128)
        self.c2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.b2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(128, 8)
    
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.out(x)
        return x
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files_s1 = mne.datasets.eegbci.load_data(subjects=[1], runs=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
files_s2 = mne.datasets.eegbci.load_data(subjects=[2], runs=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
files_s3 = mne.datasets.eegbci.load_data(subjects=[3], runs=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
files_s4 = mne.datasets.eegbci.load_data(subjects=[4], runs=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

RUN_LABELS = {
    **{r: {"T0": "rest", "T1": "left_fist", "T2": "right_fist"} for r in (3, 7, 11)},
    **{r: {"T0": "rest", "T1": "imag_left_fist", "T2": "imag_right_fist"} for r in (4, 8, 12)},
    **{r: {"T0": "rest", "T1": "both_fists", "T2": "both_feet"} for r in (5, 9, 13)},
    **{r: {"T0": "rest", "T1": "imag_both_fists", "T2": "imag_both_feet"} for r in (6, 10, 14)},
}
label_set = ["left_fist", "right_fist", "imag_left_fist", "imag_right_fist", "both_fists", "both_feet", "imag_both_fists", "imag_both_feet"]

raw_cache = {}

TARGET_LEN = 256
BATCH_SIZE = 32


def preprocess_sample(sample, mean, std, target_len=TARGET_LEN):
    # Pad/crop to a fixed length so we can train in mini-batches.
    out = np.zeros((sample.shape[0], target_len), dtype=np.float32)
    length = min(sample.shape[1], target_len)
    out[:, :length] = sample[:, :length].astype(np.float32, copy=False)
    out = (out - mean[:, None]) / std[:, None]
    return out

def read_raw(path):
    if path in raw_cache:
        return raw_cache[path]

    raw = mne.io.read_raw_edf(path, verbose="error", preload=True)
    run = int(str(path)[-6:-4])  # S001R03.edf -> 3
    raw.annotations.rename(RUN_LABELS[run])
    raw.annotations.delete(np.flatnonzero(raw.annotations.description == "rest"))
    sfreq = raw.info["sfreq"]
    raw.notch_filter(freqs=60.0, fir_design='firwin', verbose="error")
    raw.filter(l_freq=8, h_freq=40, fir_design='firwin', n_jobs=4, verbose="error")   
    #raw.filter(l_freq=0.1, h_freq=min(sfreq / 2.0 - 1, 100), fir_design='firwin', n_jobs=4, verbose="error")    

    #ica = mne.preprocessing.ICA(n_components=0.995, method='fastica', verbose="error")
    #ica.fit(raw.copy(), verbose="error")
    #muscle_idx, scores = ica.find_bads_muscle(raw, verbose="error")
    #ica.exclude = muscle_idx
    #raw = ica.apply(raw.copy(), verbose="error")
    
    raw_cache[path] = raw
    return raw

X = defaultdict(list)
y = defaultdict(list)

for file in files_s1 + files_s2 + files_s3 + files_s4:
    raw = read_raw(file)
    sfreq = raw.info["sfreq"]
    
    base_phase = None
    
    for ann in raw.annotations:
        onset = ann["onset"]
        duration = ann["duration"]
        event = ann["description"]
        
        imin, imax = raw.time_as_index([onset, min(onset + duration, raw.times[-1])])
        cropped, times = raw[:, imin:imax]
        #analytic = np.abs(scipy.signal.hilbert(cropped))
        #if base_phase is None:
        #    base_phase = np.angle(analytic)
        #phase = np.angle(analytic)
        #phase_delta = np.angle(np.exp(1j * (phase - base_phase)))
        #aligned = np.real(analytic * np.exp(-1j * phase_delta))

        X[event].append(cropped)
        y[event].append([1 if x == event else 0 for x in label_set])

dataset = []
for event, samples in X.items():
    label_idx = label_set.index(event)
    for sample in samples:
        dataset.append((sample, label_idx))
print(Counter([label for _, label in dataset]))
del X, y
labels = [label for _, label in dataset]
train, test = train_test_split(dataset, test_size=0.2, random_state=42, stratify=labels)

train_padded = np.stack([
    np.pad(
        s[:, :TARGET_LEN].astype(np.float32, copy=False),
        ((0, 0), (0, max(0, TARGET_LEN - s.shape[1]))),
        mode='constant',
    )[:, :TARGET_LEN]
    for s, _ in train
], axis=0)
channel_mean = train_padded.mean(axis=(0, 2))
channel_std = train_padded.std(axis=(0, 2))
channel_std[channel_std < 1e-6] = 1.0
#single_colony = compute_colony()
#weights = single_colony.abs_weights()
#top_25 = np.quantile(weights, 0.75)

model = BasicCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # weight_decay for L2

for epoch in range(80):
    tot_loss = 0.0
    np.random.shuffle(train)
    for i in range(0, len(train), BATCH_SIZE):
        batch = train[i:i + BATCH_SIZE]
        xb = np.stack([preprocess_sample(x, channel_mean, channel_std) for x, _ in batch], axis=0)
        yb = np.array([y for _, y in batch], dtype=np.int64)

        x = torch.as_tensor(xb, dtype=torch.float32, device=device)
        y = torch.as_tensor(yb, dtype=torch.long, device=device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
    
    model.eval()
    epoch_pred_counts = Counter()
    with torch.no_grad():
        for x, _ in test:
            x = preprocess_sample(x, channel_mean, channel_std)
            x = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x).argmax(dim=1).item()
            epoch_pred_counts[pred] += 1

    print(
        f"Epoch [{epoch + 1}/{80}], Total Loss: {tot_loss:.4f}, "
        f"Avg Loss: {tot_loss / max(len(train), 1):.4f}, "
        f"Pred Counts: {dict(epoch_pred_counts)}"
    )
    model.train()

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for x, y in test:
        x = preprocess_sample(x, channel_mean, channel_std)
        x = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        res = model(x)
        pred = res.argmax(dim=1).item()
        y_true.append(y)
        y_pred.append(pred)

print("Final Pred Counts:", dict(Counter(y_pred)))

acc = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Final Accuracy: {acc:.4f}")
print(f"Final Precision (macro): {prec_macro:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_set))))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_set,
).plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.show()
