import sys
import scipy
import pandas as pd
import os

d = sys.argv[1]

for f in sorted(os.listdir(d)):
    with open(os.path.join(d, f), 'r') as file:
        df = pd.read_csv(file)
        skew = scipy.stats.skew(df['value'])
        print(f"{f}: {skew}, mean: {df['value'].mean()}, median: {df['value'].median()} ({'right skewed' if skew > 0 else 'left skewed' if skew < 0 else 'symmetric'})")