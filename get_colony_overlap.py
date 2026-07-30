import csv
import sys
import numpy as np
import pandas as pd

col_1 = sys.argv[1]
col_2 = sys.argv[2]

with open(col_1, 'r') as f1, open(col_2, 'r') as f2:
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    
    percentiles = [90, 80, 70, 60, 50]
    
    key = ["electrode"] if "inverse" not in col_1 else ["x", "y", "z"]
    
    print("Col 1 Descriptives:")
    print(df1["value"].describe())
    print("\nCol 2 Descriptives:")
    print(df2["value"].describe())
    
    for p in percentiles:
        df1_top = df1.nlargest(int(len(df1) * (100-p) / 100), 'value')
        df2_top = df2.nlargest(int(len(df2) * (100-p) / 100), 'value')
        
        overlap = pd.merge(df1_top, df2_top, on=key)
        print(f"{len(overlap)} overlapping at {p}th percentile out of {(len(df1) * (100-p) / 100)} ({len(overlap) / (len(df1) * (100-p) / 100) * 100:.2f}%)")
        

# how do we differentiate noise? for ex, whether gamma is real or just elsewhere
    # we should probably have a magnitude threshold such that gamma doesn't seem dominant simply due to having very little footprint (because we're not even looking at the variance or the min-max...)
# + also, what about negative colonies/accumulation? negative-or-zero style
# + gotta do ICA on EMG because of gamma
# how are we going to handle per-band decoding stuffs?
    # do a model for each individual band, then all combinant? plus whole as a separate?

# to detail with occipital overload:    
    # either: subtract raw_baseline... average? this is kinda stupid
    # or: go back to normalized (a-b)/b

