# per-word hit / false-positive rates and accuracy vs training distribution,
# from the .npz score spectra saved by validate()
#   python analyze_scores.py pnpl/scores/after_run0_val_ses8.npz [more.npz ...]
# > Thanks Claude
import sys

import matplotlib.pyplot as plt
import numpy as np

TOP_K = 10

paths = sys.argv[1:]
if not paths:
    sys.exit("usage: python analyze_scores.py scores.npz [more.npz ...]")

data = [np.load(p) for p in paths]
labels = np.concatenate([d["label"] for d in data])

# training distribution from the last file (the latest checkpoint if files span several)
train_counts = {}
if "train_words" in data[-1]:
    train_counts = dict(zip(data[-1]["train_words"], data[-1]["train_counts"].tolist()))
else:
    print("(no training distribution in these files; skipping accuracy-vs-distribution)")

fig, axes = plt.subplots(2, 2, figsize=(13, 11), squeeze=False)
for row, vocab_name in enumerate(["primary", "moses"]):
    scores = np.concatenate([d[vocab_name] for d in data])  # (n, 50)
    vocab = data[0][f"{vocab_name}_vocab"]

    # only windows whose true word is in this vocab count toward accuracy
    in_vocab = np.isin(labels, vocab)
    scores, lab = scores[in_vocab], labels[in_vocab]
    n = len(lab)

    order = np.argsort(-scores, axis=1)
    in_top = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(in_top, order[:, :TOP_K], True, axis=1)
    true = lab[:, None] == vocab[None, :]                      # (n, 50)
    rank = np.argsort(order, axis=1) + 1                       # rank of every word, every window

    n_true = true.sum(0)
    hit_rate = np.divide((in_top & true).sum(0), n_true, out=np.full(len(vocab), np.nan), where=n_true > 0)
    fp_rate = (in_top & ~true).sum(0) / np.maximum((~true).sum(0), 1)
    top_rate = in_top.mean(0)                                  # how often in the top 10 at all
    n_train = np.array([train_counts.get(w, 0) for w in vocab])

    acc = (in_top & true).sum() / n
    const = np.isin(lab, vocab[np.argsort(-top_rate)[:TOP_K]]).mean()
    print(f"\n{vocab_name}: {n} windows, top-{TOP_K} accuracy {acc:.1%}")
    print(f"  if the {TOP_K} most-frequently-predicted words were always the answer: {const:.1%}")
    print(f"  {'word':>10} {'train':>5} {'n_true':>6} {'hit':>5} {'fp':>5} {'in_top':>6} "
          f"{'rank|true':>9} {'score|true':>10} {'score|else':>10}")
    for j in np.argsort(-top_rate):
        t = true[:, j]
        r_true = rank[t, j].mean() if t.any() else np.nan
        s_true = scores[t, j].mean() if t.any() else np.nan
        print(f"  {vocab[j]:>10} {n_train[j]:>5} {n_true[j]:>6} {hit_rate[j]:>5.2f} {fp_rate[j]:>5.2f} "
              f"{top_rate[j]:>6.2f} {r_true:>9.1f} {s_true:>10.3f} {scores[~t, j].mean():>10.3f}")

    has = n_true > 0
    sizes = 20 + 3 * n_true[has]

    # hit rate vs false-positive rate
    ax = axes[row, 0]
    ax.scatter(fp_rate[has], hit_rate[has], s=sizes, color="#2a78d6",
               alpha=0.7, edgecolors="white", linewidths=1)
    for j in np.where(has & ((n_true >= 10) | (fp_rate > 0.5)))[0]:
        ax.annotate(vocab[j], (fp_rate[j], hit_rate[j]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8, color="#52514e")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--")  # hit == fp: no better than ignoring the input
    ax.set_title(f"{vocab_name}: hit vs false positive (size = windows where word is true)", fontsize=10)
    ax.set_xlabel(f"false-positive rate (in top {TOP_K} when not the word)")
    ax.set_ylabel(f"hit rate (in top {TOP_K} when it is the word)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    # hit rate vs training distribution
    ax = axes[row, 1]
    if train_counts:
        x = n_train[has] + 0.5  # +0.5 so zero-count words survive log x
        ax.scatter(x, hit_rate[has], s=sizes, color="#2a78d6",
                   alpha=0.7, edgecolors="white", linewidths=1)
        for j, xj in zip(np.where(has)[0], x):
            if n_true[j] >= 5 and (hit_rate[j] > 0 or n_true[j] >= 15):  # skip the crowded zero row
                ax.annotate(vocab[j], (xj, hit_rate[j]), textcoords="offset points",
                            xytext=(5, 3), fontsize=8, color="#52514e")
        ax.set_xscale("log")
        ax.axhline(TOP_K / len(vocab), color="#52514e", lw=1, ls="--")
        ax.text(0.01, TOP_K / len(vocab) + 0.01, f"chance ({TOP_K}/{len(vocab)})", fontsize=8,
                color="#52514e", transform=ax.get_yaxis_transform())
    ax.set_title(f"{vocab_name}: hit rate vs training samples", fontsize=10)
    ax.set_xlabel("training samples (label distribution)")
    ax.set_ylim(-0.03, 1.03)

    for ax in axes[row]:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = "score_analysis.png"
fig.savefig(out, dpi=150)
print("\n->", out)
