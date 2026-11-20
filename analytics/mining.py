"""Mining layer for the Music & Mental Health survey (736 listeners)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

GENRES = ["Classical","Country","EDM","Folk","Gospel","Hip hop","Jazz","K pop","Latin","Lofi","Metal","Pop","R&B","Rap","Rock","Video game music"]
FREQ = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Very frequently": 3}
MH = ["Anxiety", "Depression", "Insomnia", "OCD"]
YN = ["While working", "Instrumentalist", "Composer", "Exploratory", "Foreign languages"]

def prep(raw):
    """Clean the raw survey: fix outliers, encode frequencies, derive features."""
    d = raw.drop(columns=[c for c in ["Timestamp", "Permissions"] if c in raw]).copy()
    d.columns = [c.replace("Frequency [", "").replace("]", "") for c in d.columns]
    d["Age"] = d["Age"].fillna(d["Age"].median())
    d["BPM"] = d["BPM"].where(d["BPM"].between(40, 250))        # source has values like 999999999
    d["Primary streaming service"] = (d["Primary streaming service"].fillna("Unknown")
        .replace({"I do not use a streaming service.": "No streaming service"}))
    for g in GENRES: d[g] = d[g].map(FREQ).fillna(0)
    for c in YN: d[c + "_n"] = d[c].map({"Yes": 1, "No": 0}).fillna(0)
    d["Age group"] = pd.cut(d["Age"], [0, 17, 24, 34, 44, 200], labels=["Under 18", "18-24", "25-34", "35-44", "45+"])
    d["Wellbeing load"] = d[MH].mean(axis=1)
    d["Improves"] = d["Music effects"].map({"Improve": 1, "No effect": 0, "Worsen": 0})
    return d.reset_index(drop=True)

def segments(d, k):
    """K-Means on genre habits, listening hours and self-rated conditions."""
    X = d[GENRES + MH + ["Hours per day"]].copy(); X["Hours per day"] = X["Hours per day"].clip(upper=16)
    Z = StandardScaler().fit_transform(X)
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    lab = pd.Series(km.labels_, index=d.index)
    names, used = {}, set()
    for c in range(k):
        m = d.loc[lab == c, GENRES].mean() - d[GENRES].mean()
        n = f"{m.idxmax()}-led"
        while n in used: n += "+"
        used.add(n); names[c] = n
    return lab.map(names), PCA(2, random_state=0).fit_transform(Z), silhouette_score(Z, km.labels_)

def silhouette_scan(d, kmax=7):
    X = d[GENRES + MH + ["Hours per day"]].copy(); X["Hours per day"] = X["Hours per day"].clip(upper=16)
    Z = StandardScaler().fit_transform(X)
    return pd.DataFrame([(k, silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z))) for k in range(2, kmax + 1)],
                        columns=["k", "silhouette"])

def genre_wellbeing_corr(d):
    """Spearman correlation between how often a genre is played and each condition."""
    return pd.DataFrame({m: d[GENRES].corrwith(d[m], method="spearman") for m in MH})

def drivers(d):
    """Random Forest: what separates listeners who say music improves their mood?"""
    m = d.dropna(subset=["Improves"]).copy(); m["BPM"] = m["BPM"].fillna(m["BPM"].median())
    f = GENRES + MH + ["Hours per day", "Age", "BPM"] + [c + "_n" for c in YN]
    y = m["Improves"].astype(int)
    rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=3, random_state=0)
    auc = cross_val_score(rf, m[f], y, cv=5, scoring="roc_auc").mean()
    imp = pd.Series(rf.fit(m[f], y).feature_importances_, index=[c.replace("_n", "") for c in f]).sort_values()
    return imp, auc, y.mean()

def insights(d, seg):
    out = []
    fav = d["Fav genre"].value_counts(normalize=True)
    out.append(f"**{fav.index[0]}** is the most common favourite genre ({fav.iloc[0]:.0%}), followed by **{fav.index[1]}** ({fav.iloc[1]:.0%}).")
    g = d.dropna(subset=["Improves"]).groupby("Fav genre")["Improves"].agg(["mean", "size"]).query("size >= 20").sort_values("mean", ascending=False)
    if len(g): out.append(f"Fans of **{g.index[0]}** are the most likely to say music improves their mood ({g['mean'].iloc[0]:.0%}). Lead with it in wellbeing-focused playlists.")
    c = genre_wellbeing_corr(d).stack(); a, b = c.abs().idxmax()
    out.append(f"The strongest genre link to a condition is **{a}** with **{b.lower()}** (Spearman {c[(a,b)]:+.2f}). It is a correlation, not proof of cause.")
    s = d.assign(seg=seg).groupby("seg")["Wellbeing load"].mean().sort_values(ascending=False)
    out.append(f"The **{s.index[0]}** segment reports the highest average condition score ({s.iloc[0]:.1f}/10). Treat that audience with extra care in messaging.")
    w = d["While working"].eq("Yes").mean()
    out.append(f"**{w:.0%}** of listeners play music while working, so focus-friendly and instrumental releases have a real audience.")
    sv = d["Primary streaming service"].value_counts(normalize=True)
    out.append(f"**{sv.index[0]}** is the primary service for {sv.iloc[0]:.0%} of respondents. Prioritise its playlists and pitching tools.")
    return out
