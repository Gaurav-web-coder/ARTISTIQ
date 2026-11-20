import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from database import db
from analytics import mining
from analytics.mining import GENRES, MH

st.set_page_config("ARTISTIQ", page_icon="◐", layout="wide")
INK, PANEL, LINE, TEXT, MUTED = "#0F1A1C", "#162427", "#2B3C40", "#E8EEEA", "#8FA3A0"
PAL = ["#E9B44C", "#6FC2B0", "#D9778B", "#8FA7E8", "#C98B5E", "#B7C77A"]
SCALE = [PANEL, "#2F6B66", PAL[1], PAL[0]]

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'DM Sans',sans-serif; }}
h1,h2,h3 {{ font-family:'Bricolage Grotesque',sans-serif !important; font-weight:700 !important; letter-spacing:-.02em; }}
h1 {{ font-size:2.6rem !important; margin-bottom:.2rem; }}
#MainMenu, footer {{ visibility:hidden; }} header[data-testid=stHeader] {{ background:transparent; }}
[data-testid=stSidebar] {{ border-right:1px solid {LINE}; }}
.brand {{ font-family:'Bricolage Grotesque'; font-size:1.7rem; font-weight:700; letter-spacing:-.03em; }}
.brand i {{ color:#E9B44C; font-style:normal; }}
.lede {{ color:{MUTED}; max-width:62ch; margin-bottom:1.6rem; }}
.band {{ display:flex; border-top:1px solid {LINE}; border-bottom:1px solid {LINE}; margin:.5rem 0 1.8rem; }}
.band div {{ flex:1; padding:1rem 1.2rem; border-left:1px solid {LINE}; }}
.band div:first-child {{ border-left:none; padding-left:0; }}
.band b {{ display:block; font-family:'Bricolage Grotesque'; font-size:2rem; font-weight:700; }}
.band span {{ color:{MUTED}; font-size:.85rem; }}
.note {{ border-left:3px solid #E9B44C; padding:.5rem 1rem; margin:.7rem 0; background:{PANEL}; border-radius:0 8px 8px 0; }}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Reading survey...")
def load(): return mining.prep(db.load("survey"))

@st.cache_data(show_spinner="Training model...")
def drivers(d): return mining.drivers(d)

ALL = load()
st.sidebar.markdown('<div class="brand">Artist<i>IQ</i></div>', unsafe_allow_html=True)
PAGE = st.sidebar.radio("Navigate", ["Overview", "Genres", "Wellbeing", "Listener segments", "What it means", "Survey data"], label_visibility="collapsed")
st.sidebar.divider(); st.sidebar.caption("Filter every page")
lo, hi = st.sidebar.slider("Age", int(ALL.Age.min()), int(ALL.Age.max()), (int(ALL.Age.min()), 60))
svc = st.sidebar.multiselect("Streaming service", sorted(ALL["Primary streaming service"].unique()))
df = ALL[ALL.Age.between(lo, hi) & (ALL["Primary streaming service"].isin(svc) if svc else True)].reset_index(drop=True)
st.sidebar.caption(f"{len(df)} of {len(ALL)} respondents")
if len(df) < 60: st.warning("Fewer than 60 respondents match. Widen the filters for reliable results."); st.stop()

def style(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=PAL,
        font=dict(family="DM Sans", color=MUTED, size=12), margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12, title=None))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, title=None); fig.update_yaxes(gridcolor=LINE, zeroline=False, title=None)
    return fig

def show(fig):
    try: st.plotly_chart(fig, width="stretch")
    except TypeError: st.plotly_chart(fig, use_container_width=True)

def band(items): st.markdown('<div class="band">' + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in items) + "</div>", unsafe_allow_html=True)
def head(t, l): st.title(t); st.markdown(f'<p class="lede">{l}</p>', unsafe_allow_html=True)
def heat(m, h=420): return style(px.imshow(m, text_auto=".1f", aspect="auto", color_continuous_scale=SCALE).update_layout(coloraxis_showscale=False), h)

def overview():
    head("Who is listening", f"A survey of {len(ALL)} listeners: what they play, on which service, and how music affects them.")
    band([("Respondents", len(df)), ("Median age", int(df.Age.median())), ("Hours of music a day", f"{df['Hours per day'].mean():.1f}"),
          ("Say music improves their mood", f"{df.Improves.mean():.0%}")])
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Streaming service")
        s = df["Primary streaming service"].value_counts().sort_values()
        show(style(px.bar(s, orientation="h", color_discrete_sequence=[PAL[1]])))
    with b:
        st.subheader("Favourite genre")
        g = df["Fav genre"].value_counts().sort_values().tail(10)
        show(style(px.bar(g, orientation="h", color_discrete_sequence=[PAL[0]])))
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Age groups")
        show(style(px.histogram(df, x="Age group", color_discrete_sequence=[PAL[3]]).update_xaxes(categoryorder="array", categoryarray=["Under 18","18-24","25-34","35-44","45+"]), 280))
    with b:
        st.subheader("Hours of music a day")
        show(style(px.histogram(df[df["Hours per day"] <= 12], x="Hours per day", nbins=13, color_discrete_sequence=[PAL[2]]), 280))

def genres():
    head("Genre habits", "How often listeners play each genre, and who their favourites are.")
    st.subheader("How often each genre is played")
    lab = {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Very frequently"}
    f = pd.DataFrame({g: df[g].map(lab).value_counts(normalize=True) for g in GENRES}).T.reindex(columns=list(lab.values())).fillna(0)
    f = f.sort_values("Very frequently")
    fig = px.bar(f, orientation="h", barmode="stack", color_discrete_sequence=[LINE, "#2F6B66", PAL[1], PAL[0]])
    fig.update_xaxes(tickformat=".0%"); show(style(fig, 520))
    st.subheader("Average condition score by favourite genre")
    c = df.groupby("Fav genre")[MH].agg("mean"); n = df["Fav genre"].value_counts()
    show(heat(c[n[c.index] >= 10].sort_values("Anxiety", ascending=False), 460)); st.caption("Scores run 0 to 10. Genres with fewer than 10 fans are hidden.")

def wellbeing():
    head("Music and wellbeing", "Self-rated anxiety, depression, insomnia and OCD, and what listeners say music does for them.")
    e = df["Music effects"].value_counts(normalize=True)
    band([("Improve", f"{e.get('Improve', 0):.0%}"), ("No effect", f"{e.get('No effect', 0):.0%}"), ("Worsen", f"{e.get('Worsen', 0):.0%}")])
    st.subheader("Do people with higher scores feel music helps more?")
    rows = []
    for m in MH:
        lvl = pd.cut(df[m], [-1, 3, 6, 10], labels=["Low (0-3)", "Mid (4-6)", "High (7-10)"])
        r = df.assign(L=lvl).dropna(subset=["Improves"]).groupby("L", observed=True)["Improves"].mean().reset_index(); r["Condition"] = m; rows.append(r)
    fig = px.bar(pd.concat(rows), x="Condition", y="Improves", color="L", barmode="group", color_discrete_sequence=[LINE, PAL[1], PAL[0]])
    fig.update_yaxes(tickformat=".0%"); show(style(fig, 320))
    st.subheader("Genre frequency against each condition")
    show(heat(mining.genre_wellbeing_corr(df).sort_values("Anxiety"), 480))
    st.caption("Spearman correlation. Values near zero mean no clear link. Correlation does not show cause: someone who is already anxious may choose different music.")

def segs():
    head("Listener segments", "K-Means groups listeners by genre habits, hours and condition scores. No genre labels are given to it.")
    k = st.slider("Number of segments", 2, 7, 4)
    lab, xy, sil = mining.segments(df, k); d = df.assign(Segment=lab, x=xy[:, 0], y=xy[:, 1])
    band([("Silhouette score", f"{sil:.2f}"), ("Segments", k), ("Listeners", len(d))])
    a, b = st.columns([3, 2], gap="large")
    with a: st.subheader("Listener map"); show(style(px.scatter(d, x="x", y="y", color="Segment", hover_data=["Age", "Fav genre"], opacity=.75), 400))
    with b: st.subheader("Does k fit?"); show(style(px.line(mining.silhouette_scan(df), x="k", y="silhouette", markers=True), 400))
    st.subheader("What each segment plays")
    show(heat(d.groupby("Segment")[GENRES].mean(), 320))
    st.subheader("Segment profiles")
    p = d.groupby("Segment").agg(listeners=("Age", "size"), age=("Age", "mean"), hours=("Hours per day", "mean"), anxiety=("Anxiety", "mean"),
        depression=("Depression", "mean"), insomnia=("Insomnia", "mean"), ocd=("OCD", "mean"), says_music_helps=("Improves", "mean")).round(2)
    st.dataframe(p.sort_values("listeners", ascending=False))

def means():
    head("What it means for artists", "A Random Forest finds what separates listeners who say music improves their mood. The notes beside it come from the survey.")
    imp, auc, base = drivers(df)
    band([("Model AUC (5-fold)", f"{auc:.2f}"), ("Say music helps", f"{base:.0%}")])
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.subheader("What predicts 'music improves my mood'")
        show(style(px.bar(imp.tail(12), orientation="h", color_discrete_sequence=[PAL[0]]), 440))
        st.caption("AUC 0.5 is a coin flip. Treat low values as a sign the survey cannot predict this well.")
    with b:
        st.subheader("Findings")
        lab, _, _ = mining.segments(df, 4)
        for t in mining.insights(df, lab): st.markdown(f'<div class="note">{t}</div>', unsafe_allow_html=True)

def survey():
    head("Survey data", "The Music & Mental Health survey behind every chart. Filters in the sidebar apply here too.")
    band([("Rows shown", len(df)), ("Columns", len(ALL.columns)), ("Missing BPM", f"{ALL.BPM.isna().mean():.0%}")])
    q = st.text_input("Search favourite genre or service")
    t = df[df.apply(lambda r: q.lower() in f"{r['Fav genre']} {r['Primary streaming service']}".lower(), axis=1)] if q else df
    st.dataframe(t.drop(columns=[c for c in t if c.endswith("_n")]), height=420)
    st.download_button("Download shown rows (CSV)", t.to_csv(index=False), "artistiq_filtered.csv")
    st.subheader("Replace the data")
    up = st.file_uploader("Survey CSV in the same format", type="csv")
    if up:
        raw = pd.read_csv(up); miss = [c for c in db.load("survey").columns if c not in raw.columns]
        if miss: st.error("Missing columns: " + ", ".join(miss[:6]))
        elif st.button("Use this file"): db.save(raw); st.cache_data.clear(); st.rerun()
    if st.button("Restore bundled dataset"): db.reset(); st.cache_data.clear(); st.rerun()

{"Overview": overview, "Genres": genres, "Wellbeing": wellbeing, "Listener segments": segs, "What it means": means, "Survey data": survey}[PAGE]()
