"""Analyses confirmatoires de la phase 1 (preregistration.md §3), figures et analyse d'erreurs.

Lit uniquement des fichiers déjà produits : `results/<tag>/*.csv` (métriques), `results/<tag>/scores/*.csv`
(scores par segment, y compris sous attaque) et les index de segments. Aucun modèle n'est rechargé.

Un seul tirage bootstrap des groupes (artiste FMA, légende FakeMusicCaps) sert à toutes les séries : deux AUC
calculées sur des segments qui partagent des groupes sont donc comparées de façon appariée, qu'il s'agisse de
deux détecteurs, de deux générateurs (les légendes sont communes aux cinq) ou d'un jeu propre et attaqué.

    python -m musicdetect.analysis --results results --fma-index data/work/index_fma_phase1.csv \
        --fmc-index data/work/index_fmc_phase1.csv --manifest data/work/manifest_fma.csv --out results/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .eval.bootstrap import holm
from .eval.metrics import rates_at_threshold

GENS = ["musicgen", "stable_audio_open", "audioldm2", "musicldm", "mustango"]
FAMILY = {"musicgen": 1, "stable_audio_open": 0, "audioldm2": 0, "musicldm": 0, "mustango": 0}  # préreg. §2
CODECS = ["encodec_24k_6kbps", "encodec_24k_12kbps"]
SEED_TAGS = ["phase1_s1", "phase1_s2", "phase1_s3"]
ATTACKED = [("phase1", "m1"), ("phase1_s1", "m3"), ("phase1_s1", "m2")]
META = ["meta_sample_rate", "meta_channels", "meta_bit_rate", "meta_duration"]

LABEL = {"m0-residual": "M0 (résiduel)", "m1": "M1 (fakeprint)", "m2": "M2 (CNN)", "m3": "M3 (log-fréq.)",
         "m4": "M4 (SONICS)"}
CODEC_LABEL = {"encodec_24k_6kbps": "6 kbps", "encodec_24k_12kbps": "12 kbps"}
GEN_LABEL = {"musicgen": "MusicGen", "stable_audio_open": "Stable Audio Open", "audioldm2": "AudioLDM2",
             "musicldm": "MusicLDM", "mustango": "Mustango"}


def _test_key(test: str) -> str:
    return test.replace(":", "_")


class Scores:
    """Accès aux fichiers de scores par segment d'un dossier de résultats."""

    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[Path, pd.DataFrame] = {}

    def path(self, tag: str, stem: str, test: str, attack: str | None = None) -> Path:
        name = f"{stem}__{_test_key(test)}.csv" if attack is None else f"attacks_{stem}__{_test_key(test)}__{attack}.csv"
        return self.root / tag / "scores" / name

    def get(self, tag: str, stem: str, test: str, attack: str | None = None) -> pd.DataFrame | None:
        p = self.path(tag, stem, test, attack)
        if p not in self.cache:
            if not p.exists():
                return None
            self.cache[p] = pd.read_csv(p, dtype={"seg_id": str, "group": str})
        return self.cache[p]


class Bootstrap:
    """Tirages multinomiaux des groupes, partagés par toutes les séries ; AUC pondérée = AUC sur le rééchantillon."""

    def __init__(self, groups: np.ndarray, B: int, seed: int):
        self.uniq = np.unique(groups.astype(str))
        n = len(self.uniq)
        rng = np.random.default_rng(seed)
        self.counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.int32)
        self.B = B
        self.memo: dict[int, np.ndarray] = {}

    def aucs(self, df: pd.DataFrame) -> np.ndarray:
        key = id(df)
        if key not in self.memo:
            y, s = df["label"].to_numpy(), df["score"].to_numpy(dtype=np.float64)
            w = self.counts[:, np.searchsorted(self.uniq, df["group"].to_numpy(dtype=str))]
            out = np.full(self.B, np.nan)
            for b in range(self.B):
                wb = w[b]
                if wb[y == 1].sum() > 0 and wb[y == 0].sum() > 0:
                    out[b] = roc_auc_score(y, s, sample_weight=wb)
            self.memo[key] = out
        return self.memo[key]


def auc_of(df: pd.DataFrame) -> float:
    return float(roc_auc_score(df["label"], df["score"]))


def ci(rep: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(rep, [2.5, 97.5])
    return float(lo), float(hi)


def p_greater(rep: np.ndarray, c: float) -> float:
    """p unilatérale pour H : statistique > c (inversion de l'intervalle percentile)."""
    r = rep[~np.isnan(rep)]
    return float((1 + np.sum(r <= c)) / (len(r) + 1))


def p_less(rep: np.ndarray, c: float) -> float:
    r = rep[~np.isnan(rep)]
    return float((1 + np.sum(r >= c)) / (len(r) + 1))


def p_two(rep: np.ndarray, c: float = 0.0) -> float:
    return min(1.0, 2 * min(p_greater(rep, c), p_less(rep, c)))


def stat_row(point: float, rep: np.ndarray, **extra) -> dict:
    lo, hi = ci(rep)
    return {**extra, "valeur": point, "ic_bas": lo, "ic_haut": hi}


# --------------------------------------------------------------------------- hypothèses


def h1_h2(sc: Scores, bs: Bootstrap, tag: str, stem_of, label: str) -> tuple[pd.DataFrame, dict]:
    """AUC par générateur, Spearman (famille, AUC) sur 5 générateurs x 2 codecs, écart MusicGen - autres."""
    rows, reps, points = [], [], []
    for codec in CODECS:
        for g in GENS:
            df = sc.get(tag, stem_of(codec), f"fmc:{g}")
            if df is None:
                return pd.DataFrame(), {}
            rep, a = bs.aucs(df), auc_of(df)
            rows.append({"detecteur": label, "codec": codec, "generateur": g, "famille": FAMILY[g], "auc": a,
                         "ic_bas": ci(rep)[0], "ic_haut": ci(rep)[1], "p_auc_sup_075": p_greater(rep, 0.75)})
            reps.append(rep)
            points.append(a)
    t = pd.DataFrame(rows)
    for codec in CODECS:
        m = t["codec"] == codec
        for i, (adj, _) in zip(t.index[m], holm(t.loc[m, "p_auc_sup_075"].tolist())):
            t.loc[i, "p_holm_auc_sup_075"] = adj

    fam = np.array([FAMILY[g] for g in GENS] * len(CODECS))
    R = np.vstack(reps)
    rho = float(spearmanr(fam, points).statistic)
    rho_rep = np.array([spearmanr(fam, R[:, b]).statistic if not np.isnan(R[:, b]).any() else np.nan
                        for b in range(R.shape[1])])
    out = {"spearman": stat_row(rho, rho_rep, p_rho_sup_0=p_greater(rho_rep, 0.0))}
    for k, codec in enumerate(CODECS):
        block = R[k * len(GENS):(k + 1) * len(GENS)]
        pts = np.array(points[k * len(GENS):(k + 1) * len(GENS)])
        d_rep = block[0] - block[1:].mean(axis=0)
        out[f"delta_{codec}"] = stat_row(float(pts[0] - pts[1:].mean()), d_rep, p_delta_sup_0=p_greater(d_rep, 0.0))
    return t, out


def h1_secondary(sc: Scores, bs: Bootstrap) -> pd.DataFrame:
    """Écart MusicGen - moyenne des autres pour chaque détecteur, graine et codec (lecture secondaire de H1)."""
    rows = []
    runs = [("phase1", "m0-residual"), ("phase1", "m1")] + [(t, d) for t in SEED_TAGS for d in ("m2", "m3")]
    for tag, det in runs:
        for codec in CODECS:
            dfs = [sc.get(tag, f"{det}__{codec}", f"fmc:{g}") for g in GENS]
            if any(d is None for d in dfs):
                continue
            R = np.vstack([bs.aucs(d) for d in dfs])
            pts = np.array([auc_of(d) for d in dfs])
            rows.append(stat_row(float(pts[0] - pts[1:].mean()), R[0] - R[1:].mean(axis=0),
                                 detecteur=det, tag=tag, codec=codec, auc_musicgen=pts[0], auc_autres=pts[1:].mean()))
    return pd.DataFrame(rows)


def h3(sc: Scores, bs: Bootstrap) -> pd.DataFrame:
    """M0 contre le hasard, puis M1 - M0 apparié, par générateur et codec."""
    rows = []
    for codec in CODECS:
        for g in GENS:
            m0 = sc.get("phase1", f"m0-residual__{codec}", f"fmc:{g}")
            m1 = sc.get("phase1", f"m1__{codec}", f"fmc:{g}")
            if m0 is None or m1 is None:
                continue
            r0, r1 = bs.aucs(m0), bs.aucs(m1)
            a0, a1 = auc_of(m0), auc_of(m1)
            rows.append({"codec": codec, "generateur": g, "auc_m0": a0, "m0_ic_bas": ci(r0)[0], "m0_ic_haut": ci(r0)[1],
                         "p_m0_sup_05": p_greater(r0, 0.5), "auc_m1": a1, "delta_m1_m0": a1 - a0,
                         "delta_ic_bas": ci(r1 - r0)[0], "delta_ic_haut": ci(r1 - r0)[1], "p_delta": p_two(r1 - r0)})
    return pd.DataFrame(rows)


def raw_metadata(sc: Scores, fma_idx: pd.DataFrame, fmc_idx: pd.DataFrame) -> pd.DataFrame:
    """AUC de chaque métadonnée de fichier prise telle quelle comme score : le raccourci AVANT la chaîne commune.
    (M0 « brut » entraîné sur les paires FMA n'apprendrait rien : réel et ré-encodé héritent des métadonnées du
    même MP3 d'origine. Le raccourci n'existe qu'entre FMA et FakeMusicCaps, c'est donc là qu'on le mesure.)"""
    meta = pd.concat([fma_idx[["seg_id", *META]], fmc_idx[["seg_id", *META]]]).drop_duplicates("seg_id")
    rows = []
    for test in [f"fmc:{g}" for g in GENS] + ["fma:encodec_24k_6kbps"]:
        base = sc.get("phase1", "m1__encodec_24k_6kbps", test)
        if base is None:
            continue
        m = base[["seg_id", "label"]].merge(meta, on="seg_id", how="left")
        for f in META:
            x = pd.to_numeric(m[f], errors="coerce")
            ok = x.notna()
            if ok.sum() == 0 or m.loc[ok, "label"].nunique() < 2:
                continue
            a = float(roc_auc_score(m.loc[ok, "label"], x[ok]))
            rows.append({"test": test, "metadonnee": f, "auc": a, "separabilite": max(a, 1 - a),
                         "valeurs_reel": ", ".join(map(str, sorted(x[ok & (m["label"] == 0)].unique())[:4])),
                         "valeurs_faux": ", ".join(map(str, sorted(x[ok & (m["label"] == 1)].unique())[:4]))})
    return pd.DataFrame(rows)


def h4(sc: Scores, bs: Bootstrap, test: str = "fmc:musicgen") -> tuple[pd.DataFrame, pd.DataFrame]:
    """ΔAUC (attaque - propre) apparié par détecteur, puis différence des ΔAUC entre M1 et M3."""
    rows, deltas = [], {}
    for tag, det in ATTACKED:
        stem = f"{det}__encodec_24k_6kbps"
        clean = sc.get(tag, stem, test, "clean")
        if clean is None:
            continue
        files = sorted((sc.root / tag / "scores").glob(f"attacks_{stem}__{_test_key(test)}__*.csv"))
        for f in files:
            att = f.stem.split("__")[-1]
            if att == "clean":
                continue
            df = sc.get(tag, stem, test, att)
            d_rep = bs.aucs(df) - bs.aucs(clean)
            d = auc_of(df) - auc_of(clean)
            deltas[(det, att)] = (d, d_rep)
            rows.append(stat_row(d, d_rep, detecteur=det, tag=tag, attaque=att, auc_propre=auc_of(clean),
                                 auc_attaque=auc_of(df), p_delta=p_two(d_rep)))
    contrast = []
    for (det, att), (d1, r1) in deltas.items():
        if det != "m1" or ("m3", att) not in deltas:
            continue
        d3, r3 = deltas[("m3", att)]
        diff = r1 - r3
        # H4 prédit une chute plus forte pour M1 sous vitesse (diff < 0) ; pour le MP3, aucune direction prédite.
        p = p_less(diff, 0.0) if att.startswith("speed") else p_two(diff)
        contrast.append(stat_row(d1 - d3, diff, attaque=att, delta_m1=d1, delta_m3=d3, p=p,
                                 test="unilateral (M1 chute plus)" if att.startswith("speed") else "bilateral"))
    return pd.DataFrame(rows), pd.DataFrame(contrast)


def criteria(t12: pd.DataFrame, h1: dict, t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    """Chaque critère de réfutation de preregistration.md §3, évalué tel qu'il est écrit."""
    rows = []
    if h1:
        s = h1["spearman"]
        rows.append({"hypothese": "H1", "critere": "rho < 0, ou IC 95 % de rho contenant 0",
                     "observe": f"rho = {s['valeur']:.3f} [{s['ic_bas']:.3f} ; {s['ic_haut']:.3f}]",
                     "refutee": bool(s["valeur"] < 0 or s["ic_bas"] <= 0 <= s["ic_haut"])})
    if len(t12):
        six = t12[t12["codec"] == "encodec_24k_6kbps"].set_index("generateur")
        mg = six.loc["musicgen"]
        hi = [g for g in GENS[1:] if six.loc[g, "auc"] >= 0.75 and six.loc[g, "ic_bas"] > 0.5]
        rows.append({"hypothese": "H2", "critere": "AUC(MusicGen) < 0,75, ou AUC >= 0,75 sur au moins deux autres",
                     "observe": f"MusicGen {mg['auc']:.3f} ; autres >= 0,75 : {', '.join(hi) or 'aucun'}",
                     "refutee": bool(mg["auc"] < 0.75 or len(hi) >= 2)})
    if len(t3):
        r = t3[(t3["codec"] == "encodec_24k_6kbps") & (t3["generateur"] == "musicgen")].iloc[0]
        contains = bool(r["m0_ic_bas"] <= 0.5 <= r["m0_ic_haut"])
        rows.append({"hypothese": "H3", "critere": "IC 95 % de M0 contenant 0,5 ET pas de chute de M1/M2 après "
                     "correction du raccourci (seconde condition non mesurée)",
                     "observe": f"M0 MusicGen {r['auc_m0']:.3f} [{r['m0_ic_bas']:.3f} ; {r['m0_ic_haut']:.3f}]",
                     "refutee": contains})
    if len(t4):
        def drops(det):
            sub = t4[(t4["detecteur"] == det) & t4["attaque"].str.startswith("speed")]
            return {a: -v for a, v in zip(sub["attaque"], sub["valeur"])}
        d1, d3 = drops("m1"), drops("m3")
        if d1 and d3:
            rows.append({"hypothese": "H4", "critere": "chute de M1 sous vitesse < 0,10, ou chute de M3 >= 0,10",
                         "observe": "chutes M1 : " + ", ".join(f"{a} {v:+.3f}" for a, v in d1.items())
                                    + " ; M3 : " + ", ".join(f"{a} {v:+.3f}" for a, v in d3.items()),
                         "refutee": bool(min(d1.values()) < 0.10 or max(d3.values()) >= 0.10)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- erreurs


def errors(sc: Scores, fma_idx: pd.DataFrame, fmc_idx: pd.DataFrame, manifest: pd.DataFrame | None,
           metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """20 faux les moins suspects par générateur, 20 réels les plus suspects ; FPR par genre au seuil de validation."""
    info = pd.concat([fma_idx[["seg_id", "path", "track_id"]], fmc_idx[["seg_id", "path", "track_id"]]])
    info = info.drop_duplicates("seg_id")
    if manifest is not None:
        info = info.merge(manifest[["track_id", "genre_top"]], on="track_id", how="left")
    worst = []
    for tag, stem in [("phase1", "m1__encodec_24k_6kbps"), ("phase1_s1", "m3__encodec_24k_6kbps")]:
        for g in GENS:
            df = sc.get(tag, stem, f"fmc:{g}")
            if df is None:
                continue
            df = df.merge(info, on="seg_id", how="left")
            miss = df[df["label"] == 1].nsmallest(20, "score").assign(erreur="faux manqué", generateur=g)
            worst.append(miss.assign(detecteur=stem, tag=tag))
        df = sc.get(tag, stem, "fmc:musicgen")
        if df is not None:
            fa = df.merge(info, on="seg_id", how="left")
            worst.append(fa[fa["label"] == 0].nlargest(20, "score").assign(erreur="fausse alarme", generateur="-",
                                                                          detecteur=stem, tag=tag))
    worst = pd.concat(worst, ignore_index=True) if worst else pd.DataFrame()

    by_genre = []
    if manifest is not None and len(metrics):
        runs = [("phase1", "m0-residual"), ("phase1", "m1")] + [(t, d) for t in SEED_TAGS for d in ("m2", "m3")]
        for tag, det in runs:
            stem = f"{det}__encodec_24k_6kbps"
            df = sc.get(tag, stem, "fma:encodec_24k_6kbps")
            thr = metrics.loc[(metrics["tag"] == tag) & (metrics["stem"] == stem), "threshold"]
            if df is None or thr.empty or pd.isna(thr.iloc[0]):
                continue
            real = df[df["label"] == 0].merge(info[["seg_id", "genre_top"]], on="seg_id", how="left")
            for genre, part in real.groupby(real["genre_top"].fillna("inconnu")):
                _, fpr = rates_at_threshold(np.zeros(len(part)), part["score"].to_numpy(), float(thr.iloc[0]))
                by_genre.append({"detecteur": det, "tag": tag, "genre": genre, "n_segments": len(part), "fpr": fpr})
    return worst, pd.DataFrame(by_genre)


# --------------------------------------------------------------------------- figures

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")


def figures(metrics: pd.DataFrame, attacks: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    _style(plt)
    cols = ["en-domaine", "fma:identity_24k"] + [f"fmc:{g}" for g in GENS]
    col_names = ["FMA, codec\nd'entraînement", "FMA identité\n(contrôle)"] + [GEN_LABEL[g].replace(" Open", "\nOpen")
                                                                             for g in GENS]

    def cell(df, codec, c):
        t = f"fma:{codec}" if c == "en-domaine" else c
        v = df.loc[df["test"] == t, "auc"]
        return float(v.mean()) if len(v) else np.nan

    rows, names = [], []
    for det in ["m0-residual", "m1", "m2", "m3"]:
        for codec in CODECS:
            sub = metrics[(metrics["stem"] == f"{det}__{codec}")]
            if sub.empty:
                continue
            n_seeds = sub["tag"].nunique()
            rows.append([cell(sub, codec, c) for c in cols])
            names.append(f"{LABEL[det]}, {CODEC_LABEL[codec]}" + (f" (moy. {n_seeds} graines)" if n_seeds > 1 else ""))
    m4 = metrics[metrics["stem"].str.startswith("m4")]
    if not m4.empty:
        rows.append([np.nan if c == "en-domaine" else cell(m4, "", c) for c in cols])
        names.append("M4 (SONICS, pré-entraîné)")
    if rows:
        A = np.array(rows, dtype=float)
        cmap = LinearSegmentedColormap.from_list("auc", [ORANGE, "#f0efec", BLUE])
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3))
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=TwoSlopeNorm(0.5, 0.0, 1.0), aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if not np.isnan(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(A[i, j] - 0.5) > 0.33 else INK)
        ax.set_xticks(range(len(cols)), col_names, fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("AUC (0,5 = hasard)")
        cb.outline.set_visible(False)
        _save(fig, out, "fig_matrice_auc")
        plt.close(fig)

    if len(attacks):
        order = ["clean", "speed_0.9", "speed_1.1", "mp3_128"]
        names = {"clean": "sans attaque", "speed_0.9": "vitesse ×0,9", "speed_1.1": "vitesse ×1,1", "mp3_128": "MP3 128"}
        att = [a for a in order if a in set(attacks["attack"])]
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        series = [("m1", BLUE, "o"), ("m3", ORANGE, "s"), ("m2", AQUA, "^")]
        for k, (det, color, marker) in enumerate(series):
            sub = attacks[attacks["detector"] == det].set_index("attack").reindex(att)
            if sub["auc"].isna().all():
                continue
            x = np.arange(len(att)) + (k - 1) * 0.18
            ax.errorbar(x, sub["auc"], yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]], fmt=marker,
                        color=color, ms=6, lw=1.2, capsize=0, label=LABEL[det])
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.text(-0.45, 0.505, "hasard", color=MUTED, fontsize=8, ha="left", va="bottom")
        ax.set_xticks(range(len(att)), [names[a] for a in att])
        ax.set_ylabel("AUC sur MusicGen (IC 95 %)")
        ax.set_ylim(0.15, 1.0)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
        _save(fig, out, "fig_attaques")
        plt.close(fig)

    m = metrics[(metrics["tag"] == "phase1") & metrics["test"].str.startswith("fmc:")]
    if not m.empty:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        for k, (det, color, marker) in enumerate([("m1", BLUE, "o"), ("m0-residual", ORANGE, "s")]):
            sub = m[m["stem"] == f"{det}__encodec_24k_6kbps"].set_index("test").reindex([f"fmc:{g}" for g in GENS])
            x = np.arange(len(GENS)) + (k - 0.5) * 0.2
            ax.errorbar(x, sub["auc"], yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]], fmt=marker,
                        color=color, ms=6, lw=1.2, capsize=0, label=LABEL[det])
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(GENS)), [GEN_LABEL[g] for g in GENS])
        ax.set_ylabel("AUC (IC 95 %)")
        ax.set_ylim(0.45, 1.0)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        _save(fig, out, "fig_m1_contre_m0")
        plt.close(fig)


# --------------------------------------------------------------------------- orchestration


def load_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("phase1")):
        for f in tag_dir.glob("*.csv"):
            if f.name == "summary.csv" or f.name.startswith("attacks_"):
                continue
            df = pd.read_csv(f)
            frames.append(df.assign(tag=tag_dir.name, stem=f.stem))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_attacks(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(root / tag / f"attacks_{det}__encodec_24k_6kbps.csv").assign(tag=tag)
              for tag, det in ATTACKED if (root / tag / f"attacks_{det}__encodec_24k_6kbps.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(results: Path, fma_index: Path, fmc_index: Path, manifest: Path | None, out: Path, B: int = 1000,
        seed: int = 0) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    sc = Scores(results)
    all_files = list(results.glob("phase1*/scores/*.csv"))
    groups = np.unique(np.concatenate([pd.read_csv(f, usecols=["group"], dtype={"group": str})["group"].to_numpy()
                                       for f in all_files]))
    bs = Bootstrap(groups, B, seed)
    fma_idx = pd.read_csv(fma_index, dtype={"seg_id": str})
    fmc_idx = pd.read_csv(fmc_index, dtype={"seg_id": str})
    man = pd.read_csv(manifest) if manifest and Path(manifest).exists() else None
    metrics = load_metrics(results)
    log = []

    def note(msg: str):
        print(msg, flush=True)
        log.append(msg)

    note(f"bootstrap : {B} tirages sur {len(groups)} groupes, graine {seed}")
    t12, h1 = h1_h2(sc, bs, "phase1", lambda c: f"m1__{c}", "m1")
    t12.to_csv(out / "h1_h2_m1_par_generateur.csv", index=False)
    pd.DataFrame([{"statistique": k, **v} for k, v in h1.items()]).to_csv(out / "h1_m1.csv", index=False)
    h1_secondary(sc, bs).to_csv(out / "h1_secondaire_tous_detecteurs.csv", index=False)
    t3 = h3(sc, bs)
    t3.to_csv(out / "h3_m0_et_m1_moins_m0.csv", index=False)
    raw_metadata(sc, fma_idx, fmc_idx).to_csv(out / "h3_metadonnees_brutes.csv", index=False)
    t4, c4 = h4(sc, bs)
    t4.to_csv(out / "h4_delta_auc_par_attaque.csv", index=False)
    c4.to_csv(out / "h4_m1_contre_m3.csv", index=False)
    worst, by_genre = errors(sc, fma_idx, fmc_idx, man, metrics)
    worst.to_csv(out / "erreurs_20_pires.csv", index=False)
    if len(by_genre):
        by_genre.to_csv(out / "fpr_par_genre.csv", index=False)
    else:
        note("FPR par genre sauté : manifeste FMA (manifest_fma.csv) introuvable")

    # Tests confirmatoires (un par hypothèse), corrigés ensemble par Holm.
    conf = []
    if h1:
        d = h1["delta_encodec_24k_6kbps"]
        conf.append({"hypothese": "H1", "test": "AUC(M1->MusicGen) - moyenne autres > 0 (6 kbps)",
                     "valeur": d["valeur"], "ic_bas": d["ic_bas"], "ic_haut": d["ic_haut"], "p": d["p_delta_sup_0"]})
        r = t12[(t12["codec"] == "encodec_24k_6kbps") & (t12["generateur"] == "musicgen")].iloc[0]
        conf.append({"hypothese": "H2", "test": "AUC(M1->MusicGen) > 0,75 (6 kbps)", "valeur": r["auc"],
                     "ic_bas": r["ic_bas"], "ic_haut": r["ic_haut"], "p": r["p_auc_sup_075"]})
    if len(t3):
        r = t3[(t3["codec"] == "encodec_24k_6kbps") & (t3["generateur"] == "musicgen")].iloc[0]
        conf.append({"hypothese": "H3", "test": "AUC(M0->MusicGen) > 0,5 (6 kbps)", "valeur": r["auc_m0"],
                     "ic_bas": r["m0_ic_bas"], "ic_haut": r["m0_ic_haut"], "p": r["p_m0_sup_05"]})
    if len(c4) and (c4["attaque"] == "speed_1.1").any():
        r = c4[c4["attaque"] == "speed_1.1"].iloc[0]
        conf.append({"hypothese": "H4", "test": "ΔAUC(M1) - ΔAUC(M3) < 0, vitesse x1,1", "valeur": r["valeur"],
                     "ic_bas": r["ic_bas"], "ic_haut": r["ic_haut"], "p": r["p"]})
    else:
        note("H4 : scores par segment sous attaque absents, test confirmatoire non calculé")
    conf = pd.DataFrame(conf)
    if len(conf):
        adj = holm(conf["p"].tolist())
        conf["p_holm"] = [a for a, _ in adj]
        conf["rejet_h0_holm"] = [r for _, r in adj]
    conf.to_csv(out / "confirmatoires_holm.csv", index=False)
    crit = criteria(t12, h1, t3, t4)
    crit.to_csv(out / "criteres_refutation.csv", index=False)

    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    figures(metrics, load_attacks(results), figdir)

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format,
                           "display.max_colwidth", 120):
        note("\n=== Critères de réfutation pré-enregistrés (§3)\n" + crit.to_string(index=False))
        note("\n=== Tests confirmatoires (Holm sur les 4 hypothèses). « rejet_h0 » ne veut pas dire « hypothèse "
             "confirmée » : voir les critères ci-dessus.\n" + conf.to_string(index=False))
        if h1:
            s = h1["spearman"]
            note(f"\nH1 Spearman(famille, AUC M1), 5 générateurs x 2 codecs : rho = {s['valeur']:.3f} "
                 f"[{s['ic_bas']:.3f} ; {s['ic_haut']:.3f}], p(rho > 0) = {s['p_rho_sup_0']:.3f}")
            note("\nH2 AUC M1 par générateur\n" + t12[["codec", "generateur", "auc", "ic_bas", "ic_haut",
                                                       "p_holm_auc_sup_075"]].to_string(index=False))
        if len(t4):
            note("\nH4 ΔAUC (attaque - propre) sur MusicGen\n" + t4[["detecteur", "attaque", "auc_propre",
                 "auc_attaque", "valeur", "ic_bas", "ic_haut"]].to_string(index=False))
            note("\nH4 ΔAUC(M1) - ΔAUC(M3)\n" + c4.to_string(index=False))
    (out / "journal.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nécrit dans {out}")
    return conf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, type=Path, help="dossier contenant phase1, phase1_s1, ...")
    p.add_argument("--fma-index", required=True, type=Path)
    p.add_argument("--fmc-index", required=True, type=Path)
    p.add_argument("--manifest", type=Path, help="manifest_fma.csv (genre) ; sans lui, le FPR par genre est sauté")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    run(a.results, a.fma_index, a.fmc_index, a.manifest, a.out, a.bootstrap, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
