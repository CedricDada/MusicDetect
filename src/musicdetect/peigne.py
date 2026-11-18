"""Ce que voit M1 en fréquence : empreintes moyennes, poids appris et contraste de peigne.

Un décodeur qui suréchantillonne à cadence de trame fixe laisse des pics aux multiples de cette cadence
(EnCodec 24 kHz : 75 trames/s). Le contraste de peigne d'un profil pour un espacement s est la moyenne,
en écarts-types du profil, de ses valeurs aux multiples de s (tolérance d'un bin). On le calcule pour la
différence d'empreinte moyenne (faux - réels FMA) de chaque source, et pour les poids de M1.

    python -m musicdetect.peigne --model musicdetect-etat/models/phase1/m1__encodec_24k_6kbps.joblib \
        --fma-index data/work/index_fma_phase1.csv --fma-dir data/work/segments/fma_phase1 \
        --fmc-index data/work/index_fmc_phase1.csv --fmc-dir data/work/segments/fmc_phase1 --out results/peigne
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import AQUA, BLUE, GRID, MUTED, ORANGE, _comma, _save, _style
from .audio.io import read_wav
from .detectors.base import Detector
from .detectors.features import fakeprint

YELLOW, MAGENTA = "#eda100", "#e87ba4"
SOURCES = [("encodec_24k_6kbps", "FMA EnCodec 6 kbps"), ("fmc:musicgen", "MusicGen"),
           ("fmc:audioldm2", "AudioLDM2"), ("fmc:stable_audio_open", "Stable Audio Open")]


def mean_fakeprint(paths: list[Path], cfg) -> tuple[np.ndarray, np.ndarray]:
    acc, f = [], None
    for p in paths:
        x, sr = read_wav(p)
        f, v = fakeprint(x, sr, cfg)
        acc.append(v)
    return f, np.mean(acc, axis=0)


def comb_contrast(profile: np.ndarray, f: np.ndarray, s: float) -> float:
    z = (profile - profile.mean()) / profile.std()
    step = f[1] - f[0]
    k = np.arange(np.ceil(f[0] / s), np.floor(f[-1] / s) + 1) * s
    idx = np.clip(np.round((k - f[0]) / step).astype(int), 0, len(f) - 1)
    return float(np.mean([z[max(i - 1, 0):i + 2].max() for i in idx]))


def run(model: Path, fma_index: Path, fma_dir: Path, fmc_index: Path, fmc_dir: Path, out: Path,
        n: int = 300, seed: int = 0) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    m1 = Detector.load(model)
    cfg = m1.fp_cfg
    fw, w = m1.weights()

    fma = pd.read_csv(fma_index, dtype=str)
    fmc = pd.read_csv(fmc_index, dtype=str)
    test = fma[fma["split"] == "test"]
    rng = np.random.default_rng(seed)
    pairs = rng.choice(sorted(set(test["pair_id"])), size=min(n, test["pair_id"].nunique()), replace=False)
    sel = test[test["pair_id"].isin(pairs)]

    f, real = mean_fakeprint([fma_dir / p for p in sel.loc[sel["source"] == "real_sr24000", "path"]], cfg)
    diffs = {}
    for src, name in SOURCES:
        if src.startswith("fmc:"):
            rows = fmc[fmc["source"] == src].sample(n, random_state=seed)
            paths = [fmc_dir / p for p in rows["path"]]
        else:
            paths = [fma_dir / p for p in sel.loc[sel["source"] == src, "path"]]
        diffs[name] = mean_fakeprint(paths, cfg)[1] - real
    pd.DataFrame({"freq_hz": f, "reel_fma": real, **diffs}).to_csv(out / "empreintes_moins_reel.csv", index=False)
    pd.DataFrame({"freq_hz": fw, "poids_m1": w}).to_csv(out / "poids_m1.csv", index=False)

    spacings = np.arange(40.0, 160.01, 0.5)
    curves = {name: [comb_contrast(d, f, s) for s in spacings] for name, d in diffs.items()}
    curves["Poids de M1"] = [comb_contrast(w, fw, s) for s in spacings]
    table = pd.DataFrame({"espacement_hz": spacings, **curves})
    table.to_csv(out / "contraste_peigne.csv", index=False)

    summary = pd.DataFrame([{"serie": k, **{f"peigne_{int(s)}hz": comb_contrast(*((w, fw) if k == "Poids de M1"
                                                                                 else (diffs[k], f)), s)
                                            for s in (50, 75, 100, 150)}} for k in curves])
    summary.to_csv(out / "contraste_peigne_resume.csv", index=False)
    top = np.argsort(w)[-12:][::-1]
    pd.DataFrame({"freq_hz": fw[top], "poids": w[top], "multiple_de_75": fw[top] / 75}).to_csv(
        out / "poids_m1_plus_grands.csv", index=False)

    figure(table, out)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 140):
        print(summary.to_string(index=False))
        print("\n12 plus grands poids de M1 (Hz) :", ", ".join(f"{x:.0f}" for x in fw[top]))
    return summary


def figure(table: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    panels = [
        ("Domaine d'entraînement", [("FMA EnCodec 6 kbps", BLUE, "-"), ("Poids de M1", ORANGE, "--")]),
        ("Générateurs FakeMusicCaps", [("MusicGen", AQUA, "-"), ("AudioLDM2", YELLOW, "--"),
                                       ("Stable Audio Open", MAGENTA, ":")]),
    ]
    x = table["espacement_hz"]
    for ax, (title, series) in zip(axes, panels):
        for s in (50, 75, 100, 150):
            ax.axvline(s, color=GRID, lw=1, zorder=0)
            ax.text(s, 1.0, f"{s}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color=MUTED)
        for name, color, ls in series:
            ax.plot(x, table[name], color=color, ls=ls, lw=1.6, label=name)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("espacement des pics (Hz)")
        ax.set_title(title, fontsize=9, pad=14)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.yaxis.set_major_formatter(_comma(1))
    axes[0].set_ylabel("contraste de peigne (é.-t.)")
    fig.tight_layout()
    _save(fig, out, "fig_peigne")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="m1__encodec_24k_6kbps.joblib")
    p.add_argument("--fma-index", required=True, type=Path)
    p.add_argument("--fma-dir", required=True, type=Path)
    p.add_argument("--fmc-index", required=True, type=Path)
    p.add_argument("--fmc-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=300, help="segments tirés par série")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    run(a.model, a.fma_index, a.fma_dir, a.fmc_index, a.fmc_dir, a.out, a.n, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
