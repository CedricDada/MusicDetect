"""Ligne de commande : `musicdetect <commande> [options]` (voir README pour l'enchaînement complet)."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import load_config
from .pipeline import (
    collect_results,
    get_paths,
    step_build_fma,
    step_build_fmc,
    step_compare,
    step_evaluate,
    step_fmc_index,
    step_manifest,
    step_train,
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="fichier YAML fusionné sur la configuration par défaut")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLE=VALEUR",
                   help="surcharge, ex. --set audio.anchor_sr=16000 (répétable)")
    p.add_argument("--tag", help="suffixe des dossiers de données / modèles / résultats (ablations)")
    p.add_argument("--device", help="cpu | cuda | cuda:0 ... (défaut : auto)")
    p.add_argument("--jobs", type=int, default=1, help="processus pour l'extraction de caractéristiques")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="musicdetect", description=__doc__)
    ap.add_argument("--version", action="version", version=f"musicdetect {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("config", help="affiche la configuration effective")
    _common(p)

    p = sub.add_parser("manifest", help="FMA : pistes retenues et split par artiste -> manifest_fma.csv")
    _common(p)

    p = sub.add_parser("fmc-index", help="FakeMusicCaps : liste des clips trouvés (à contrôler)")
    _common(p)

    p = sub.add_parser("build-fma", help="chaîne commune + codecs : écrit les segments WAV réels et faux")
    _common(p)
    p.add_argument("--codec", action="append", required=True, help="nom de la section `codecs` (répétable)")
    p.add_argument("--no-resume", action="store_true", help="reconstruit tout, sans reprise")

    p = sub.add_parser("build-fmc", help="FakeMusicCaps : clips passés par la chaîne (étapes 4 à 6)")
    _common(p)

    p = sub.add_parser("train", help="entraîne un détecteur (m0 | m1 | m2 | m3 ; m4 est en inférence seule)")
    _common(p)
    p.add_argument("--detector", required=True, choices=["m0", "m1", "m2", "m3", "m4"])
    p.add_argument("--codec", required=True, help="codec des faux d'entraînement")
    p.add_argument("--variant", help="m0 : residual | raw")

    p = sub.add_parser("evaluate", help="AUC avec IC groupé, EER, TPR/FPR au seuil de validation ; attaques")
    _common(p)
    p.add_argument("--detector", required=True, choices=["m0", "m1", "m2", "m3", "m4"])
    p.add_argument("--codec", required=True, help="codec des faux d'entraînement (sert aussi à retrouver le seuil)")
    p.add_argument("--test", action="append", required=True, metavar="SPEC",
                   help="fma:<codec>[:val|test] ou fmc:<générateur> (répétable)")
    p.add_argument("--variant")
    p.add_argument("--attacks", action="store_true", help="évalue la grille d'attaques sur le premier jeu de test")
    p.add_argument("--attack-on", metavar="SPEC", help="jeu de test des attaques (défaut : le premier --test)")
    p.add_argument("--bootstrap", type=int, help="nombre de tirages (défaut : eval.bootstrap)")

    p = sub.add_parser("compare", help="ΔAUC apparié entre deux fichiers de scores (results/<tag>/scores/*.csv)")
    p.add_argument("scores_a")
    p.add_argument("scores_b")
    p.add_argument("--bootstrap", type=int, default=1000)

    p = sub.add_parser("report", help="rassemble les CSV de résultats d'un dossier en summary.csv")
    _common(p)

    p = sub.add_parser("selftest", help="vérifie l'installation sur données synthétiques (sans PyTorch)")
    p.add_argument("--workdir", help="dossier à conserver (défaut : temporaire)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        from .selftest import run_selftest
        run_selftest(args.workdir)
        return 0
    if args.cmd == "compare":
        print(json.dumps(step_compare(args.scores_a, args.scores_b, args.bootstrap), indent=2))
        return 0

    cfg = load_config(args.config, args.overrides)
    paths = get_paths(cfg, args.tag)
    if args.cmd == "config":
        print(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "manifest":
        step_manifest(cfg, paths)
    elif args.cmd == "fmc-index":
        step_fmc_index(cfg, paths)
    elif args.cmd == "build-fma":
        step_build_fma(cfg, paths, args.codec, args.device, resume=not args.no_resume)
    elif args.cmd == "build-fmc":
        step_build_fmc(cfg, paths)
    elif args.cmd == "train":
        step_train(cfg, paths, args.detector, args.codec, args.device, args.jobs, args.variant)
    elif args.cmd == "evaluate":
        step_evaluate(cfg, paths, args.detector, args.codec, args.test, args.attacks, args.attack_on,
                      args.bootstrap, args.device, args.jobs, args.variant)
    elif args.cmd == "report":
        df = collect_results(paths.results)
        out = paths.results / "summary.csv"
        df.to_csv(out, index=False)
        cols = [c for c in ["detector", "variant", "train_codec", "test", "auc", "auc_lo", "auc_hi", "eer", "tpr", "fpr"]
                if c in df.columns]
        print(df[cols].to_string(index=False) if len(df) else "aucun résultat")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
