"""musicdetect : ce que voit un détecteur de musique IA.

Sous-paquets
------------
audio       décodage, rééchantillonnage, chaîne de normalisation commune, attaques
data        FMA (musique réelle), FakeMusicCaps (test), index de segments
codecs      codecs neuronaux (EnCodec, DAC, Mimi) et substituts numpy pour les tests
detectors   M0 (métadonnées), M1 (fakeprint), M2 (CNN), M3 (corrélation croisée), M4 (SONICS)
eval        métriques, bootstrap groupé, calibration, protocole d'évaluation
"""

__version__ = "0.1.0"
