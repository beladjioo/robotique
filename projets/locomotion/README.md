# Branche « Locomotion » — robots à pattes et humanoïdes

> Faire marcher, trotter et courir des robots à pattes, des quadrupèdes d'inspection aux
> humanoïdes, en restant stables face aux poussées et aux terrains irréguliers.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `lipm` | Pendule inversé linéaire (solution analytique), point de capture, énergie orbitale, marche par contrôle du DCM robuste aux poussées |
| `cpg` | Générateur central de rythme à 4 oscillateurs de Hopf couplés ; allures marche, trot, amble, bond ; transitions continues |
| `leg` | Cinématique directe/inverse d'une jambe 2 segments, trajectoire de pied appui/vol |

```bash
python projets/locomotion/examples/demo_marche.py
pytest projets/locomotion
```

## Pistes porteuses (prochaines pierres)

1. **Marche 3D** : DCM latéral + sagittal, planification de pas, ZMP preview control (Kajita).
2. **MPC centroidale** (modèle à corps rigide unique) pour quadrupèdes, type MIT Cheetah.
3. **Commande corps-complet** (whole-body control, QP hiérarchiques) pour humanoïdes.
4. **Apprentissage par renforcement massivement parallèle** (Isaac Lab, MuJoCo MJX) puis sim-to-real.
5. **Perception du terrain** : cartes d'élévation, choix de points d'appui.
6. **Humanoïdes généralistes** : imitation de mouvements humains (motion retargeting), modèles de fondation.
