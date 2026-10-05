# Branche « Manipulation » — bras robotiques

> Faire saisir, déplacer et assembler des objets à un bras robotique, du cobot industriel
> au bras d'assistance.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `kinematics` | Chaîne série DH standard (rotoïde/prismatique), cinématique directe, jacobienne géométrique, butées, manipulabilité de Yoshikawa |
| `ik` | Cinématique inverse numérique par moindres carrés amortis (pose 6D ou position seule) |
| `trajectory` | Polynômes quintiques multi-articulations (C2) et profils de vitesse trapézoïdaux |
| `robots` | Bras plans 2R/3R et cobot UR5e (paramètres DH publiés) |

```bash
python projets/manipulation/examples/demo_pick_place.py
pytest projets/manipulation
```

## Pistes porteuses (prochaines pierres)

1. **Dynamique** : Newton-Euler récursif (RNEA), commande en couple calculé, compensation de gravité.
2. **Planification de mouvement** : RRT-Connect / CHOMP / TrajOpt dans l'espace articulaire,
   détection de collision (FCL, maillages convexes).
3. **Saisie** : génération de prises sur nuages de points (GraspNet, Contact-GraspNet).
4. **Commande en effort / impédance** pour la manipulation au contact et la cobotique sûre (ISO/TS 15066).
5. **Apprentissage par démonstration** : téléopération + politiques de type ACT,
   Diffusion Policy, modèles VLA (π0, OpenVLA) — voir la branche *apprentissage*.
6. **Intégration** : MoveIt 2 via la branche *ros2*, description URDF/MJCF, simulation MuJoCo.
