# Branche « Vision » — perception visuelle

> Donner des yeux au robot : se repérer, reconnaître et localiser des objets,
> reconstruire la scène en 3D.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `camera` | Caméra sténopé (convention OpenCV), distorsion radiale k1/k2 et son inverse, projection avec visibilité, rétroprojection |
| `geometry` | Homographie par DLT normalisée (Hartley), triangulation linéaire, matrices de projection, pose « look-at », erreur de reprojection |

```bash
python projets/vision/examples/demo_stereo.py
pytest projets/vision
```

## Pistes porteuses (prochaines pierres)

1. **Calibration** (méthode de Zhang sur damier) et calibration main-œil (lien *manipulation*).
2. **Marqueurs fiduciaires** ArUco / AprilTag : estimation de pose (PnP) pour la vérité terrain.
3. **Détection et segmentation** : YOLO, SAM 2 ; estimation de pose 6D d'objets (FoundationPose).
4. **Profondeur** : stéréo dense, profondeur monoculaire (Depth Anything), nuages de points.
5. **Modèles de fondation vision-langage** (CLIP, DINOv2, VLM) pour la perception à vocabulaire ouvert
   — base des politiques VLA de la branche *apprentissage*.
6. **Odométrie visuelle** en lien avec la branche *estimation*.
