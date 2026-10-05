# Branche « Estimation » — localisation, fusion de capteurs, SLAM

> Savoir où est le robot et à quoi ressemble son environnement, à partir de capteurs
> imparfaits (odométrie, lidar, caméra, IMU, GNSS).

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `models` | Modèle de mouvement unicycle (point milieu) et mesure distance/gisement, avec jacobiennes analytiques vérifiées par différences finies |
| `ekf` | Localisation EKF sur amers connus (forme de Joseph, distance de Mahalanobis pour le rejet d'aberrations) |
| `particle_filter` | Localisation de Monte-Carlo : poids en log, rééchantillonnage systématique, localisation globale (« robot kidnappé ») |
| `simulation` | Simulateur de trajectoire, odométrie bruitée et mesures à portée limitée |

```bash
python projets/estimation/examples/demo_localisation.py --plot
pytest projets/estimation
```

## Pistes porteuses (prochaines pierres)

1. **EKF-SLAM puis SLAM par graphe** (factor graphs, GTSAM / g2o), fermeture de boucle.
2. **Cartographie d'occupation** par log-odds à partir d'un lidar (lien direct avec *navigation*).
3. **Fusion IMU** : filtre de Kalman d'erreur (ESKF) sur SO(3), préintégration IMU.
4. **Odométrie visuelle / visuelle-inertielle** (ORB-SLAM3, VINS-Fusion) — lien avec *vision*.
5. **Cartes neuronales** : NeRF / 3D Gaussian Splatting pour la reconstruction dense en temps réel.
6. **Estimation robuste** : M-estimateurs, gestion des aberrations, intégrité de la localisation.
