# Architecture

## Couches

```
┌──────────────────────────────────────────────────────────────────────┐
│ Intégration : projets/ros2 (nœuds, launch, Docker)                   │  adaptateurs
├──────────────────────────────────────────────────────────────────────┤
│ Branches métier (Python) :                                           │
│ manipulation · navigation · estimation · commande · vision · drones  │  logique pure,
│ locomotion · essaims · apprentissage · planification                 │  testable seule
├──────────────────────────────────────────────────────────────────────┤
│ Socle : socle/ (robocore) — transformations, angles, intégrateurs    │  maths partagées
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│ projets/embarque (C++17) — boucle temps réel sur microcontrôleur     │  matériel, via HAL
└──────────────────────────────────────────────────────────────────────┘
```

### Règles de dépendance

- Une branche métier ne dépend **que** du socle (et de NumPy). Pas de dépendance croisée
  entre branches : les compositions se font dans `ros2` ou dans des démonstrateurs.
- `ros2` dépend des branches, jamais l'inverse : la logique ne connaît ni `rclpy` ni les messages.
- `embarque` ne dépend de rien ; l'algorithme PID existe en double (Python et C++) avec la
  même sémantique et des tests équivalents.

## Conventions

| Sujet | Convention |
|---|---|
| Unités | SI (m, s, kg, N, rad) |
| Repères | Directs ; robot mobile : x avant, y gauche, z haut (REP-103) ; caméra : x droite, y bas, z avant (OpenCV) |
| Poses | `T_a_b` = pose du repère *b* dans *a*, `p_a = T_a_b @ p_b` |
| Quaternions | `robocore` : `(w, x, y, z)` ; ROS : `(x, y, z, w)` — la conversion est faite dans `robotique_ros.conversions` |
| Cartes | Ligne 0 = bas de la carte (`y` minimal), comme `nav_msgs/OccupancyGrid` |
| Aléatoire | Toujours via un `numpy.random.Generator` injectable (tests reproductibles) |

## Stratégie de test

1. **Exactitude mathématique** : comparaisons à des solutions analytiques, à des différences
   finies (jacobiennes) ou à la force brute (allocation, plus courts chemins).
2. **Comportement en boucle fermée** : simulations courtes à graine fixe (le robot atteint son
   but, le pendule est stabilisé, l'essaim s'aligne).
3. **Robustesse** : cas limites (singularités, objectifs inatteignables, saturations, poussées).
4. **Intégration** : démos exécutées en CI ; tests ROS 2 (`launch_testing`) à venir.

## Langage naturel et LLM

La branche *planification* applique le schéma « LLM + planificateur » : le modèle de langage
(Claude, via sortie JSON contrainte par schéma) ne produit qu'un **objectif** ; celui-ci est
validé (objets connus, cohérence) puis un planificateur symbolique calcule un plan dont chaque
action est vérifiée par rejeu avant toute exécution. Le LLM n'émet jamais directement de commande moteur.
