# Feuille de route

## Phase 0 — Fondations ✅

Les douze branches et le socle existent, chacune avec une bibliothèque minimale testée,
une démo et une CI (lint, tests Python 3.10/3.12, tests C++ g++/clang, démos).

## Phase 1 — Simulation et intégration

- [ ] **ros2** : valider la démo de navigation dans le conteneur `ros:jazzy` et l'ajouter à la CI.
- [ ] Choisir le simulateur physique de référence (MuJoCo pour la manipulation et la locomotion,
      Gazebo pour la robotique mobile) et décrire un premier robot en URDF/MJCF.
- [ ] **navigation** + **estimation** : cartographie d'occupation par lidar simulé et navigation dans la carte construite.
- [ ] **manipulation** : dynamique (RNEA) et planification de mouvement avec collisions.
- [ ] **commande** : MPC linéaire contraint, réutilisé par *drones* et *locomotion*.

## Phase 2 — Matériel

- [ ] **embarque** : base mobile différentielle sur ESP32 avec micro-ROS (odométrie, `cmd_vel`).
- [ ] **manipulation** : bras bas coût (type SO-100 / LeRobot) piloté depuis ROS 2.
- [ ] **vision** : calibration caméra et marqueurs AprilTag pour la vérité terrain.
- [ ] Banc de tests matériel en boucle (HIL) branché sur la CI.

## Phase 3 — Intelligence

- [ ] **apprentissage** : passage à Gymnasium + MuJoCo, PPO/SAC, sim-to-real.
- [ ] **apprentissage** + **manipulation** : collecte de démonstrations par téléopération,
      politiques ACT / Diffusion Policy, évaluation de modèles VLA.
- [ ] **planification** : agent LLM avec outils robotiques, replanification sur échec,
      ancrage perceptif de l'état symbolique (*vision*).
- [ ] **locomotion** : RL massivement parallèle pour un quadrupède simulé.

## Projets « fil rouge » (intégration de plusieurs branches)

| Démonstrateur | Branches mobilisées |
|---|---|
| **Robot mobile autonome** qui cartographie puis navigue dans un bâtiment | embarque, estimation, navigation, commande, ros2 |
| **Bras qui range une table sur instruction vocale** (« mets les cubes rouges dans la boîte ») | vision, planification, manipulation, apprentissage, ros2 |
| **Essaim de drones d'inspection** qui se répartit des zones et vole en formation | drones, essaims, estimation, commande |
| **Quadrupède** qui marche sur terrain irrégulier | locomotion, commande, apprentissage, estimation |

## Prochaine pierre de chaque branche

| Branche | Prochaine pierre |
|---|---|
| manipulation | Newton-Euler récursif + couple calculé |
| navigation | Hybrid A* pour robot non holonome |
| estimation | Cartographie d'occupation log-odds |
| commande | MPC linéaire avec contraintes (QP) |
| vision | Estimation de pose PnP + marqueurs |
| drones | Quadrirotor 3D, commande géométrique SE(3) |
| locomotion | Marche 3D (DCM latéral) et planification de pas |
| essaims | Allocation dynamique CBBA |
| apprentissage | Wrapper Gymnasium + PPO |
| planification | Import/export PDDL |
| embarque | Portage micro-ROS sur ESP32 |
| ros2 | Démo validée en CI + intégration Nav2 |
