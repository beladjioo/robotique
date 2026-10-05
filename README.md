# Robotique

Un monorepo pour explorer les domaines porteurs de la robotique, une **branche** par
domaine. Chaque branche pose sa première pierre : une petite bibliothèque testée, une
démo exécutable et une feuille de route vers l'état de l'art.

```
robotique/
├── socle/                 robocore : maths partagées (SE(2)/SE(3), quaternions, intégrateurs…)
├── projets/
│   ├── manipulation/      bras série : DH, jacobienne, IK, trajectoires (UR5e)
│   ├── navigation/        robot mobile : carte d'occupation, A*, pure pursuit
│   ├── estimation/        localisation : EKF, filtre particulaire
│   ├── commande/          automatique : PID industriel, LQR, pendule inversé
│   ├── vision/            caméra sténopé, homographie, triangulation stéréo
│   ├── drones/            quadrirotor, commande en cascade, trajectoires lisses
│   ├── locomotion/        marche bipède (LIPM/DCM), CPG quadrupède
│   ├── essaims/           consensus, formation, flocking, enchères
│   ├── apprentissage/     RL : Q-learning, CEM, randomisation de domaine
│   ├── planification/     planificateur STRIPS + instructions en langage naturel (Claude)
│   ├── embarque/          C++17 pour microcontrôleur : PID, codeur, odométrie, HAL
│   └── ros2/              intégration ROS 2 Jazzy : nœuds, launch, Docker
├── docs/architecture.md   couches, dépendances, conventions
└── ROADMAP.md             feuille de route globale et projets « fil rouge »
```

## Les branches

| Branche | Première pierre (implémentée, testée) | Démo | Pistes porteuses |
|---|---|---|---|
| [manipulation](projets/manipulation) | Chaîne DH, jacobienne, IK par moindres carrés amortis, trajectoires quintiques et trapézoïdales, UR5e | prise et dépose avec un UR5e | MoveIt 2, saisie apprise, politiques VLA |
| [navigation](projets/navigation) | Carte d'occupation, gonflage, A* sans coupe de coin, raccourcissement, pure pursuit | traversée d'entrepôt | Hybrid A*, MPC, Nav2, navigation sociale |
| [estimation](projets/estimation) | EKF sur amers, filtre particulaire (localisation globale), simulateur de capteurs | odométrie vs EKF vs MCL | SLAM par graphe, VIO, Gaussian Splatting |
| [commande](projets/commande) | PID anti-emballement, LQR continu/discret sans SciPy, moteur CC, pendule inversé | pendule stabilisé depuis 20° | MPC, commande robuste, filtres de sûreté |
| [vision](projets/vision) | Caméra sténopé + distorsion, homographie DLT, triangulation, look-at | reconstruction stéréo d'un cube | AprilTag, pose 6D, modèles de fondation |
| [drones](projets/drones) | Quadrirotor plan, mélangeur, commande en cascade, jerk minimal | vol par points de passage | SE(3), minimum-snap, PX4 / MAVLink |
| [locomotion](projets/locomotion) | LIPM, point de capture, marche DCM robuste aux poussées, CPG de Hopf, jambe 2 segments | bipède poussé + quadrupède trot → bond | MPC centroidale, RL massivement parallèle, humanoïdes |
| [essaims](projets/essaims) | Laplacien, consensus, formation, boids, allocation par enchères | allocation + hexagone + flocking | CBBA, MAPF, SLAM collaboratif |
| [apprentissage](projets/apprentissage) | Environnements API Gymnasium, Q-learning, CEM, randomisation de domaine | labyrinthe + atteinte robuste | PPO/SAC sur GPU, imitation, VLA |
| [planification](projets/planification) | STRIPS optimal, monde des blocs, instruction FR → objectif → plan (règles ou Claude) | « empile le bleu sur le vert… » | PDDL, TAMP, agents LLM, arbres de comportement |
| [embarque](projets/embarque) | C++17 sans allocation : PID, décodeur quadrature, odométrie, file SPSC, HAL, asservissement | croquis Arduino | micro-ROS, FreeRTOS/Zephyr, FOC, CAN |
| [ros2](projets/ros2) | Paquet `robotique_ros` : simulateur, planificateur, suivi de chemin, launch, RViz, Docker | navigation complète sans Gazebo | Nav2, Gazebo/Isaac, ros2_control, MoveIt 2 |

## Démarrage rapide

```bash
make install        # dépendances + toutes les branches Python en mode éditable
make test           # 98 tests Python + tests C++ (g++/clang)
make demos          # exécute la démo de chaque branche
python projets/navigation/examples/demo_navigation.py --plot   # figures dans outputs/
```

Prérequis : Python ≥ 3.10 (NumPy est la seule dépendance d'exécution), CMake ≥ 3.16
et un compilateur C++17 pour la branche embarquée, Docker pour la branche ROS 2.
Le mode langage naturel de *planification* nécessite en plus `pip install anthropic`.

## Principes

- **Une branche = un dossier autonome** (`src/`, `tests/`, `examples/`, `README.md`,
  `pyproject.toml`), installable seule et ne dépendant que du socle.
- **Logique d'abord, middleware ensuite** : les algorithmes sont testables sans ROS ni
  matériel ; les nœuds ROS 2 et le code microcontrôleur ne sont que des adaptateurs.
- **Chaque pierre est vérifiée** : jacobiennes contrôlées par différences finies, plans
  comparés à la force brute, comportements en boucle fermée simulés dans les tests.
- **Documentation en français, identifiants en anglais**, unités SI, angles en radians.

Détails : [docs/architecture.md](docs/architecture.md) · feuille de route :
[ROADMAP.md](ROADMAP.md) · contribuer : [CONTRIBUTING.md](CONTRIBUTING.md).
