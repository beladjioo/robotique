# Branche « Navigation » — robotique mobile autonome

> Amener un robot mobile (AMR d'entrepôt, robot de service, rover) d'un point A à un
> point B, sans collision, dans un environnement connu puis inconnu.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `grid` | Carte d'occupation (convention ROS), conversion monde/cellule, gonflage des obstacles |
| `planners` | A* 4/8-connexe sans coupe de coin, visibilité (Bresenham), raccourcissement de chemin, rééchantillonnage |
| `models` | Cinématique différentielle (roues ↔ torseur), intégration exacte du modèle unicycle |
| `controllers` | Suivi de chemin *pure pursuit* avec ralentissement à l'arrivée et saturation en lacet |
| `demo_maps` | Carte d'entrepôt partagée par les démos, les tests et la branche ROS 2 |

```bash
python projets/navigation/examples/demo_navigation.py --plot
pytest projets/navigation
```

## Pistes porteuses (prochaines pierres)

1. **Planification avancée** : Hybrid A* / State Lattice (contraintes non holonomes), RRT*, planification
   sur cartes de coût multicouches.
2. **Évitement local** : DWA, TEB, MPC — obstacles dynamiques et humains (navigation sociale).
3. **Couplage SLAM** : naviguer dans une carte construite en ligne (branche *estimation*).
4. **Flottes d'AMR** : gestion de trafic, planification multi-agents (CBS) — branche *essaims*.
5. **Navigation apprise** : politiques bout-en-bout, modèles de fondation pour la navigation
   guidée par le langage (« va à la cuisine »).
6. **Intégration** : Nav2 via la branche *ros2* (déjà amorcée avec un nœud pure pursuit).
