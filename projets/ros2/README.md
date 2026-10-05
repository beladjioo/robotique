# Branche « ROS 2 » — intégration système

> Assembler les briques des autres branches en un système robotique réel : nœuds,
> messages, transformations, outils (RViz, rosbag), et à terme Nav2 / MoveIt 2.

## Première pierre

Paquet `ament_python` **robotique_ros** (ROS 2 Jazzy, LTS jusqu'en 2029) :

| Nœud | Rôle | Bibliothèque utilisée |
|---|---|---|
| `unicycle_sim` | Simulateur cinématique : `cmd_vel` → `odom` + TF `odom → base_link` | *navigation* (`unicycle_step`) |
| `path_planner` | Publie la carte (`map`) et un chemin A* (`plan`) en QoS persistante | *navigation* (`astar`, `shortcut_path`) |
| `pure_pursuit` | Suit `plan` à partir de `odom`, publie `cmd_vel` | *navigation* (`PurePursuit`) |

Architecture : **les nœuds sont de fines couches d'adaptation**. Toute la logique vit dans
les bibliothèques Python des branches, testables sans ROS ; seul `conversions.py`
(pur Python, testé dans la CI principale) fait le pont entre conventions.

```bash
# Avec Docker (aucune installation ROS nécessaire), depuis la racine du dépôt
docker build -f projets/ros2/Dockerfile -t robotique-ros2 .
docker run -it --rm robotique-ros2

# Ou dans un espace de travail ROS 2 Jazzy existant
pip install -e socle -e projets/navigation
ln -s "$PWD/projets/ros2/robotique_ros" ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch robotique_ros demo_navigation.launch.py rviz:=true
```

> ⚠️ Les nœuds et l'image Docker n'ont pas encore été exécutés dans un environnement
> ROS 2 réel : seule la partie pure Python (`conversions`) est testée par la CI.
> Première tâche de cette branche : valider la démo sous Jazzy et ajouter un job CI
> dans le conteneur `ros:jazzy`.

## Pistes porteuses (prochaines pierres)

1. **Nav2** : remplacer le planificateur maison par un plugin de planificateur/contrôleur Nav2.
2. **Simulation physique** : Gazebo (gz-sim) ou Isaac Sim avec un robot décrit en URDF/Xacro.
3. **ros2_control** : interfaces matérielles reliées à la branche *embarque* via micro-ROS.
4. **MoveIt 2** pour la branche *manipulation* (planification, collisions, servo).
5. **Observabilité** : rosbag2, Foxglove, métriques ; tests d'intégration `launch_testing`.
6. **Sécurité** : SROS2, ROS 2 en temps réel (executors, PREEMPT_RT).
