# Branche « Embarqué » — logiciel temps réel sur microcontrôleur

> Le lien entre les algorithmes et le matériel : lire les capteurs, piloter les moteurs,
> boucler la commande à haute fréquence sur des cibles contraintes.

## Première pierre (implémentée et testée)

Bibliothèque C++17 *header-only*, **sans allocation dynamique ni exception** :

| En-tête | Contenu |
|---|---|
| `pid.hpp` | PID jumeau de la version Python (*commande*) : dérivée filtrée sur la mesure, anti-emballement |
| `quadrature_decoder.hpp` | Décodage x4 d'un codeur incrémental par table, détection des impulsions manquées |
| `diff_drive_odometry.hpp` | Odométrie différentielle (intégration au point milieu) |
| `spsc_queue.hpp` | File sans verrou interruption → boucle principale |
| `hal.hpp` | Interfaces matérielles `IMotor` / `IEncoder` |
| `motor_speed_controller.hpp` | Asservissement en vitesse complet, testé sur PC contre un moteur simulé |

Un croquis Arduino (`examples/arduino/vitesse_moteur`) montre l'implémentation de la HAL sur cible.

```bash
cmake -S projets/embarque -B build/embarque && cmake --build build/embarque
ctest --test-dir build/embarque --output-on-failure
```

## Pistes porteuses (prochaines pierres)

1. **micro-ROS** sur ESP32 / STM32 : publier odométrie et recevoir `cmd_vel` depuis la branche *ros2*.
2. **RTOS** : FreeRTOS ou Zephyr, tâches périodiques, mesure de gigue.
3. **Bus de terrain** : CAN / CAN-FD (variateurs), EtherCAT pour les bras.
4. **Commande moteur avancée** : FOC (commande vectorielle) des moteurs brushless.
5. **Fusion IMU embarquée** (filtre de Madgwick / Mahony) pour drones et robots à pattes.
6. **Sûreté de fonctionnement** : chien de garde, arrêt d'urgence, tests matériels en boucle (HIL).
