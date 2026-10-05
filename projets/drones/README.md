# Branche « Drones » — robotique aérienne

> Faire voler des multirotors de façon autonome : stabilisation, suivi de trajectoire,
> inspection, livraison, essaims de drones.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `quadrotor` | Dynamique du quadrirotor plan (y, z, roulis), mélangeur moteurs saturé |
| `controller` | Commande en cascade : PD position + anticipation d'accélération → consigne d'inclinaison → PD attitude |
| `trajectory` | Points de passage reliés par des segments à jerk minimal (position, vitesse, accélération) |

```bash
python projets/drones/examples/demo_vol.py --plot
pytest projets/drones
```

## Pistes porteuses (prochaines pierres)

1. **Passage en 3D** : dynamique 6 DDL, commande géométrique sur SE(3) (Lee et al.), quaternions.
2. **Trajectoires minimum-snap** (Mellinger & Kumar) exploitant la platitude différentielle.
3. **Estimation** : fusion IMU + flux optique + baromètre (lien *estimation*).
4. **Autopilotes** : interface PX4 / ArduPilot via MAVLink et uXRCE-DDS (ROS 2).
5. **Vol agile et apprentissage** : MPC, politiques RL type course de drones (sim-to-real).
6. **Essaims aériens** : formation et allocation de tâches (lien *essaims*).
