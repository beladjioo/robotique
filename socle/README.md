# Socle — `robocore`

Bibliothèque mathématique minimale partagée par toutes les branches Python du dépôt.
Elle ne dépend que de NumPy et ne contient **aucune logique métier** : uniquement des
briques génériques et testées.

| Module | Contenu |
|---|---|
| `angles` | `wrap_angle`, `angle_diff` |
| `transforms` | SE(2), SE(3), rotations élémentaires, `so3_exp` / `so3_log`, RPY, quaternions |
| `integrators` | `euler_step`, `rk4_step` (commande bloquée sur le pas) |
| `numerics` | `numerical_jacobian`, `expm` (exponentielle de matrice) |
| `profiles` | `min_jerk` (profil polynomial d'ordre 5) |

## Conventions

- Unités SI, angles en radians, repères directs.
- Quaternions au format `(w, x, y, z)`.
- `T_a_b` : pose du repère *b* exprimée dans le repère *a* (`p_a = T_a_b @ p_b`).

## Règle d'or

Une fonction n'entre dans le socle que si **au moins deux branches** en ont besoin.
