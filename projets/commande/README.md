# Branche « Commande » — automatique pour la robotique

> Faire suivre une consigne à un système physique de façon stable, précise et robuste :
> moteurs, articulations, véhicules, robots sous-actionnés.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `pid` | PID industriel : dérivée filtrée sur la mesure, saturation, anti-emballement (jumeau C++ dans *embarque*) |
| `lqr` | LQR continu (sous-espace stable de l'hamiltonien) et discret (algorithme de doublement), sans SciPy |
| `linear` | Linéarisation numérique, discrétisation exacte (bloqueur d'ordre zéro), commandabilité |
| `systems` | Moteur à courant continu, pendule inversé sur chariot (non linéaire + linéarisé) |

```bash
python projets/commande/examples/demo_pendule_inverse.py --plot
pytest projets/commande
```

## Pistes porteuses (prochaines pierres)

1. **Commande prédictive (MPC)** avec contraintes : QP (OSQP), MPC non linéaire (acados, CasADi).
2. **Swing-up** du pendule par énergie puis bascule LQR — démonstrateur de commande hybride.
3. **Commande robuste et adaptative** : H∞, commande par modes glissants, estimation de paramètres en ligne.
4. **Identification de systèmes** à partir de données réelles (moindres carrés, SINDy).
5. **Commande apprise** : MPC différentiable, politiques RL avec filtre de sûreté (Control Barrier Functions).
6. **Commande de robots** : couple calculé, impédance/admittance (lien *manipulation*, *locomotion*).
