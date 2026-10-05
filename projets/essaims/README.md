# Branche « Essaims » — systèmes multi-robots

> Coordonner des dizaines de robots (AMR, drones, robots sous-marins) sans chef unique :
> se répartir le travail, se déplacer en formation, rester groupés.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `graphs` | Graphes de communication (anneau, complet, disque), laplacien, connectivité algébrique (valeur de Fiedler) |
| `consensus` | Consensus moyen distribué, commande de formation par déplacements relatifs |
| `boids` | Flocking de Reynolds vectorisé (séparation, alignement, cohésion), paramètre d'ordre |
| `assignment` | Allocation de tâches par enchères de Bertsekas, optimale (vérifiée contre la force brute) |

```bash
python projets/essaims/examples/demo_essaim.py
pytest projets/essaims
```

## Pistes porteuses (prochaines pierres)

1. **Allocation dynamique** : CBBA (consensus-based bundle algorithm), tâches séquentielles.
2. **Planification multi-agents sans collision** : CBS, ORCA / vitesses réciproques.
3. **Formations robustes** : graphes rigides, maintien de connectivité, pannes de robots.
4. **SLAM collaboratif** et partage de cartes (lien *estimation*).
5. **Apprentissage multi-agents** (MARL) et comportements émergents.
6. **Intégration** : ROS 2 multi-robots (namespaces, DDS), essaims de drones (lien *drones*).
