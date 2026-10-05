# Branche « Apprentissage » — robot learning

> Apprendre des comportements plutôt que les programmer : apprentissage par renforcement,
> par imitation, et modèles de fondation pour la robotique.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `envs` | Environnements à l'API Gymnasium : labyrinthe discret, atteinte de cible par une masse ponctuelle avec **randomisation de domaine** |
| `agents` | Q-learning tabulaire, politique linéaire, évaluation reproductible, méthode de l'entropie croisée (CEM) |

La démo montre que la CEM retrouve d'elle-même un correcteur de type PD
(gain positif sur l'erreur, négatif sur la vitesse), robuste aux variations de masse.

```bash
python projets/apprentissage/examples/demo_apprentissage.py
pytest projets/apprentissage
```

## Pistes porteuses (prochaines pierres)

1. **Passage à l'échelle** : Gymnasium + MuJoCo / Isaac Lab, PPO et SAC (Stable-Baselines3, CleanRL),
   simulation massivement parallèle sur GPU.
2. **Sim-to-real** : randomisation de domaine étendue, identification, adaptation en ligne.
3. **Apprentissage par imitation** : clonage comportemental, ACT, Diffusion Policy, jeux de données
   LeRobot / Open X-Embodiment.
4. **Modèles VLA** (vision-langage-action) : RT-2, OpenVLA, π0 — politiques généralistes pilotées par le langage.
5. **Apprentissage sûr** : contraintes, filtres de sûreté (lien *commande*), RL hors ligne.
6. **Modèles du monde** pour la planification et l'évaluation de politiques.
