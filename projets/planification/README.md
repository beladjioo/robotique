# Branche « Planification » — tâches, raisonnement et langage

> Passer d'une intention (« range les cubes par couleur ») à une séquence d'actions
> exécutables et vérifiables par un robot.

## Première pierre (implémentée et testée)

| Module | Contenu |
|---|---|
| `strips` | Planificateur symbolique STRIPS (A*, plan optimal), heuristique, validation de plan par rejeu |
| `blocks` | Domaine « monde des blocs » pour bras + pince (prendre, poser, empiler, dépiler) |
| `language` | Instruction en français → objectif validé → plan. Analyseur à règles hors ligne, ou analyseur **Claude** (sortie JSON contrainte par schéma, repli automatique en cas de refus) |

Principe de sûreté (*LLM + planificateur*) : le modèle de langage ne fait que traduire
l'intention en objectif ; un planificateur symbolique produit un plan dont chaque
action est vérifiée avant exécution.

```bash
python projets/planification/examples/demo_instruction.py
pip install anthropic && python projets/planification/examples/demo_instruction.py --llm \
    "construis une tour avec le jaune en bas puis le rouge"
pytest projets/planification   # les tests du mode Claude utilisent un faux client, sans réseau
```

## Pistes porteuses (prochaines pierres)

1. **PDDL** : import/export de domaines, planificateurs externes (Fast Downward), domaines temporels.
2. **TAMP** (planification tâche-mouvement) : coupler les actions symboliques aux trajectoires de *manipulation*.
3. **Ancrage perceptif** : construire l'état symbolique à partir de la *vision* (détection des cubes).
4. **Agents LLM** avec outils robotiques (appel d'outils, vérification, replanification sur échec).
5. **Arbres de comportement** (BehaviorTree.CPP, py_trees) pour l'exécution réactive sous ROS 2.
6. **Modèles VLA** de bout en bout en complément de la planification explicite (branche *apprentissage*).
