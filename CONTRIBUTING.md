# Contribuer

## Mise en place

```bash
make install   # pip install -r requirements-dev.txt + paquets en mode éditable
make test      # pytest + ctest
make lint      # ruff check + ruff format --check
```

## Flux de travail git

- `main` reste toujours vert (CI).
- Une branche git par sujet, préfixée par la branche robotique concernée :
  `navigation/hybrid-astar`, `embarque/micro-ros`, `ros2/ci-jazzy`…
- Une pull request = un sujet, avec tests et mise à jour du README de la branche.

## Règles de code

- **Langue** : documentation, docstrings et messages en français ; identifiants en anglais.
- **Unités SI**, angles en radians, repères directs ; voir [docs/architecture.md](docs/architecture.md).
- **Dépendances** : une branche ne dépend que de `numpy` et du socle `robocore`. Toute nouvelle
  dépendance lourde (SciPy, PyTorch, MuJoCo…) doit rester optionnelle (`optional-dependencies`).
- **Tests obligatoires** : chaque fonction publique est couverte. Privilégier les tests de
  propriétés physiques (conservation, convergence, comparaison analytique / différences finies)
  et les simulations en boucle fermée à graine fixe.
- **Socle** : une fonction n'entre dans `robocore` que si au moins deux branches en ont besoin.
- **Embarqué** : C++17, pas d'allocation dynamique ni d'exception, compilation sans avertissement
  avec `-Wall -Wextra -Wpedantic -Wconversion`.

## Ajouter une branche

1. Créer `projets/<nom>/` avec `pyproject.toml`, `src/<nom>/`, `tests/`, `examples/`, `README.md`
   (première pierre + pistes porteuses), sur le modèle d'une branche existante.
2. Ajouter `projets/<nom>/src` aux listes `pythonpath` et `src` du `pyproject.toml` racine.
3. Ajouter `<nom>` à `PY_PROJECTS` dans le `Makefile`.
4. Référencer la branche dans le `README.md` et le `ROADMAP.md`.
