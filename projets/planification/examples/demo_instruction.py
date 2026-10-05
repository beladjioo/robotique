"""Démo : une instruction en français devient un plan d'actions pour un bras robotique.

    python projets/planification/examples/demo_instruction.py
    python projets/planification/examples/demo_instruction.py --llm   # analyse par Claude
    python projets/planification/examples/demo_instruction.py "pose le vert sur la table"

L'option --llm nécessite `pip install anthropic` et une clé (ANTHROPIC_API_KEY ou `ant auth login`).
"""

import argparse

from planification import (
    ClaudeGoalParser,
    GoalParsingError,
    KeywordGoalParser,
    plan_from_instruction,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "instruction", nargs="?", default="Empile le cube bleu sur le vert, puis le rouge sur le bleu"
)
parser.add_argument(
    "--llm", action="store_true", help="utiliser Claude pour comprendre l'instruction"
)
args = parser.parse_args()

scene = {"rouge": "bleu", "bleu": "table", "vert": "table", "jaune": "vert"}
print("Scène :", ", ".join(f"{cube} sur {support}" for cube, support in scene.items()))
print("Instruction :", args.instruction)

goal_parser = ClaudeGoalParser() if args.llm else KeywordGoalParser()
try:
    goal, steps = plan_from_instruction(args.instruction, scene, goal_parser)
except GoalParsingError as error:
    raise SystemExit(f"Instruction non comprise : {error}") from error

print("Objectif :", sorted(goal))
if steps is None:
    print("Objectif irréalisable.")
else:
    for k, action in enumerate(steps, 1):
        print(f"  {k}. {action}")
