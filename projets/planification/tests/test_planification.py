import json
from types import SimpleNamespace

import pytest

from planification import (
    ClaudeGoalParser,
    GoalParsingError,
    KeywordGoalParser,
    blocks_world,
    plan,
    plan_from_instruction,
    validate_goal,
    validate_plan,
)

SUSSMAN = {"C": "A", "A": "table", "B": "table"}


def test_sussman_anomaly_optimal_plan():
    problem = blocks_world(SUSSMAN, [("sur", "A", "B"), ("sur", "B", "C")])
    steps = plan(problem)
    assert [str(a) for a in steps] == [
        "depiler(C, A)",
        "poser(C)",
        "prendre(B)",
        "empiler(B, C)",
        "prendre(A)",
        "empiler(A, B)",
    ]
    assert validate_plan(problem, steps)
    assert not validate_plan(problem, steps[1:])


def test_goal_already_satisfied_gives_empty_plan():
    problem = blocks_world(SUSSMAN, [("sur", "C", "A")])
    assert plan(problem) == []


def test_unreachable_goal_returns_none():
    problem = blocks_world({"A": "table", "B": "table"}, [("sur", "A", "B"), ("sur", "B", "A")])
    assert plan(problem) is None


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Mets le cube rouge sur le cube bleu", {("sur", "rouge", "bleu")}),
        ("pose le vert sur la table", {("sur_table", "vert")}),
        (
            "Empile le bloc bleu sur le vert, puis le rouge sur le bleu",
            {("sur", "bleu", "vert"), ("sur", "rouge", "bleu")},
        ),
    ],
)
def test_keyword_parser(instruction, expected):
    assert KeywordGoalParser().parse(instruction, ["rouge", "bleu", "vert"]) == expected


def test_keyword_parser_rejects_unknown_objects_and_nonsense():
    with pytest.raises(GoalParsingError, match="inconnu"):
        KeywordGoalParser().parse("mets le cube violet sur le bleu", ["rouge", "bleu"])
    with pytest.raises(GoalParsingError):
        KeywordGoalParser().parse("fais la vaisselle", ["rouge", "bleu"])


def test_validate_goal_detects_conflicts():
    with pytest.raises(GoalParsingError, match="lui-même"):
        validate_goal([("sur", "a", "a")], ["a"])
    with pytest.raises(GoalParsingError, match="contradictoire"):
        validate_goal([("sur", "a", "b"), ("sur_table", "a")], ["a", "b", "table"])
    with pytest.raises(GoalParsingError, match="deux cubes"):
        validate_goal([("sur", "a", "c"), ("sur", "b", "c")], ["a", "b", "c"])


def test_plan_from_instruction_end_to_end():
    scene = {"rouge": "bleu", "bleu": "table", "vert": "table"}
    goal, steps = plan_from_instruction("mets le bleu sur le vert et le rouge sur le bleu", scene)
    assert goal == {("sur", "bleu", "vert"), ("sur", "rouge", "bleu")}
    assert len(steps) == 6


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="text", text=self.text)] if self.text is not None else []
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


def fake_client(text, stop_reason="end_turn"):
    messages = FakeMessages(text, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_claude_parser_builds_structured_request_and_validates_answer():
    answer = json.dumps(
        {"faits": [{"cube": "rouge", "support": "vert"}, {"cube": "vert", "support": "table"}]}
    )
    client, messages = fake_client(answer)
    goal = ClaudeGoalParser(client).parse("Range le rouge sur le vert", ["vert", "rouge"])
    assert goal == {("sur", "rouge", "vert"), ("sur_table", "vert")}
    request = messages.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    schema = request["output_config"]["format"]["schema"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    cube_names = schema["properties"]["faits"]["items"]["properties"]["cube"]["enum"]
    assert cube_names == ["rouge", "vert"]


@pytest.mark.parametrize(
    ("text", "stop_reason"),
    [
        (None, "refusal"),
        ("{}", "end_turn"),
        ("pas du json", "end_turn"),
        ('{"faits": []', "max_tokens"),
    ],
)
def test_claude_parser_errors(text, stop_reason):
    client, _ = fake_client(text, stop_reason)
    with pytest.raises(GoalParsingError):
        ClaudeGoalParser(client).parse("mets le rouge sur le vert", ["rouge", "vert"])
