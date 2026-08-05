import ast
from importlib.metadata import version
from pathlib import Path

from eckity.genetic_operators import TournamentSelection


ROOT = Path(__file__).parents[1]


def test_release_metadata_and_public_imports():
    from eckity_bert_gp import BERTUniformMutation, BertMutation

    assert version("eckity") == "0.4.2"
    assert version("eckity-bert-gp") == "0.1.1"
    assert BertMutation is not None
    assert BERTUniformMutation is not None


def test_runner_uses_eckity_042_selection_api():
    tree = ast.parse((ROOT / "runner.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        assert not (
            call_name in {"TournamentSelection", "ElitismSelection"}
            and any(keyword.arg == "higher_is_better" for keyword in node.keywords)
        )
        for keyword in node.keywords:
            if keyword.arg == "selection_methods" and isinstance(keyword.value, ast.List):
                assert all(not isinstance(item, ast.Tuple) for item in keyword.value.elts)

    assert TournamentSelection(tournament_size=2) is not None
