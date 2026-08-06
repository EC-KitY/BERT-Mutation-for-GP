import ast
from importlib.metadata import requires, version
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

from eckity.genetic_operators import TournamentSelection


ROOT = Path(__file__).parents[1]


def test_release_metadata_and_public_imports():
    import eckity_bert_gp
    from eckity_bert_gp import BertGPEckity, BertMutation

    assert version("eckity") == "0.4.2"
    assert version("eckity-bert-gp") == "0.1.1"
    assert BertMutation is not None
    assert BertGPEckity is not None
    assert eckity_bert_gp.__all__ == ["BertMutation", "BertGPEckity"]
    assert not hasattr(eckity_bert_gp, "BERT" + "UniformMutation")
    assert "higher_is_better" not in signature(BertMutation).parameters

    dependencies = [requirement.lower() for requirement in requires("eckity-bert-gp") or []]
    assert not any(requirement.startswith("scikit-learn") for requirement in dependencies)
    assert any(requirement.startswith("scipy") for requirement in dependencies)


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


def test_policy_reward_uses_individual_fitness_direction():
    from eckity_bert_gp import BertMutation

    bert = BertMutation.__new__(BertMutation)
    bert.diff_reward = True

    minimizer = SimpleNamespace(higher_is_better=False)
    maximizer = SimpleNamespace(higher_is_better=True)

    assert bert._policy_reward(3.0, 2.0, minimizer) == -1.0
    assert bert._policy_reward(2.0, 3.0, minimizer) == 1.0
    assert bert._policy_reward(2.0, 3.0, maximizer) == -1.0
    assert bert._policy_reward(3.0, 2.0, maximizer) == 1.0


def test_legacy_adapter_name_is_removed_from_source_and_docs():
    legacy_name = "BERT" + "UniformMutation"
    checked_files = [
        ROOT / "README.md",
        ROOT / "runner.py",
        ROOT / "uniform_mutation.py",
        *sorted((ROOT / "eckity_bert_gp").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]

    for path in checked_files:
        assert legacy_name not in path.read_text(encoding="utf-8")
