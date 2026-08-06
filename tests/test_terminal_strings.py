import random

import numpy as np
import torch
from eckity.base.untyped_functions import f_add
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree

from aux_func import program_to_labels
from eckity_bert_gp import BertGPEckity, BertMutation


def test_program_to_labels_preserves_numpy_string_terminal():
    tree = Tree(
        function_set=[f_add],
        terminal_set=["move"],
        tree=[TerminalNode(np.str_("move"))],
    )

    labels = program_to_labels(tree, [])

    assert labels == ["move"]
    assert "const" not in labels


def test_adapter_preserves_named_terminals_across_two_mutations(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    tree = Tree(
        function_set=[f_add],
        terminal_set=["x", "y"],
        tree=[FunctionNode(f_add), TerminalNode("x"), TerminalNode("y")],
    )
    bert_model = BertMutation(
        operators_list=np.array([f_add.__name__]),
        constant_names=["x", "y"],
        get_fitness_func=lambda individual: 0.0,
        batch_size=64,
        epsilon_greedy=1.0,
        word_embedding_dim=4,
        context_size=8,
        n_layers=1,
        n_attention_heads=1,
        internal_size=4,
        full_trajectory_query=False,
        function_mappings={f_add.__name__: f_add},
        allow_constant_terminals=False,
    )
    mutation = BertGPEckity(
        bert_model=bert_model,
        probability=1.0,
        node_probability=0.5,
        max_trajectory_length=4,
    )
    masks = iter(
        [
            [np.array([False, True, False])],
            [np.array([False, False, True])],
        ]
    )
    monkeypatch.setattr(mutation, "_sample_masks", lambda individuals: next(masks))

    mutation.attempt_operator([tree], attempt_num=0)
    first_terminal = tree.tree[1].value
    assert type(first_terminal) is str

    mutation.attempt_operator([tree], attempt_num=0)

    terminal_values = [
        node.value for node in tree.tree if isinstance(node, TerminalNode)
    ]
    labels = program_to_labels(tree, [])
    assert all(type(value) is str for value in terminal_values)
    assert set(terminal_values) <= {"x", "y"}
    assert "const" not in labels
