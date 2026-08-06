import random

import numpy as np
import torch
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.base.untyped_functions import f_add, f_sub
from eckity.creators.gp_creators.half import HalfCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_encodings.gp import TerminalNode
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.subpopulation import Subpopulation

from eckity_bert_gp import BertGPEckity, BertMutation


class TinySymbolicRegressionEvaluator(SimpleIndividualEvaluator):
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, -1.0])

    def evaluate_individual(self, individual):
        prediction = np.asarray(individual.execute(x=self.x, y=self.y), dtype=float)
        return float(np.mean(np.abs(prediction - (self.x + self.y))))


def test_tiny_symbolic_regression_evolves_and_trains_bert(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    evaluator = TinySymbolicRegressionEvaluator()
    functions = [f_add, f_sub]
    bert_model = BertMutation(
        operators_list=np.array([function.__name__ for function in functions]),
        constant_names=["x", "y"],
        get_fitness_func=evaluator.evaluate_individual,
        batch_size=2,
        learning_rate=1e-3,
        epsilon_greedy=1.0,
        word_embedding_dim=4,
        context_size=32,
        n_layers=1,
        n_attention_heads=1,
        internal_size=4,
        full_trajectory_query=False,
        function_mappings={function.__name__: function for function in functions},
        allow_constant_terminals=False,
    )
    assert bert_model.vocab_size == len(bert_model.token_to_id)
    assert bert_model.mask_id == bert_model.token_to_id["<mask>"]
    assert not hasattr(bert_model, "token_encoder")
    mutation = BertGPEckity(
        bert_model=bert_model,
        probability=1.0,
        node_probability=0.5,
        max_trajectory_length=16,
    )
    evolution = SimpleEvolution(
        population=Subpopulation(
            creators=HalfCreator(
                init_depth=(1, 2),
                terminal_set=["x", "y"],
                function_set=functions,
                erc_range=None,
            ),
            population_size=4,
            evaluator=evaluator,
            higher_is_better=False,
            elitism_rate=0.5,
            operators_sequence=[mutation],
            selection_methods=[TournamentSelection(tournament_size=2)],
        ),
        max_workers=1,
        max_generation=4,
        random_seed=7,
    )

    evolution.evolve()

    individuals = evolution.population.sub_populations[0].individuals
    best = evolution.best_of_run_
    best_output = np.asarray(
        best.execute(x=evaluator.x, y=evaluator.y),
        dtype=float,
    )
    terminal_values = [
        node.value
        for individual in individuals
        for node in individual.tree
        if isinstance(node, TerminalNode)
    ]
    assert evolution.generation_num == 4
    assert len(individuals) == 4
    assert terminal_values
    assert all(type(value) is str for value in terminal_values)
    assert set(terminal_values) <= {"x", "y"}
    assert np.isfinite(best.get_pure_fitness())
    assert np.all(np.isfinite(best_output))
    assert bert_model.optimizer.state
    assert all(torch.isfinite(parameter).all() for parameter in bert_model.model.parameters())
