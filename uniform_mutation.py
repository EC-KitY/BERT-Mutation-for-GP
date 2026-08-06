from typing import Any, List, Tuple

from eckity.base.utils import arity
from overrides import override
import numpy as np
from eckity.genetic_encodings.gp import Tree, TreeNode, FunctionNode, TerminalNode
from eckity.genetic_operators import FailableOperator
import random

from bert_mutation import BertMutation
from aux_func import program_to_labels


class UniformNodeMutation(FailableOperator):
    def __init__(
            self,
            probability: float = 1.0,
            node_probability: float = 0.1,
            events=None,
            attempts=1,
    ):
        super().__init__(
            probability=probability, arity=1, events=events, attempts=attempts
        )
        self.node_probability = node_probability

    @override
    def attempt_operator(
            self, payload: Any, attempt_num: int
    ) -> Tuple[bool, Any]:
        """
        Perform subtree mutation: select a subtree at random
        to be replaced by a new, randomly generated subtree.

        Returns
        -------
        Tuple[bool, Any]
            A tuple containing a boolean indicating whether the operator was
            successful and a list of the individuals.
        """
        individuals: List[Tree] = payload
        uniform_masks = self._sample_masks(individuals)
        for ind, mask in zip(individuals, uniform_masks):
            for i, node in enumerate(ind.tree):
                if mask[i]:
                    replacement = self._get_node_replacement(ind, node)
                    ind.tree[i] = replacement

        self.applied_individuals = individuals
        return True, individuals

    def _sample_masks(self, individuals: List[Tree]):
        masks = []
        for ind in individuals:
            mask = np.random.choice([True, False], size=len(ind.tree),
                                    p=[self.node_probability, 1 - self.node_probability])
            masks.append(mask)
        return masks

    def _get_node_replacement(self, ind: Tree, node: TreeNode):
        if type(node) is FunctionNode:
            cur_arity = node.n_args
            relevant_functions = [func for func in ind.function_set if arity(func) == cur_arity]
            func = random.choice(relevant_functions)
            return FunctionNode(func)


        elif type(node) is TerminalNode:
            return ind.random_terminal(node_type=node.node_type)
        else:
            raise ValueError(f"Node type {type(node)} not supported")


class BertGPEckity(FailableOperator):
    def __init__(
            self,
            bert_model: BertMutation,
            probability: float = 1.0,
            node_probability: float = 0.1,
            max_trajectory_length=100,
            events=None,
            attempts=1,
    ):
        super().__init__(
            probability=probability, arity=1, events=events, attempts=attempts
        )
        self.node_probability = node_probability
        self.bert_model = bert_model
        self.max_trajectory_length = max_trajectory_length

    @override
    def attempt_operator(
            self, payload: Any, attempt_num: int
    ) -> Tuple[bool, Any]:
        """
        Perform subtree mutation: select a subtree at random
        to be replaced by a new, randomly generated subtree.

        Returns
        -------
        Tuple[bool, Any]
            A tuple containing a boolean indicating whether the operator was
            successful and a list of the individuals.
        """
        individuals: List[Tree] = payload
        uniform_masks = self._sample_masks(individuals)
        assert len(individuals) == 1
        individual = individuals[0]
        mutation_mask = uniform_masks[0]

        allowed_functions = np.array(list(self.bert_model.function_mappings.keys()))
        allowed_functions_arity = np.array([arity(func) for func in list(self.bert_model.function_mappings.values())])
        functions_mutation_mask = np.array([type(node) is FunctionNode for node in individual.tree])
        masked_functions = np.where(functions_mutation_mask & mutation_mask)[0]
        masked_variables = np.where(~functions_mutation_mask & mutation_mask)[0]

        if len(masked_functions) > 0:
            program_labels = program_to_labels(individual, mutation_mask & functions_mutation_mask)
            self.bert_model.mutate(program_labels, allowed_functions, individual,
                                   masked_functions, self._get_arity_of_masked_nodes(individual, mutation_mask),
                                   allowed_functions_arity)

        if len(masked_variables) > 0:
            program_labels = program_to_labels(individual, mutation_mask & ~functions_mutation_mask)
            self.bert_model.mutate(program_labels, self.bert_model.terminals, individual,
                                   masked_variables, None, None, terminal_traj=True)

        self.applied_individuals = individuals
        return True, individuals

    def _sample_masks(self, individuals: List[Tree]):
        masks = []
        for ind in individuals:
            if len(ind.tree) * self.node_probability < self.max_trajectory_length:
                mask = np.random.choice([True, False], size=len(ind.tree),
                                        p=[self.node_probability, 1 - self.node_probability])
            else:
                mask = np.random.choice([True, False], size=len(ind.tree),
                                        p=[self.max_trajectory_length / len(ind.tree),
                                           1 - self.max_trajectory_length / len(ind.tree)])
            masks.append(mask)
        return masks

    def _get_arity_of_masked_nodes(self, ind: Tree, mask: np.ndarray):
        arities = []
        for i, node in enumerate(ind.tree):
            if mask[i] and type(node) is FunctionNode:
                arities.append(node.n_args)
        return np.array(arities)

    def _get_node_replacement(self, ind: Tree, node: TreeNode):
        if type(node) is FunctionNode:
            cur_arity = node.n_args
            relevant_functions = [func for func in ind.function_set if arity(func) == cur_arity]
            func = random.choice(relevant_functions)
            return FunctionNode(func)


        elif type(node) is TerminalNode:
            return ind.random_terminal(node_type=node.node_type)
        else:
            raise ValueError(f"Node type {type(node)} not supported")
