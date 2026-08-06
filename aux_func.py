from typing import List

import numpy as np
from eckity.genetic_encodings.gp import Tree, FunctionNode, TerminalNode


def prefix_to_postfix(tokens, precedence):
    stack = []
    tokens = tokens[::-1]
    original_indexes = list(range(len(tokens)))[::-1]

    for original_index, t in zip(original_indexes, tokens):
        if t == 'const' or t.startswith('x'):
            arity = 0
        else:
            _, arity = precedence[t]

        if arity > 0:
            operators = [stack.pop() for _ in range(arity)]
            temp_exp = tuple()

            for op in operators:
                temp_exp += op

            temp_exp += ((original_index, t),)
            stack.append(temp_exp)

        else:
            stack.append(((original_index, t),))

    assert len(stack) == 1
    indexes = [x[0] for x in stack[0]]
    return indexes


def prefix_to_infix(tokens, precedence):
    stack = []
    tokens = tokens[::-1]
    original_indexes = list(range(len(tokens)))[::-1]

    for original_index, t in zip(original_indexes, tokens):
        if t == 'const' or t.startswith('x'):
            arity = 0
        else:
            _, arity = precedence[t]

        if arity > 0:

            operators = [stack.pop() for _ in range(arity)]
            temp_exp = tuple()

            if len(operators) == 1:
                temp_exp += ((original_index, t),)
                temp_exp += operators[0]
            elif len(operators) == 2:
                temp_exp += operators[0]
                temp_exp += ((original_index, t),)
                temp_exp += operators[1]

            else:
                raise ValueError("Invalid arity")

            stack.append(temp_exp)

        else:
            stack.append(((original_index, t),))

    assert len(stack) == 1
    indexes = [x[0] for x in stack[0]]
    return indexes


def get_inverse_mapping(origin_to_target_mapping: np.ndarray) -> np.ndarray:
    inverse_mapping = np.zeros_like(origin_to_target_mapping)
    for index, target_mapping in enumerate(origin_to_target_mapping):
        inverse_mapping[target_mapping] = index
    return inverse_mapping


def program_to_labels(program: Tree, mask_indexes) -> List[str]:
    labels = []
    for index, node in enumerate(program.tree):
        if index < len(mask_indexes) and mask_indexes[index]:
            labels.append('<mask>')
        elif type(node) is FunctionNode:
            labels.append(node.function.__name__)
        elif type(node) is TerminalNode:
            if isinstance(node.value, str):
                labels.append(node.value)
            else:
                labels.append('const')

        else:
            raise ValueError(f"Node type {type(node)} not supported")
    return labels
