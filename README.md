# BERT Mutation for EC-KitY Genetic Programming

`eckity-bert-gp` provides the BERT mutation operator for tree-based genetic programming in [EC-KitY](https://github.com/EC-KitY/EC-KitY).

The operator is described in **“BERT Mutation: Deep Transformer Model for Masked Uniform Mutation in Genetic Programming”**, Mathematics 2025, 13(5), 779 ([paper](https://doi.org/10.3390/math13050779)). It masks selected GP-tree nodes and uses a compact BERT masked-language model to sample replacements that preserve the required node arity.

## Installation

```bash
pip install eckity-bert-gp
```

## Public API

```python
from eckity_bert_gp import BertMutation, BERTUniformMutation
```

`BertMutation` owns and trains the BERT policy. `BERTUniformMutation` adapts that policy to EC-KitY's genetic-operator interface.

## Usage

The BERT model needs the function names, terminal names, fitness callback, and mappings back to the EC-KitY functions:

```python
import numpy as np
from eckity.base.untyped_functions import f_add, f_div, f_mul, f_sub
from eckity_bert_gp import BertMutation, BERTUniformMutation

function_set = [f_add, f_sub, f_mul, f_div]
terminal_set = ["x", "y", "z"]
function_mappings = {function.__name__: function for function in function_set}

bert_model = BertMutation(
    operators_list=np.array(list(function_mappings)),
    constant_names=terminal_set,
    get_fitness_func=evaluator.evaluate_individual,
    context_size=256,
    word_embedding_dim=20,
    n_layers=1,
    n_attention_heads=1,
    function_mappings=function_mappings,
    higher_is_better=False,
)

bert_mutation = BERTUniformMutation(
    bert_model=bert_model,
    probability=1.0,
    node_probability=0.1,
)
```

Add `bert_mutation` to the EC-KitY subpopulation's `operators_sequence`.

- `get_fitness_func` accepts an EC-KitY GP tree and returns its fitness.
- `function_mappings` maps each function name used by BERT back to the callable stored in GP trees.
- Terminal mappings default to the names supplied in `constant_names`.
- `probability` controls whether the EC-KitY mutation operator runs.
- `node_probability` controls the probability of masking each tree node.
- `context_size` must be large enough for the longest tree representation expected during evolution.

The model is initialized locally from `BertConfig`; installing or constructing the operator does not download pretrained model weights.

## Compatibility

- Python 3.10 or newer
- EC-KitY 0.4.x
- NumPy 2.0.2 or newer
- SciPy 1.13.0 or newer
- PyTorch 2.7.1 or newer
- Transformers 4.50.0 or newer
- scikit-learn 1.5.0 or newer

These bounds are compatible with `eckity-dnc` and `eckity-bert-ga`; none of the three packages directly depends on another operator package.

## Repository experiment

The repository includes the paper's experiment runner and datasets. They are development resources and are not included in the wheel.

Install the development dependencies and run the symbolic-regression example:

```bash
python -m pip install -e ".[dev]"
python runner.py
```

Additional benchmark data is stored under `data/`, and Artificial Ant maps are stored under `ant_opt/`.

## Development

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --resolution lowest-direct
uv run pytest
uv build
```

Release preparation and manual PyPI upload commands are documented in [`RELEASING.md`](RELEASING.md).

## License

This project is licensed under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
