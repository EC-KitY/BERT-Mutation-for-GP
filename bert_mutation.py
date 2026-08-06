import random

import numpy as np
import torch
from eckity.genetic_encodings.gp import TerminalNode, FunctionNode
from transformers import BertConfig
from transformers import BertForMaskedLM
from torch.optim import Adam
from aux_func import program_to_labels


def convert_arity_to_tensors(allowed_operators, allowed_operators_arity, arity_of_masked_locations,
                             mask_indices):
    # if no arity is provided, assume all operators have the same arity (set as 0)
    if arity_of_masked_locations is None:
        arity_of_masked_locations = torch.zeros(len(mask_indices))
    if allowed_operators_arity is None:
        allowed_operators_arity = torch.zeros(len(allowed_operators))
    arity_of_masked_locations = torch.Tensor(arity_of_masked_locations).type(torch.LongTensor)
    allowed_operators_arity = torch.Tensor(allowed_operators_arity).type(torch.LongTensor)
    return allowed_operators_arity, arity_of_masked_locations


def get_transformed_notation(arity_ndarray, masked_nodes, program_tokens, unmasked_tokens):
    # default order
    mapped_tokens_indices = np.arange(len(unmasked_tokens))
    sorted_mask_order = np.argsort(masked_nodes)
    mapped_tokens = program_tokens
    mapped_mask_arity = arity_ndarray
    mapped_masked_nodes = np.array(masked_nodes)

    return mapped_mask_arity, mapped_masked_nodes, mapped_tokens, mapped_tokens_indices, sorted_mask_order


class BertMutation:
    # todo: when a program is too long, take only the last 2048 tokens

    def __init__(self, operators_list, constant_names, get_fitness_func, batch_size=64, learning_rate=1e-3,
                 adam_decay=0,
                 epsilon_greedy=0.01, word_embedding_dim=120, context_size=2048, n_layers=3, n_attention_heads=3,
                 internal_size=128, clip_grad_norm=1.0, full_trajectory_query=True, diff_reward=True,
                 function_mappings=None, terminals_mappings=None, allow_constant_terminals=True):

        if constant_names is None:
            constant_names = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        if allow_constant_terminals:
            self.terminals = np.array(constant_names + ['const'])
        else:
            self.terminals = np.array(constant_names)

        vocabulary = list(operators_list) + ['<mask>'] + list(self.terminals)
        if not all(isinstance(token, str) for token in vocabulary):
            raise TypeError("BERT mutation tokens must be strings")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("BERT mutation tokens must be unique")

        # Keep token IDs deterministic without an external encoder.
        self.vocabulary = tuple(sorted(vocabulary))
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.vocabulary)
        }
        self.vocab_size = len(self.vocabulary)
        self.mask_id = self.token_to_id['<mask>']
        self.bert_config = {
            'vocab_size': self.vocab_size,
            'hidden_size': word_embedding_dim,
            'num_hidden_layers': n_layers,
            'num_attention_heads': n_attention_heads,
            'intermediate_size': internal_size,
            'max_position_embeddings': context_size
        }

        self.model = BertForMaskedLM(BertConfig(**self.bert_config)).to(self.device)
        self.action_probabilities = []
        self.rewards = []
        self.batch_size = batch_size
        self.trajectory_probabilities = []
        self.n_features = len(constant_names)
        self.rewards = []
        self.get_fitness_func = get_fitness_func
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=adam_decay)
        self.epsilon_greedy = epsilon_greedy
        self.clip_grad_norm = clip_grad_norm
        self.full_trajectory_query = full_trajectory_query
        self.diff_reward = diff_reward
        self.function_mappings = function_mappings

        if terminals_mappings is None:
            self.terminals_mappings = {var: i for i, var in enumerate(constant_names)}
            if allow_constant_terminals:
                self.terminals_mappings['const'] = self.n_features
        else:
            self.terminals_mappings = terminals_mappings

    def mutate(self, program_tokens, allowed_operators, tree_program, masked_nodes,
               arity_ndarray=None, allowed_operators_arity=None, terminal_traj=False):
        """

        Parameters
        ----------
        program_tokens: list of string tokens (length == program length). Example : ['add', 'x', 'const']
        allowed_operators: list of allowed operators to be used in the mutation. Example: ['add', 'sub']
        tree_program: eckity object of the tree
        masked_nodes: indexes of the masked nodes in the program
        arity_ndarray: numpy array of the arity of the masked nodes (length == masked_nodes length)
        allowed_operators_arity: numpy array of the arity of the allowed operators (length == allowed_operators length)
        terminal_traj: boolean, if True, the mutation will be done on the terminal nodes, otherwise on
        the function nodes
        Returns
        -------

        """
        unmasked_tokens = program_to_labels(tree_program, [])

        mapped_mask_arity, mapped_masked_nodes, mapped_tokens, mapped_tokens_indices, sorted_mask_order = get_transformed_notation(
            arity_ndarray, masked_nodes, program_tokens, unmasked_tokens)

        initial_fitness = self.get_fitness_func(tree_program)
        tokens_ids = torch.tensor(
            [[self.token_to_id[token] for token in mapped_tokens]],
            dtype=torch.long,
            device=self.device,
        )
        logits = self.model(tokens_ids, attention_mask=torch.ones_like(tokens_ids).to(self.device)).logits
        mask_indices = torch.where(tokens_ids == self.mask_id)[1]

        suggested_mutation, trajectory_action_probabilities = self.masked_trajectory_generation(allowed_operators,
                                                                                                logits, mask_indices,
                                                                                                mapped_mask_arity,
                                                                                                allowed_operators_arity,
                                                                                                torch.clone(tokens_ids))

        # return the suggested mutation to the original order
        # notice that the suggested_mutation is returned in sorted order, so we use the sorted_mask_order to realign it
        realigned_order = mapped_tokens_indices[mapped_masked_nodes[sorted_mask_order]]

        for node, current_mutation in zip(realigned_order, suggested_mutation):

            if terminal_traj:
                current_mapping = self.terminals_mappings[current_mutation]
            else:
                current_mapping = self.function_mappings[current_mutation]

            if current_mutation == 'const':
                if type(tree_program.erc_range[0]) is float:
                    rand_constant = random.uniform(*tree_program.erc_range)
                else:
                    rand_constant = random.randint(*tree_program.erc_range)

                tree_program.tree[node] = TerminalNode(rand_constant)
            elif current_mutation in self.function_mappings:
                tree_program.tree[node] = FunctionNode(current_mapping)
            else:
                if callable(self.terminals_mappings[current_mutation]):
                    tree_program.tree[node] = TerminalNode(self.terminals_mappings[current_mutation])
                else:
                    tree_program.tree[node] = TerminalNode(current_mutation)

        new_fitness = self.get_fitness_func(tree_program)

        reward = self._policy_reward(initial_fitness, new_fitness, tree_program)

        trajectory_probability = torch.log(torch.cat(trajectory_action_probabilities)).sum().unsqueeze(
            0).unsqueeze(0)
        self.rewards.append(torch.full_like(trajectory_probability, reward))
        self.trajectory_probabilities.append(trajectory_probability)
        self.run_epoch()

    def _policy_reward(self, initial_fitness, new_fitness, tree_program):
        reward = (
            new_fitness - initial_fitness
            if self.diff_reward
            else new_fitness
        )
        return -reward if tree_program.higher_is_better else reward

    def masked_trajectory_generation(self, allowed_operators, logits, mask_indices, arity_of_masked_locations,
                                     allowed_operators_arity, tokens_ids):
        """
        :param tokens_ids:
        :param allowed_operators: list of allowed operators
        :param logits: model logits
        :param mask_indices: indices of the masked tokens
        :param arity_of_masked_locations: arity of the masked tokens
        :param allowed_operators_arity: arity of the allowed operators
        :return: suggested mutation and trajectory action probabilities
        """
        allowed_operators_arity, arity_of_masked_locations = convert_arity_to_tensors(allowed_operators,
                                                                                      allowed_operators_arity,
                                                                                      arity_of_masked_locations,
                                                                                      mask_indices)

        masked_softmax_indexes = torch.tensor(
            [self.token_to_id[token] for token in allowed_operators],
            dtype=torch.long,
        )
        suggested_mutation = []
        trajectory_action_probabilities = []

        # masked trajectory generation
        for trajectory_index in range(len(mask_indices)):
            current_mask_arity = arity_of_masked_locations[trajectory_index]
            current_allowed_operators = allowed_operators[current_mask_arity == allowed_operators_arity]
            current_masked_softmax_indexes = masked_softmax_indexes[
                current_mask_arity == allowed_operators_arity].to(self.device)

            # get the probability of the allowed operators and normalize them
            mask_index = torch.tensor([mask_indices[trajectory_index]]).type(torch.LongTensor)
            operators_proba = torch.softmax(logits[0, mask_index], dim=-1)[:,
                              current_masked_softmax_indexes].to(self.device)
            operators_proba = operators_proba / operators_proba.sum(dim=-1).unsqueeze(-1)

            # sample an operator with epsilon greedy
            if torch.rand(1) < self.epsilon_greedy:
                sampled_operators_dist = torch.randint(0, len(current_allowed_operators), (1,)).to(self.device)
            else:
                sampled_operators_dist = torch.distributions.Categorical(operators_proba).sample().to(self.device)

            sampled_actions_probability = torch.gather(operators_proba, dim=1,
                                                       index=sampled_operators_dist.unsqueeze(-1))
            trajectory_action_probabilities.append(sampled_actions_probability)
            sampled_operator = current_allowed_operators[
                sampled_operators_dist.detach().cpu().numpy()
            ][0]
            suggested_mutation.append(str(sampled_operator))

            if self.full_trajectory_query:
                tokens_ids = torch.clone(tokens_ids)
                tokens_ids[0, mask_index] = current_masked_softmax_indexes[sampled_operators_dist]
                logits = self.model(tokens_ids, attention_mask=torch.ones_like(tokens_ids).to(self.device)).logits

        return suggested_mutation, trajectory_action_probabilities

    def run_epoch(self, numerical_stability=1e-10):
        current_batch_size = sum([len(reward) for reward in self.rewards])
        if current_batch_size < self.batch_size:
            return

        all_traj_proba = torch.cat(self.trajectory_probabilities, dim=0).to(self.device)
        all_rewards = torch.cat(self.rewards, dim=0).to(self.device)

        self.trajectory_probabilities.clear()
        self.rewards.clear()

        self.optimizer.zero_grad()
        advantages = (all_rewards - torch.mean(all_rewards)) / (torch.std(all_rewards) + numerical_stability)
        # advantages = all_rewards
        advantages = advantages.to(self.device)
        loss = torch.mean(all_traj_proba * advantages).to(self.device)
        loss.backward()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        print(f'loss: {loss}, reward: {torch.mean(all_rewards)}')
