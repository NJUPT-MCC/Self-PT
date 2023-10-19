# # The codes are from https://github.com/NJUPT-MCC/Self-PT

# # Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering. ACM MM 2023


import torch
import torch.nn as nn
from .prompt_modeling import InputPrompts, PHMLayer
from .hypercomplex.layers import PHMLinear
from torch.nn.parameter import Parameter

from transformers.activations import get_activation

class PromptController(nn.Module):
    """
    Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_seq_len = config.pre_seq_len  # prefix sequence length
        self.hidden_size = config.d_model
        self.reduction_factor = config.prompt_config.reduction_factor

        self.middle_size = self.hidden_size // self.reduction_factor
        print(self.middle_size)
        if self.config.prompt_type == 'orip':
            self.embedding = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.middle_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.middle_size, self.hidden_size))
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        elif self.config.prompt_type == 'ap':
            self.trans_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.middle_size),
                                               nn.ReLU(),
                                               nn.Linear(self.middle_size, self.pre_seq_len * self.hidden_size),
                                               )

        elif self.config.prompt_type == 'hyper':
            self.trans_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.middle_size),
                                               Activations(config.prompt_config.non_linearity.lower()),
                                               # nn.ReLU(),
                                               )

            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.middle_size * self.hidden_size))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_size))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

        elif self.config.prompt_type == 'cocoop':
            # self.embedding = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
            ctx_vectors = torch.empty(self.pre_seq_len, self.hidden_size)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.embedding = nn.Parameter(ctx_vectors)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.middle_size),
                # torch.nn.Tanh(),
                torch.nn.ReLU(),
                torch.nn.Linear(self.middle_size, self.hidden_size)
            )

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        elif self.config.prompt_type == 'ap_phm':
            self.trans_encoder = nn.Sequential(PHMLayer(4, self.hidden_size, self.middle_size),
                                               nn.ReLU(),
                                               PHMLayer(4, self.middle_size, self.pre_seq_len * self.hidden_size),
                                               )

        elif self.config.prompt_type == 'hyper_phm':
            self.trans_encoder = nn.Sequential(PHMLayer(4, self.hidden_size, self.middle_size),
                                               nn.ReLU())
            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.fc_list = nn.ModuleList()
            for _ in range(self.task_embedding_dim + 1):  # +1 for bias
                self.fc_list.append(PHMLayer(4, self.middle_size, self.hidden_size))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

        elif self.config.prompt_type == 'hyper_phm_new':
            print('phmrank',config.prompt_config.phm_rank)
            self.trans_encoder = nn.Sequential(PHMLinear(in_features=self.hidden_size,
                                                  out_features=self.middle_size,
                                                  bias=True,
                                                  phm_dim=config.prompt_config.hypercomplex_division,
                                                  learn_phm=config.prompt_config.learn_phm,
                                                  w_init=config.prompt_config.hypercomplex_nonlinearity,
                                                  shared_phm_rule=config.prompt_config.shared_phm_rule,
                                                  factorized_phm=config.prompt_config.factorized_phm,
                                                  shared_W_phm=config.prompt_config.shared_W_phm,
                                                  factorized_phm_rule=config.prompt_config.factorized_phm_rule,
                                                  c_init=config.prompt_config.phm_c_init,
                                                  phm_rank=config.prompt_config.phm_rank,
                                                  phm_init_range=config.prompt_config.phm_init_range,
                                                  kronecker_prod=config.prompt_config.kronecker_prod),
                                               Activations(config.prompt_config.non_linearity.lower()))
            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.fc_list = nn.ModuleList()
            for _ in range(self.task_embedding_dim + 1):  # +1 for bias
                self.fc_list.append(PHMLinear(in_features=self.middle_size,
                                              out_features=self.hidden_size,
                                              bias=True,
                                              phm_dim=config.prompt_config.hypercomplex_division,
                                              learn_phm=config.prompt_config.learn_phm,
                                              w_init=config.prompt_config.hypercomplex_nonlinearity,
                                              shared_phm_rule=config.prompt_config.shared_phm_rule,
                                              factorized_phm=config.prompt_config.factorized_phm,
                                              shared_W_phm=config.prompt_config.shared_W_phm,
                                              factorized_phm_rule=config.prompt_config.factorized_phm_rule,
                                              c_init=config.prompt_config.phm_c_init,
                                              phm_rank=config.prompt_config.phm_rank,
                                              phm_init_range=config.prompt_config.phm_init_range,
                                              kronecker_prod=config.prompt_config.kronecker_prod))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

    def forward(self, hidden_states):
        if self.config.is_decoder is False:
            if self.config.prompt_input_type == 'cls':
                input_embed = hidden_states[:, 0, :]
            elif self.config.prompt_input_type == 'mean':
                input_embed = torch.mean(hidden_states, dim=1) #.data
            elif self.config.prompt_input_type == 'max':
                input_embed = torch.max(hidden_states, dim=1).values #.data
        else:
            # input_embed = hidden_states[:, 0, :].data
            input_embed = hidden_states[:, 0, :]

        if self.config.prompt_type == 'orip':
            prompt_embed = self.original_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'ap' or self.config.prompt_type == 'ap_phm':
            prompt_embed = self.ap_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'hyper':
            prompt_embed = self.hyper_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'cocoop':
            prompt_embed = self.cocoop_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'hyper_phm' or self.config.prompt_type == 'hyper_phm_new':
            prompt_embed = self.hyperPHM_prompt(input_embed, hidden_states.shape)

        return prompt_embed

    def original_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bzs, -1).to(input_embed.device)
        prefix_tokens = self.embedding(prefix_tokens)
        prompt_embed = self.trans(prefix_tokens)
        return prompt_embed

    def ap_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        prompt_embed = self.trans_encoder(input_embed).view(bzs, self.pre_seq_len, self.hidden_size)
        return prompt_embed

    def hyper_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        prompt_embed = self.trans_encoder(input_embed).view(bzs, self.middle_size)
        prefix_tokens = self.prefix_tokens.to(input_embed.device)  # pre_seq_len
        prefix_tokens = self.prompt_id_embeddings(prefix_tokens) # pre_seq_len, task_embedding_dim
        # prefix_tokens = nn.Softmax(dim=-1)(prefix_tokens)
        prefix_tokens = self.LayerNorm(prefix_tokens)
        weight = self.weight_generator(prefix_tokens).view(self.pre_seq_len, self.middle_size, self.hidden_size).unsqueeze(0)
        bias = self.bias_generator(prefix_tokens).view(self.pre_seq_len, self.hidden_size).unsqueeze(0)
        prompt_embed = prompt_embed.unsqueeze(1).unsqueeze(1)
        prompt_embed = (prompt_embed @ weight).squeeze(2) + bias
        return prompt_embed

    def cocoop_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        meta = self.trans(input_embed).unsqueeze(1)
        # prompt_embed = self.prefix_tokens.unsqueeze(0).expand(bzs, -1).to(hidden_states.device)
        # prompt_embed = self.embedding(prompt_embed)
        prompt_embed = self.embedding.unsqueeze(0).expand(bzs, self.pre_seq_len, self.hidden_size)
        prompt_embed = prompt_embed + meta
        return prompt_embed

    def hyperPHM_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        q_embed = self.trans_encoder(input_embed).view(bzs, self.middle_size)
        prefix_tokens = self.prefix_tokens.to(input_embed.device)  # pre_seq_len
        prefix_tokens = self.prompt_id_embeddings(prefix_tokens) # pre_seq_len, task_embedding_dim
        # prefix_tokens = nn.Softmax(dim=-1)(prefix_tokens)
        prefix_tokens = self.LayerNorm(prefix_tokens)
        prompt_embed_list = []
        for i in range(self.task_embedding_dim):
            prompt_embed = self.fc_list[i](q_embed).view(bzs, self.hidden_size)
            prompt_embed_list.append(prompt_embed)
        bias_embed = self.fc_list[self.task_embedding_dim](q_embed).view(bzs, self.hidden_size)  # bias
        prompt_embed = [lf for lf in prompt_embed_list]
        prompt_embed = torch.stack(prompt_embed, dim=1).to(input_embed.device)  # bz, task_embedding_dim, hidden_size
        prompt_embed = prefix_tokens.unsqueeze(0) @ prompt_embed + bias_embed.unsqueeze(1)
        return prompt_embed

class PromptController_cross(nn.Module):
    """
    Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_seq_len = config.pre_seq_len
        self.hidden_size = config.d_model
        self.reduction_factor = config.prompt_config.reduction_factor

        self.middle_size = self.hidden_size // self.reduction_factor
        print(self.middle_size)

        if self.config.prompt_type == 'ap':
            self.trans_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.middle_size),
                                               nn.ReLU(),
                                               nn.Linear(self.middle_size, self.pre_seq_len * self.hidden_size),
                                               )

        elif self.config.prompt_type == 'hyper':
            self.trans_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.middle_size),
                                               Activations(config.prompt_config.non_linearity.lower()),
                                               # nn.ReLU(),
                                               )

            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.middle_size * self.hidden_size))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_size))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

        elif self.config.prompt_type == 'cocoop':
            # self.embedding = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
            ctx_vectors = torch.empty(self.pre_seq_len, self.hidden_size)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.embedding = nn.Parameter(ctx_vectors)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.middle_size),
                # torch.nn.Tanh(),
                torch.nn.ReLU(),
                torch.nn.Linear(self.middle_size, self.hidden_size)
            )

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        elif self.config.prompt_type == 'ap_phm':
            self.trans_encoder = nn.Sequential(PHMLayer(4, self.hidden_size, self.middle_size),
                                               nn.ReLU(),
                                               PHMLayer(4, self.middle_size, self.pre_seq_len * self.hidden_size),
                                               )

        elif self.config.prompt_type == 'hyper_phm':
            self.trans_encoder = nn.Sequential(PHMLayer(4, self.hidden_size, self.middle_size),
                                               nn.ReLU())
            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.fc_list = nn.ModuleList()
            for _ in range(self.task_embedding_dim + 1):  # +1 for bias
                self.fc_list.append(PHMLayer(4, self.middle_size, self.hidden_size))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

        elif self.config.prompt_type == 'hyper_phm_new':
            print('phmrank',config.prompt_config.phm_rank)
            self.trans_encoder = nn.Sequential(PHMLinear(in_features=self.hidden_size,
                                                  out_features=self.middle_size,
                                                  bias=True,
                                                  phm_dim=config.prompt_config.hypercomplex_division,
                                                  learn_phm=config.prompt_config.learn_phm,
                                                  w_init=config.prompt_config.hypercomplex_nonlinearity,
                                                  shared_phm_rule=config.prompt_config.shared_phm_rule,
                                                  factorized_phm=config.prompt_config.factorized_phm,
                                                  shared_W_phm=config.prompt_config.shared_W_phm,
                                                  factorized_phm_rule=config.prompt_config.factorized_phm_rule,
                                                  c_init=config.prompt_config.phm_c_init,
                                                  phm_rank=config.prompt_config.phm_rank,
                                                  phm_init_range=config.prompt_config.phm_init_range,
                                                  kronecker_prod=config.prompt_config.kronecker_prod),
                                               Activations(config.prompt_config.non_linearity.lower()))
            self.task_embedding_dim = config.prompt_index_dim
            # Considers weight and bias parameters for generating weights.
            self.fc_list = nn.ModuleList()
            for _ in range(self.task_embedding_dim + 1):  # +1 for bias
                self.fc_list.append(PHMLinear(in_features=self.middle_size,
                                              out_features=self.hidden_size,
                                              bias=True,
                                              phm_dim=config.prompt_config.hypercomplex_division,
                                              learn_phm=config.prompt_config.learn_phm,
                                              w_init=config.prompt_config.hypercomplex_nonlinearity,
                                              shared_phm_rule=config.prompt_config.shared_phm_rule,
                                              factorized_phm=config.prompt_config.factorized_phm,
                                              shared_W_phm=config.prompt_config.shared_W_phm,
                                              factorized_phm_rule=config.prompt_config.factorized_phm_rule,
                                              c_init=config.prompt_config.phm_c_init,
                                              phm_rank=config.prompt_config.phm_rank,
                                              phm_init_range=config.prompt_config.phm_init_range,
                                              kronecker_prod=config.prompt_config.kronecker_prod))

            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prompt_id_embeddings = nn.Embedding(self.pre_seq_len, self.task_embedding_dim)

            self.LayerNorm = nn.LayerNorm(self.task_embedding_dim, eps=1e-6)

    def forward(self, hidden_states):
        input_embed = hidden_states[:, 0, :]

        if self.config.prompt_type == 'ap' or self.config.prompt_type == 'ap_phm':
            prompt_embed = self.ap_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'hyper':
            prompt_embed = self.hyper_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'cocoop':
            prompt_embed = self.cocoop_prompt(input_embed, hidden_states.shape)
        elif self.config.prompt_type == 'hyper_phm' or self.config.prompt_type == 'hyper_phm_new':
            prompt_embed = self.hyperPHM_prompt(input_embed, hidden_states.shape)

        return prompt_embed

    def ap_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        prompt_embed = self.trans_encoder(input_embed).view(bzs, self.pre_seq_len, self.hidden_size)
        return prompt_embed

    def hyper_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        prompt_embed = self.trans_encoder(input_embed).view(bzs, self.middle_size)
        prefix_tokens = self.prefix_tokens.to(input_embed.device)  # pre_seq_len
        prefix_tokens = self.prompt_id_embeddings(prefix_tokens) # pre_seq_len, task_embedding_dim
        # prefix_tokens = nn.Softmax(dim=-1)(prefix_tokens)
        prefix_tokens = self.LayerNorm(prefix_tokens)
        weight = self.weight_generator(prefix_tokens).view(self.pre_seq_len, self.middle_size, self.hidden_size).unsqueeze(0)
        bias = self.bias_generator(prefix_tokens).view(self.pre_seq_len, self.hidden_size).unsqueeze(0)
        prompt_embed = prompt_embed.unsqueeze(1).unsqueeze(1)
        prompt_embed = (prompt_embed @ weight).squeeze(2) + bias
        return prompt_embed

    def cocoop_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        meta = self.trans(input_embed).unsqueeze(1)
        # prompt_embed = self.prefix_tokens.unsqueeze(0).expand(bzs, -1).to(hidden_states.device)
        # prompt_embed = self.embedding(prompt_embed)
        prompt_embed = self.embedding.unsqueeze(0).expand(bzs, self.pre_seq_len, self.hidden_size)
        prompt_embed = prompt_embed + meta
        return prompt_embed

    def hyperPHM_prompt(self, input_embed, shape_info):
        bzs, seq_length, embed_dim = shape_info
        q_embed = self.trans_encoder(input_embed).view(bzs, self.middle_size)
        prefix_tokens = self.prefix_tokens.to(input_embed.device)  # pre_seq_len
        prefix_tokens = self.prompt_id_embeddings(prefix_tokens) # pre_seq_len, task_embedding_dim
        # prefix_tokens = nn.Softmax(dim=-1)(prefix_tokens)
        prefix_tokens = self.LayerNorm(prefix_tokens)
        prompt_embed_list = []
        for i in range(self.task_embedding_dim):
            prompt_embed = self.fc_list[i](q_embed).view(bzs, self.hidden_size)
            prompt_embed_list.append(prompt_embed)
        bias_embed = self.fc_list[self.task_embedding_dim](q_embed).view(bzs, self.hidden_size)  # bias
        prompt_embed = [lf for lf in prompt_embed_list]
        prompt_embed = torch.stack(prompt_embed, dim=1).to(input_embed.device)  # bz, task_embedding_dim, hidden_size
        prompt_embed = prefix_tokens.unsqueeze(0) @ prompt_embed + bias_embed.unsqueeze(1)
        return prompt_embed


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module"""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    # linear = nn.Linear(input_dim, output_dim, bias=False)
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class ORIPromptController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prompts = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.use_input_prompt = config.use_input_prompt
        self.use_single_prompt = config.use_single_prompt
        self.prompts = self.construct_prompts(self.tasks)

    def get_task(self, task):
        return task

    def construct_prompts(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """

        if self.use_single_prompt:
            if self.use_input_prompt:
                prompt = InputPrompts(self.config)

            for task in tasks:
                self.prompts[task] = prompt

        else:
            for task in tasks:
                if self.use_input_prompt:
                    prompt = InputPrompts(self.config)

                    self.prompts[task] = prompt

        return self.prompts

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def get_prompt(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.prompts[task]

    def forward(self, bsz, device, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer.
        """
        task = self.get_task(task)
        # Enables the adapter layer for the given task.
        prompt_module = self.get_prompt(task)

        trainable_prompt = prompt_module.get_prompt(bsz, device)

        return trainable_prompt