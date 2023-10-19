from dataclasses import dataclass


@dataclass
class EncoderPromptConfig(object):
    # general
    seq_len = 0
    input_dim = 768
    mid_dim = 768
    use_input_prompt = True
    use_single_prompt = False

@dataclass
class DecoderPromptConfig(object):
    # general
    seq_len = 0
    input_dim = 768
    mid_dim = 768
    use_input_prompt = True
    use_single_prompt = False


@dataclass
class PromptConfig(object):
    non_linearity: str = "gelu_new"
    reduction_factor: int = 6
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # cross atten prompt
    prompt_cross = False

    # Hypercomplex parameters
    hypercomplex_division = 4
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False  #True
    factorized_phm = True    #False #True  # TODO
    shared_W_phm = False
    factorized_phm_rule = False
    phm_c_init = "normal"
    phm_rank = 8   #1  # 8
    phm_init_range = 0.0001  #0.0001 0.01
    kronecker_prod = False