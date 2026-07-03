from dataclasses import dataclass


@dataclass
class MaiaConfig:
    # Dataset
    history: int = 8
    use_padding: bool = True
    include_time_info: bool = False

    # Embeddings
    dim_emb: int = 128

    # Transformer
    dim_vit: int = 256
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    # Heads
    head_hid_dim: int = 256

    # GAB
    use_gab: bool = True
    use_relative_bias: bool = False
    use_absolute_pe: bool = False

    gab_gen_size: int = 64
    gab_per_square_dim: int = 0
    gab_intermediate_dim: int = 64

    # Transformer options
    use_rms_norm: bool = True
    omit_qkv_biases: bool = True
    activation: str = "gelu"


def get_maia3_5m_config():
    """
    Configuration matching the official Maia3-5M checkpoint.
    """
    return MaiaConfig()


def get_maia3_3m_config():
    cfg = MaiaConfig()
    cfg.dim_vit = 192
    cfg.head_hid_dim = 192
    cfg.num_heads = 6
    return cfg


def get_maia3_23m_config():
    cfg = MaiaConfig()
    cfg.dim_vit = 512
    cfg.head_hid_dim = 512
    cfg.num_heads = 16
    cfg.gab_gen_size = 128
    cfg.gab_per_square_dim = 32
    cfg.gab_intermediate_dim = 128
    return cfg


def get_maia3_79m_config():
    cfg = MaiaConfig()
    cfg.dim_vit = 1024
    cfg.head_hid_dim = 1024
    cfg.num_heads = 32
    cfg.gab_gen_size = 128
    cfg.gab_per_square_dim = 32
    cfg.gab_intermediate_dim = 128
    return cfg