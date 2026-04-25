from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from efficient_kan import KAN
except Exception:
    KAN = None


class MLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.05,
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_kan(in_dim: int, out_dim: int, hidden_dims: Tuple[int, int] = (256, 128)) -> nn.Module:
    if KAN is None:
        raise ImportError(
            "efficient-kan is required for KAN models. Install with: pip install efficient-kan"
        )
    return KAN([in_dim, *list(hidden_dims), out_dim])


def _build_activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _stage_hidden_dims(default_dims: Sequence[int], stage_name: str, deployable_stage_sizing: bool, stronger_dims: Sequence[int], smaller_dims: Sequence[int]) -> Tuple[int, ...]:
    if not deployable_stage_sizing:
        return tuple(default_dims)
    if stage_name == "stage1_6s":
        return tuple(stronger_dims)
    if stage_name == "stage2_residual":
        return tuple(smaller_dims)
    return tuple(default_dims)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation: str = "relu", dropout: float = 0.05, use_layernorm: bool = False):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_build_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.05,
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        stem_layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_layernorm:
            stem_layers.append(nn.LayerNorm(hidden_dim))
        stem_layers.append(_build_activation(activation))
        if dropout > 0:
            stem_layers.append(nn.Dropout(dropout))
        self.stem = nn.Sequential(*stem_layers)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class SharedTrunkMultiHeadMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        trunk_dims: Sequence[int] = (512, 256),
        head_hidden_dim: int = 64,
        dropout: float = 0.05,
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        trunk_layers: List[nn.Module] = []
        last = in_dim
        for h in trunk_dims:
            trunk_layers.append(nn.Linear(last, h))
            if use_layernorm:
                trunk_layers.append(nn.LayerNorm(h))
            trunk_layers.append(_build_activation(activation))
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            last = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(last, head_hidden_dim),
                    _build_activation(activation),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(out_dim)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


class SharedTrunkMultiHeadKAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, trunk_dims: Sequence[int] = (192, 128), head_dims: Sequence[int] = (64,)):
        super().__init__()
        if KAN is None:
            raise ImportError(
                "efficient-kan is required for KAN models. Install with: pip install efficient-kan"
            )
        self.trunk = KAN([in_dim, *list(trunk_dims)])
        trunk_out = int(trunk_dims[-1])
        self.heads = nn.ModuleList([KAN([trunk_out, *list(head_dims), 1]) for _ in range(out_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def _build_mlp_variant(
    in_dim: int,
    out_dim: int,
    mlp_arch: str,
    activation: str,
    use_layernorm: bool,
    stage_name: str,
    deployable_stage_sizing: bool,
) -> nn.Module:
    variant = mlp_arch.lower()
    if variant == "baseline":
        hidden_dims = _stage_hidden_dims((256, 128), stage_name, deployable_stage_sizing, stronger_dims=(512, 256, 128), smaller_dims=(128, 64))
        return MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, dropout=0.05, activation=activation, use_layernorm=use_layernorm)
    if variant == "wider_gelu":
        hidden_dims = _stage_hidden_dims((512, 256, 128), stage_name, deployable_stage_sizing, stronger_dims=(512, 256, 128), smaller_dims=(256, 128))
        return MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, dropout=0.05, activation="gelu", use_layernorm=use_layernorm)
    if variant == "layernorm_gelu":
        hidden_dims = _stage_hidden_dims((256, 128), stage_name, deployable_stage_sizing, stronger_dims=(512, 256, 128), smaller_dims=(128, 64))
        return MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, dropout=0.05, activation="gelu", use_layernorm=True)
    if variant == "residual":
        hidden_dim = 256
        num_blocks = 2
        if deployable_stage_sizing and stage_name == "stage1_6s":
            hidden_dim = 384
            num_blocks = 3
        elif deployable_stage_sizing and stage_name == "stage2_residual":
            hidden_dim = 128
            num_blocks = 2
        return ResidualMLPRegressor(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=0.05,
            activation=activation,
            use_layernorm=use_layernorm,
        )
    if variant == "shared_trunk_multihead":
        trunk_dims = (512, 256)
        head_hidden = 64
        if deployable_stage_sizing and stage_name == "stage2_residual":
            trunk_dims = (256, 128)
            head_hidden = 32
        return SharedTrunkMultiHeadMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            trunk_dims=trunk_dims,
            head_hidden_dim=head_hidden,
            dropout=0.05,
            activation=activation,
            use_layernorm=use_layernorm,
        )
    raise ValueError(f"Unknown mlp_arch: {mlp_arch}")


def _build_kan_variant(
    in_dim: int,
    out_dim: int,
    kan_arch: str,
    stage_name: str,
    deployable_stage_sizing: bool,
) -> nn.Module:
    variant = kan_arch.lower()
    if variant == "baseline":
        hidden_dims = _stage_hidden_dims((256, 128), stage_name, deployable_stage_sizing, stronger_dims=(256, 128), smaller_dims=(128, 64))
        return _build_kan(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims)
    if variant == "small":
        hidden_dims = _stage_hidden_dims((128, 64), stage_name, deployable_stage_sizing, stronger_dims=(256, 128), smaller_dims=(128, 64))
        return _build_kan(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims)
    if variant == "balanced_deep":
        hidden_dims = _stage_hidden_dims((192, 128, 64), stage_name, deployable_stage_sizing, stronger_dims=(192, 128, 64), smaller_dims=(128, 64))
        return _build_kan(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims)
    if variant == "large_deep":
        hidden_dims = _stage_hidden_dims((256, 128, 64), stage_name, deployable_stage_sizing, stronger_dims=(256, 128, 64), smaller_dims=(128, 64))
        return _build_kan(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims)
    if variant == "shared_trunk_multihead":
        trunk_dims = (192, 128)
        head_dims = (64,)
        if deployable_stage_sizing and stage_name == "stage2_residual":
            trunk_dims = (128, 96)
            head_dims = (48,)
        return SharedTrunkMultiHeadKAN(in_dim=in_dim, out_dim=out_dim, trunk_dims=trunk_dims, head_dims=head_dims)
    raise ValueError(f"Unknown kan_arch: {kan_arch}")


def build_model(
    model_name: str,
    in_dim: int,
    out_dim: int,
    stage_name: str = "single_stage",
    mlp_arch: str = "baseline",
    kan_arch: str = "baseline",
    activation: str = "relu",
    use_layernorm: bool = False,
    deployable_stage_sizing: bool = False,
) -> Tuple[nn.Module, bool]:
    key = model_name.lower()
    if key == "mlp":
        return _build_mlp_variant(
            in_dim=in_dim,
            out_dim=out_dim,
            mlp_arch=mlp_arch,
            activation=activation,
            use_layernorm=use_layernorm,
            stage_name=stage_name,
            deployable_stage_sizing=deployable_stage_sizing,
        ), False
    if key == "pmlp":
        return _build_mlp_variant(
            in_dim=in_dim,
            out_dim=out_dim,
            mlp_arch=mlp_arch,
            activation=activation,
            use_layernorm=use_layernorm,
            stage_name=stage_name,
            deployable_stage_sizing=deployable_stage_sizing,
        ), True
    if key == "kan":
        return _build_kan_variant(
            in_dim=in_dim,
            out_dim=out_dim,
            kan_arch=kan_arch,
            stage_name=stage_name,
            deployable_stage_sizing=deployable_stage_sizing,
        ), False
    if key == "pkan":
        return _build_kan_variant(
            in_dim=in_dim,
            out_dim=out_dim,
            kan_arch=kan_arch,
            stage_name=stage_name,
            deployable_stage_sizing=deployable_stage_sizing,
        ), True
    raise ValueError(f"Unknown model name: {model_name}")
