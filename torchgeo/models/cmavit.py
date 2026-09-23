# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Climate Management-Aware Vision Transformer (CMAViT)."""

from collections.abc import Sequence
from typing import Literal, overload

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.modules.utils import _pair

from ..datasets.utils import lazy_import


class FeedForward(nn.Module):
    """Position-wise feedforward block."""

    def __init__(self, dim: int, mult: float = 4, dropout: float = 0.0) -> None:
        """Initialize a new FeedForward instance.

        Args:
            dim: Input and output feature dimension.
            mult: Hidden dimension expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input of shape (B, N, dim).

        Returns:
            Output of shape (B, N, dim).
        """
        return self.net(x)


class TextEmbed(nn.Module):
    """Frozen pre-trained text encoder.

    The pre-trained backbone is never fine-tuned and always stays in eval mode.
    Only :class:`TextEncoder`, which consumes its embeddings, is trained.
    """

    def __init__(self, model_name: str, max_length: int) -> None:
        """Initialize a new TextEmbed instance.

        Args:
            model_name: Name of a Hugging Face text encoder.
            max_length: Number of tokens every text is padded or truncated to.

        Raises:
            DependencyNotFoundError: If transformers is not installed.
        """
        super().__init__()
        transformers = lazy_import('transformers')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dim: int = self.model.config.hidden_size

        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> 'TextEmbed':
        """Set the training mode, keeping the frozen backbone in eval mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            The module itself.
        """
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            texts: Sequence of B raw text strings.

        Returns:
            Token embeddings of shape (B, max_length, dim) and a boolean attention
            mask of shape (B, max_length), True for real (non-padding) tokens.
        """
        inputs = self.tokenizer(
            list(texts),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embeddings: Tensor = self.model(**inputs).last_hidden_state
        return embeddings, inputs['attention_mask'].bool()


class TextEncoder(nn.Module):
    """Pre-norm self-attention layer applied to the text token embeddings."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mult: float = 4, dropout: float = 0.0
    ) -> None:
        """Initialize a new TextEncoder instance.

        Args:
            dim: Token embedding dimension.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            mult: Feedforward hidden dimension expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult, dropout)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Token embeddings of shape (B, N, dim).
            mask: Boolean attention mask of shape (B, N), True for tokens to attend to.

        Returns:
            Output of shape (B, N, dim) and attention weights of shape
            (B, heads, N, N).
        """
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        )
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots.masked_fill(
            ~rearrange(mask, 'b n -> b 1 1 n'), -torch.finfo(dots.dtype).max
        )
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = x + self.to_out(rearrange(out, 'b h n d -> b n (h d)'))
        return x + self.ff(self.norm2(x)), attn


class ImageEmbed(nn.Module):
    """Patch embedding of a time series of images."""

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
        dropout: float,
        num_observations: int,
    ) -> None:
        """Initialize a new ImageEmbed instance.

        Args:
            img_size: Height and width of each image.
            patch_size: Height and width of each patch.
            in_channels: Number of image channels.
            embed_dim: Embedding dimension.
            dropout: Dropout probability.
            num_observations: Maximum number of observations (timesteps).
        """
        super().__init__()
        img_size, patch_size = _pair(img_size), _pair(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches * num_observations + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Images of shape (B, T, C, H, W), with T at most *num_observations*.

        Returns:
            Tokens of shape (B, 1 + T * num_patches, embed_dim), where the first
            token is a learned class token.
        """
        b = x.shape[0]
        x = self.proj(rearrange(x, 'b t c h w -> (b t) c h w'))
        x = self.norm(rearrange(x, '(b t) e h w -> b (t h w) e', b=b))
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        # Observations are ordered in time, so a shorter series uses a prefix
        x = x + self.position_embeddings[:, : x.shape[1]]
        return self.dropout(x)


class MeteorologyEmbed(nn.Module):
    """Token embedding of a time series of meteorological observations.

    Meteorology has no spatial extent, so all the variables of a timestep are
    embedded together as a single token. That token is repeated for every image
    patch of the same timestep, such that it can bias the attention between the
    image tokens.
    """

    def __init__(
        self,
        met_channels: int,
        embed_dim: int,
        dropout: float,
        num_observations: int,
        num_patches: int,
    ) -> None:
        """Initialize a new MeteorologyEmbed instance.

        Args:
            met_channels: Number of meteorological variables per timestep.
            embed_dim: Embedding dimension.
            dropout: Dropout probability.
            num_observations: Maximum number of observations (timesteps).
            num_patches: Number of image patches per observation.
        """
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(met_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_observations + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, met: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            met: Meteorology of shape (B, T, C), where T is at most
                *num_observations* and C equals *met_channels*.

        Returns:
            Tokens of shape (B, 1 + T * num_patches, embed_dim), aligned with the
            output of :class:`ImageEmbed`.
        """
        b, t, _ = met.shape
        x = self.proj(met)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = self.dropout(x + self.position_embeddings[:, : t + 1])

        # The image tokens are a class token followed by the patches of each timestep
        cls_index = torch.zeros(1, dtype=torch.long, device=x.device)
        patch_index = torch.arange(1, t + 1, device=x.device)
        return x[
            :, torch.cat((cls_index, patch_index.repeat_interleave(self.num_patches)))
        ]


class SpatialMetLayer(nn.Module):
    """Transformer layer whose attention is biased by the meteorological tokens."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mult: float, dropout: float
    ) -> None:
        """Initialize a new SpatialMetLayer instance.

        Args:
            dim: Token embedding dimension.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            mult: Feedforward hidden dimension expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult, dropout)

    def forward(self, x: Tensor, met: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Image tokens of shape (B, N, dim).
            met: Meteorological tokens of shape (B, N, dim).

        Returns:
            Output of shape (B, N, dim).
        """
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        )
        # The meteorological tokens share the query/key projection of the image
        # tokens and are not normalized, they only add a bias to the attention logits
        qm, km, _ = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in self.to_qkv(met).chunk(3, dim=-1)
        )
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + torch.einsum('b h i d, b h j d -> b h i j', qm, km) * self.scale
        out = torch.einsum('b h i j, b h j d -> b h i d', dots.softmax(dim=-1), v)
        x = x + self.to_out(rearrange(out, 'b h n d -> b n (h d)'))
        return x + self.ff(self.norm2(x))


class CrossModalLayer(nn.Module):
    """Transformer layer with cross-attention from image/meteorology to text."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int,
        dim_head: int,
        mult: float,
        dropout: float,
    ) -> None:
        """Initialize a new CrossModalLayer instance.

        Args:
            dim: Query embedding dimension.
            context_dim: Text embedding dimension.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            mult: Feedforward hidden dimension expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult, dropout)

    def forward(
        self, x: Tensor, context: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Query tokens of shape (B, N, dim).
            context: Text tokens of shape (B, M, context_dim).
            mask: Boolean attention mask of shape (B, M), True for text tokens to
                attend to.

        Returns:
            Output of shape (B, N, dim) and attention weights of shape
            (B, heads, N, M).
        """
        q = rearrange(self.to_q(self.norm1(x)), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(context), 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(self.to_v(context), 'b m (h d) -> b h m d', h=self.heads)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots.masked_fill(
            ~rearrange(mask, 'b m -> b 1 1 m'), -torch.finfo(dots.dtype).max
        )
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = x + self.to_out(rearrange(out, 'b h n d -> b n (h d)'))
        return x + self.ff(self.norm2(x)), attn


class RegressionHead(nn.Module):
    """Regression head that maps fused features to a square prediction map."""

    def __init__(self, embed_dim: int, output_size: int) -> None:
        """Initialize a new RegressionHead instance.

        Args:
            embed_dim: Dimension of each of the two fused feature vectors.
            output_size: Height and width of the predicted map.
        """
        super().__init__()
        self.output_size = output_size
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.fc = nn.Linear(embed_dim * 2, output_size**2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Fused features of shape (B, 2 * embed_dim).

        Returns:
            Prediction map of shape (B, 1, output_size, output_size).
        """
        x = self.fc(self.norm(x))
        return rearrange(x, 'b (h w) -> b 1 h w', h=self.output_size)


class CMAViT(nn.Module):
    """Climate Management-Aware Vision Transformer (CMAViT).

    CMAViT predicts pixel-level yield maps throughout the growing season from three
    modalities: a time series of satellite imagery, a time series of meteorological
    observations, and a text report on soil and management practices. It consists
    of:

    * a spatio-temporal multimodal encoder in which attention logits computed from
      the meteorological tokens are added to those computed from the image tokens
    * a self-attention encoder for the text
    * a cross-attention encoder in which every image/meteorology token attends to
      the tokens of the text, ignoring padding
    * a regression head that predicts a map

    The default hyperparameters of the input follow the paper, which uses 16 x 16
    pixel chips split into 2 x 2 patches, 6 channels (4 Sentinel-2 bands and 2
    Sentinel-1 bands) plus the sine of the day of the year, and 4 meteorological
    variables (minimum and maximum temperature, precipitation, and vapor pressure).

    Meteorology has no spatial extent, so each timestep is embedded as a single
    token, which biases the attention between all the image tokens of that timestep.
    The image size, the number of patches, and the number of meteorological
    variables are therefore independent.

    The text is embedded by a frozen pre-trained Hugging Face encoder (by default
    `DistilBERT <https://huggingface.co/distilbert/distilbert-base-uncased>`_),
    which requires the optional ``transformers`` dependency and, unless it is
    already cached, internet access.

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2026.3669511

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 16,
        patch_size: int | tuple[int, int] = 8,
        in_channels: int = 7,
        met_channels: int = 4,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        text_dim_head: int = 8,
        spatial_dim_head: int = 96,
        ff_mult: float = 4,
        proj_dropout: float = 0.0,
        timeseries: bool = False,
        num_observations: int = 15,
        text_model_name: str = 'distilbert-base-uncased',
        text_max_length: int = 250,
        output_size: int = 16,
    ) -> None:
        """Initialize a new CMAViT instance.

        Args:
            img_size: Height and width of each image.
            patch_size: Height and width of each image patch.
            in_channels: Number of image channels.
            met_channels: Number of meteorological variables per timestep.
            embed_dim: Embedding dimension of the image, meteorology, and
                cross-modal encoders.
            num_layers: Depth of the text, spatial-meteorology, and cross-modal
                encoders.
            num_heads: Number of attention heads of every encoder.
            text_dim_head: Dimension of each attention head of the text encoder.
            spatial_dim_head: Dimension of each attention head of the
                spatial-meteorology and cross-modal encoders.
            ff_mult: Hidden dimension expansion ratio of every feedforward block.
            proj_dropout: Dropout probability.
            timeseries: If True, predict after each of the first 1 to T
                observations of the input. If False, make a single prediction from
                all T observations.
            num_observations: Maximum number of timesteps T of the input time
                series. Shorter series are supported.
            text_model_name: Name of the pre-trained Hugging Face text encoder.
            text_max_length: Number of tokens every text is padded or truncated to.
            output_size: Height and width of each predicted map.

        Raises:
            DependencyNotFoundError: If transformers is not installed.
        """
        super().__init__()

        img_h, img_w = _pair(img_size)
        patch_h, patch_w = _pair(patch_size)
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        self.timeseries = timeseries
        self.num_observations = num_observations

        self.text_embed = TextEmbed(text_model_name, text_max_length)
        context_dim = self.text_embed.dim
        self.text_transformer = nn.ModuleList(
            [
                TextEncoder(
                    context_dim, num_heads, text_dim_head, ff_mult, proj_dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.text_norm = nn.LayerNorm(context_dim)

        self.img_embed = ImageEmbed(
            img_size, patch_size, in_channels, embed_dim, proj_dropout, num_observations
        )
        self.met_embed = MeteorologyEmbed(
            met_channels, embed_dim, proj_dropout, num_observations, num_patches
        )

        self.spatialmet_encoder = nn.ModuleList(
            [
                SpatialMetLayer(
                    embed_dim, num_heads, spatial_dim_head, ff_mult, proj_dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.spatialmet_norm = nn.LayerNorm(embed_dim)

        self.cross_attn_encoder = nn.ModuleList(
            [
                CrossModalLayer(
                    embed_dim,
                    context_dim,
                    num_heads,
                    spatial_dim_head,
                    ff_mult,
                    proj_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.head = RegressionHead(embed_dim, output_size)

        # The pre-trained text backbone keeps its own weights
        for name, module in self.named_children():
            if name != 'text_embed':
                module.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize the weights of a single module.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _encode_text(
        self, context: Sequence[str], return_attention: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Embed the texts.

        Args:
            context: Sequence of B raw text strings.
            return_attention: Whether to also return the attention weights.

        Returns:
            The text tokens of shape (B, N, context_dim), the boolean mask of shape
            (B, N) that is True for real (non-padding) tokens, and, if requested,
            the text self-attention weights of shape (B, heads, N, N, num_layers),
            where N is the text length.
        """
        x, mask = self.text_embed(context)
        attns = []
        for layer in self.text_transformer:
            x, attn = layer(x, mask)
            if return_attention:
                attns.append(attn)

        text_attn = torch.stack(attns, dim=4) if return_attention else None
        return self.text_norm(x), mask, text_attn

    def _predict(
        self,
        img: Tensor,
        met: Tensor,
        text: Tensor,
        mask: Tensor,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict a map from a window of observations.

        Args:
            img: Images of shape (B, T, C, H, W).
            met: Meteorology of shape (B, T, C).
            text: Text tokens of shape (B, N, context_dim).
            mask: Boolean text mask of shape (B, N).
            return_attention: Whether to also return the attention weights.

        Returns:
            Predicted map of shape (B, 1, output_size, output_size) and, if
            requested, the cross-attention weights of shape
            (B, heads, T * num_patches, N, num_layers).
        """
        x, m = self.img_embed(img), self.met_embed(met)
        for spatialmet_layer in self.spatialmet_encoder:
            x = spatialmet_layer(x, m)
        # The class token is only a register for the attention
        x = self.spatialmet_norm(x)[:, 1:]

        # Every image/meteorology token gathers the text tokens it needs
        fused = x
        attns = []
        for cross_layer in self.cross_attn_encoder:
            fused, attn = cross_layer(fused, text, mask)
            if return_attention:
                attns.append(attn)
        fused = self.cross_attn_norm(fused)

        features = torch.cat((fused.mean(dim=1), x.mean(dim=1)), dim=-1)
        cross_attn = torch.stack(attns, dim=4) if return_attention else None
        return self.head(features), cross_attn

    @overload
    def forward(
        self,
        img: Tensor,
        context: Sequence[str],
        met: Tensor,
        return_attention: Literal[False] = False,
    ) -> list[Tensor]: ...

    @overload
    def forward(
        self,
        img: Tensor,
        context: Sequence[str],
        met: Tensor,
        return_attention: Literal[True],
    ) -> tuple[list[Tensor], Tensor, list[Tensor]]: ...

    def forward(
        self,
        img: Tensor,
        context: Sequence[str],
        met: Tensor,
        return_attention: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], Tensor, list[Tensor]]:
        """Forward pass of the model.

        Args:
            img: Images of shape (B, T, C, H, W), with T at most *num_observations*.
            context: Sequence of B raw text strings describing the management.
            met: Meteorology of shape (B, T, C), with C equal to *met_channels*.
            return_attention: Whether to also return the attention weights, which
                are large and only useful for interpretation.

        Returns:
            A list of predicted maps, each of shape
            (B, 1, output_size, output_size), one per window of observations: a
            single one if *timeseries* is False, otherwise T of them, where the
            i-th only uses the first i observations.

            If *return_attention* is True, a tuple of the list of predicted maps,
            the text self-attention weights of shape (B, heads, N, N, num_layers),
            where N is the text length, and a list with the cross-attention weights
            of each window, of shape (B, heads, i * num_patches, N, num_layers),
            which show which text tokens each image patch attends to.

        Raises:
            ValueError: If the number of timesteps of *img* and *met* differ, or
                exceeds *num_observations*.
        """
        num_timesteps = img.shape[1]
        if met.shape[1] != num_timesteps:
            raise ValueError(
                f'img has {num_timesteps} timesteps but met has {met.shape[1]}'
            )
        if num_timesteps > self.num_observations:
            raise ValueError(
                f'The input has {num_timesteps} timesteps, more than '
                f'num_observations ({self.num_observations})'
            )

        text, mask, text_attn = self._encode_text(context, return_attention)
        windows = range(1, num_timesteps + 1) if self.timeseries else (num_timesteps,)
        outputs = [
            self._predict(img[:, :t], met[:, :t], text, mask, return_attention)
            for t in windows
        ]
        preds = [pred for pred, _ in outputs]
        if not return_attention:
            return preds

        assert text_attn is not None
        cross_attn = [attn for _, attn in outputs if attn is not None]
        return preds, text_attn, cross_attn
