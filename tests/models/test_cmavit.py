# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.models import CMAViT
from torchgeo.models.cmavit import CrossModalLayer, MeteorologyEmbed, TextEncoder

transformers = pytest.importorskip('transformers')

VOCAB = [
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '[MASK]',
    'corn',
    'irrigated',
    'no',
    'till',
]
TEXTS = ['corn no till', 'irrigated corn']

BATCH_SIZE = 2
IMG_SIZE = 64
PATCH_SIZE = 32
MET_CHANNELS = 4
NUM_OBSERVATIONS = 3
TEXT_MAX_LENGTH = 8
OUTPUT_SIZE = 16


class TestCMAViT:
    @pytest.fixture(autouse=True)
    def text_model(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        """Replace the pre-trained text encoder with a tiny one that needs no download."""
        vocab_file = tmp_path / 'vocab.txt'
        vocab_file.write_text('\n'.join(VOCAB))
        tokenizer = transformers.DistilBertTokenizerFast(str(vocab_file))
        config = transformers.DistilBertConfig(
            vocab_size=len(VOCAB), dim=32, n_layers=1, n_heads=2, hidden_dim=64
        )
        model = transformers.DistilBertModel(config)
        monkeypatch.setattr(
            transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer
        )
        monkeypatch.setattr(
            transformers.AutoModel, 'from_pretrained', lambda *a, **kw: model
        )

    @pytest.fixture(params=[False, True])
    def model(self, request: pytest.FixtureRequest) -> CMAViT:
        return CMAViT(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=3,
            met_channels=MET_CHANNELS,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            spatial_dim_head=16,
            timeseries=request.param,
            num_observations=NUM_OBSERVATIONS,
            text_max_length=TEXT_MAX_LENGTH,
            output_size=OUTPUT_SIZE,
        )

    @pytest.fixture
    def img(self) -> torch.Tensor:
        return torch.randn(BATCH_SIZE, NUM_OBSERVATIONS, 3, IMG_SIZE, IMG_SIZE)

    @pytest.fixture
    def met(self) -> torch.Tensor:
        return torch.randn(BATCH_SIZE, NUM_OBSERVATIONS, MET_CHANNELS)

    @torch.no_grad()
    def test_forward(self, model: CMAViT, img: torch.Tensor, met: torch.Tensor) -> None:
        preds = model(img, TEXTS, met)
        # One prediction per growing window of observations, or a single one
        assert len(preds) == (NUM_OBSERVATIONS if model.timeseries else 1)
        for pred in preds:
            assert pred.shape == (BATCH_SIZE, 1, OUTPUT_SIZE, OUTPUT_SIZE)

    @torch.no_grad()
    def test_return_attention(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        preds, text_attn, cross_attn = model(img, TEXTS, met, return_attention=True)
        assert text_attn.shape == (BATCH_SIZE, 2, TEXT_MAX_LENGTH, TEXT_MAX_LENGTH, 2)
        # One set of cross-attention weights per prediction, over its own tokens
        assert len(cross_attn) == len(preds)
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        windows = range(1, NUM_OBSERVATIONS + 1) if model.timeseries else [3]
        for attn, t in zip(cross_attn, windows):
            assert attn.shape == (BATCH_SIZE, 2, t * num_patches, TEXT_MAX_LENGTH, 2)
            # Every patch distributes its attention over the real text tokens only
            torch.testing.assert_close(
                attn.sum(dim=3), torch.ones(BATCH_SIZE, 2, t * num_patches, 2)
            )
            assert torch.all(attn[1, :, :, 4:, :] == 0)

    @torch.no_grad()
    def test_text_attention_ignores_padding(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        _, text_attn, _ = model(img, TEXTS, met, return_attention=True)
        # 'irrigated corn' is [CLS] irrigated corn [SEP] followed by padding
        assert torch.all(text_attn[1, :, :, 4:, :] == 0)

    @torch.no_grad()
    def test_time_series_uses_growing_window(
        self, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        model = CMAViT(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=3,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            spatial_dim_head=16,
            timeseries=True,
            num_observations=NUM_OBSERVATIONS,
            text_max_length=TEXT_MAX_LENGTH,
        ).eval()
        preds = model(img, TEXTS, met)
        # Later observations must not change earlier predictions
        img2, met2 = img.clone(), met.clone()
        img2[:, -1] = torch.randn_like(img2[:, -1])
        met2[:, -1] = torch.randn_like(met2[:, -1])
        preds2 = model(img2, TEXTS, met2)
        for early, early2 in zip(preds[:-1], preds2[:-1]):
            torch.testing.assert_close(early, early2)
        assert not torch.allclose(preds[-1], preds2[-1])

    def test_backward(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        preds = model(img, TEXTS, met)
        sum(pred.sum() for pred in preds).backward()

        assert model.head.fc.weight.grad is not None
        assert model.text_transformer[0].to_qkv.weight.grad is not None
        assert model.img_embed.proj.weight.grad is not None
        # The cross-attention must learn which text tokens matter
        for layer in model.cross_attn_encoder:
            assert layer.to_q.weight.grad is not None
            assert layer.to_k.weight.grad is not None
            assert layer.to_q.weight.grad.abs().sum() > 0
            assert layer.to_k.weight.grad.abs().sum() > 0
        for param in model.text_embed.model.parameters():
            assert not param.requires_grad
            assert param.grad is None

    def test_pretrained_text_encoder_stays_frozen(self, model: CMAViT) -> None:
        model.train()
        assert model.training
        assert model.text_embed.training
        assert not model.text_embed.model.training

        model.eval()
        assert not model.text_embed.model.training

    def test_init_weights(self, model: CMAViT) -> None:
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                assert torch.all(module.weight == 1)
                assert torch.all(module.bias == 0)

    @torch.no_grad()
    def test_non_square_images(self, met: torch.Tensor) -> None:
        # 2 x 2 patches per image
        model = CMAViT(
            img_size=(64, 96),
            patch_size=(32, 48),
            in_channels=3,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            spatial_dim_head=16,
            num_observations=NUM_OBSERVATIONS,
            text_max_length=TEXT_MAX_LENGTH,
        )
        img = torch.randn(BATCH_SIZE, NUM_OBSERVATIONS, 3, 64, 96)
        preds = model(img, TEXTS, met)
        assert preds[0].shape == (BATCH_SIZE, 1, 16, 16)

    @torch.no_grad()
    @pytest.mark.parametrize(('patch_size', 'met_channels'), [(8, 1), (16, 4), (32, 9)])
    def test_patches_independent_of_met_channels(
        self, patch_size: int, met_channels: int
    ) -> None:
        model = CMAViT(
            img_size=IMG_SIZE,
            patch_size=patch_size,
            in_channels=3,
            met_channels=met_channels,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            spatial_dim_head=16,
            num_observations=NUM_OBSERVATIONS,
            text_max_length=TEXT_MAX_LENGTH,
        )
        img = torch.randn(BATCH_SIZE, NUM_OBSERVATIONS, 3, IMG_SIZE, IMG_SIZE)
        met = torch.randn(BATCH_SIZE, NUM_OBSERVATIONS, met_channels)
        preds = model(img, TEXTS, met)
        assert preds[0].shape == (BATCH_SIZE, 1, 16, 16)

    @torch.no_grad()
    def test_meteorology_tokens_align_with_image_tokens(self) -> None:
        embed = MeteorologyEmbed(
            met_channels=5, embed_dim=8, dropout=0.0, num_observations=4, num_patches=3
        )
        nn.init.normal_(embed.position_embeddings)
        tokens = embed(torch.randn(BATCH_SIZE, 2, 5))
        # Class token, then 3 patches for each of the 2 timesteps
        assert tokens.shape == (BATCH_SIZE, 1 + 2 * 3, 8)
        assert torch.equal(tokens[:, 1], tokens[:, 3])
        assert torch.equal(tokens[:, 4], tokens[:, 6])
        assert not torch.equal(tokens[:, 1], tokens[:, 4])

    def test_fractional_ff_mult(self) -> None:
        model = CMAViT(
            embed_dim=768, num_layers=1, ff_mult=2048 / 768, num_observations=2
        )
        assert model.spatialmet_encoder[0].ff.net[0].out_features == 2048

    @torch.no_grad()
    def test_predictions_depend_on_text(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        model.eval()
        preds = model(img, TEXTS, met)
        preds2 = model(img, ['no till', 'corn'], met)
        assert not torch.equal(preds[-1], preds2[-1])

    @torch.no_grad()
    def test_cross_attention_weights(self) -> None:
        layer = CrossModalLayer(
            dim=32, context_dim=16, heads=2, dim_head=8, mult=2, dropout=0.0
        )
        x = torch.randn(BATCH_SIZE, 5, 32)
        context = torch.randn(BATCH_SIZE, 8, 16)
        mask = torch.tensor([[True] * 3 + [False] * 5, [True] * 8])
        out, attn = layer(x, context, mask)
        assert out.shape == x.shape
        assert attn.shape == (BATCH_SIZE, 2, 5, 8)
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones(BATCH_SIZE, 2, 5))
        assert torch.all(attn[0, ..., 3:] == 0)

    @torch.no_grad()
    def test_text_encoder_is_residual(self) -> None:
        layer = TextEncoder(dim=16, heads=2, dim_head=8)
        for param in layer.parameters():
            param.zero_()
        x = torch.randn(BATCH_SIZE, 6, 16)
        out, _ = layer(x, torch.ones(BATCH_SIZE, 6, dtype=torch.bool))
        torch.testing.assert_close(out, x)

    @torch.no_grad()
    def test_head_is_unbounded(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        # Targets may be negative, e.g., when they are standardized
        model.head.fc.bias.fill_(-5.0)
        preds = model(img, TEXTS, met)
        assert all(pred.max() < -1 for pred in preds)

    @torch.no_grad()
    @pytest.mark.parametrize('timesteps', [1, 2])
    def test_fewer_observations(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor, timesteps: int
    ) -> None:
        preds = model(img[:, :timesteps], TEXTS, met[:, :timesteps])
        assert len(preds) == (timesteps if model.timeseries else 1)

    def test_invalid_timesteps(
        self, model: CMAViT, img: torch.Tensor, met: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match='img has 3 timesteps but met has 2'):
            model(img, TEXTS, met[:, :2])
        with pytest.raises(ValueError, match=r'more than num_observations \(3\)'):
            model(torch.cat((img, img), dim=1), TEXTS, torch.cat((met, met), dim=1))
