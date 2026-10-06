# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Aurora models."""

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from torch import nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import Sample, lazy_import

if TYPE_CHECKING:
    from aurora import Batch

# Aurora operates on the raw unnormalized data.
_aurora_transforms = nn.Identity()

_aurora_meta = {
    'dataset': 'Aurora',
    'model': None,
    'resolution': None,
    'patch_size': None,
    'surf_vars': None,
    'atmos_vars': None,
    'static_vars': None,
    'architecture': '3D Swin Transformer U-Net',
    'encoder': '3D Perceiver',
    'hf_repo': 'microsoft/aurora',
    'filename': None,
    'publication': 'https://arxiv.org/abs/2409.16252',
    'repo': 'https://github.com/microsoft/aurora',
    'license': 'MIT',
    'lead-time': '6 hours',
    'units': 'degrees',
}


class Aurora_Weights(WeightsEnum):
    """Aurora weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    .. versionadded:: 0.8
    """

    HRES_T0_PRETRAINED_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-pretrained.ckpt',
            'model': 'AuroraPretrained',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_PRETRAINED_12HR_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-12h-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-12h-pretrained.ckpt',
            'model': 'Aurora12hPretrained',
            'lead-time': '12 hours',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_PRETRAINED_SMALL_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-small-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-small-pretrained.ckpt',
            'model': 'AuroraSmallPretrained',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-finetuned.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-finetuned.ckpt',
            'model': 'Aurora',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_HIGH_RES_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.1-finetuned.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.1-finetuned.ckpt',
            'model': 'AuroraHighRes',
            'resolution': 0.1,
            'patch_size': 10,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_CAMS_AIR_POLLUTION_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.4-air-pollution.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.4-air-pollution.ckpt',
            'model': 'AuroraAirPollution',
            'resolution': 0.4,
            'patch_size': 3,
            'surf_vars': (
                '2t',
                '10u',
                '10v',
                'msl',
                'pm1',
                'pm2p5',
                'pm10',
                'tcco',
                'tc_no',
                'tcno2',
                'gtco3',
                'tcso2',
            ),
            'atmos_vars': ('z', 'u', 'v', 't', 'q', 'co', 'no', 'no2', 'go3', 'so2'),
            'static_vars': (
                'lsm',
                'z',
                'slt',
                'static_ammonia',
                'static_ammonia_log',
                'static_co',
                'static_co_log',
                'static_nox',
                'static_nox_log',
                'static_so2',
                'static_so2_log',
            ),
        },
    )

    HRES_WAM0_WAVE_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-wave.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-wave.ckpt',
            'model': 'AuroraWave',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': (
                '2t',
                '10u',
                '10v',
                'msl',
                'swh',
                'mwd',
                'mwp',
                'pp1d',
                'shww',
                'mdww',
                'mpww',
                'shts',
                'mdts',
                'mpts',
                'swh1',
                'mwd1',
                'mwp1',
                'swh2',
                'mwd2',
                'mwp2',
                '10u_wave',
                '10v_wave',
                'wind',
            ),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt', 'wmb', 'lat_mask'),
        },
    )


def aurora_swin_unet(
    weights: WeightsEnum | None = None,
    *args: Any,
    variable_mapping: Mapping[str, str] | None = None,
    static_mapping: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Aurora model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    This dataset requires the following additional library to be installed:

    * `microsoft-aurora <https://pypi.org/project/microsoft-aurora/>`_ to load the models.

    .. versionadded:: 0.8

    .. versionadded:: 0.11
       The *variable_mapping* and *static_mapping* parameters.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to ``aurora.Aurora``
        variable_mapping: Mapping from dataset names to Aurora names. When supplied,
            accept flat weather dictionaries and return predictions under dataset names.
            Surface inputs have shape ``(B, T, H, W)`` and atmospheric inputs have
            shape ``(B, T, L, H, W)``. Predictions retain one time step.
        static_mapping: Mapping from static dataset names to Aurora names. Static
            inputs have shape ``(B, H, W)`` or ``(B, T, H, W)`` on a shared grid.
        **kwargs: Additional keyword arguments to pass to ``aurora.Aurora``

    Returns:
        An Aurora model.
    """
    aurora = lazy_import('aurora')

    if weights is None:
        model = aurora.Aurora(*args, **kwargs)
    else:
        model = getattr(aurora, weights.meta['model'])(*args, **kwargs)
        model.load_checkpoint(
            repo=weights.meta['hf_repo'], name=weights.meta['filename']
        )

    if variable_mapping is not None:
        model.register_forward_pre_hook(
            partial(
                _to_aurora_batch,
                variables=variable_mapping,
                static_vars=static_mapping or {},
            )
        )
        model.register_forward_hook(
            partial(_from_aurora_batch, variables=variable_mapping)
        )
    return cast(nn.Module, model)


def _to_aurora_batch(
    module: nn.Module,
    inputs: tuple[Sample, ...],
    variables: Mapping[str, str],
    static_vars: Mapping[str, str],
) -> tuple['Batch']:
    """Convert a weather dictionary to Aurora's batch object."""
    aurora = lazy_import('aurora')
    batch = inputs[0]
    surface: Sample = {}
    atmosphere: Sample = {}
    for source, name in variables.items():
        if batch[source].ndim == 4:
            surface[name] = batch[source]
        else:
            atmosphere[name] = batch[source]
    static = {
        name: batch[source][0] if batch[source].ndim == 3 else batch[source][0, 0]
        for source, name in static_vars.items()
    }
    metadata = aurora.Metadata(
        lat=batch['latitude'][0],
        lon=batch['longitude'][0],
        atmos_levels=tuple(batch['level'][0].tolist()),
        time=tuple(
            datetime.fromtimestamp(time, tz=UTC).replace(tzinfo=None)
            for time in batch['time'][:, -1].tolist()
        ),
    )
    return (aurora.Batch(surface, static, atmosphere, metadata),)


def _from_aurora_batch(
    module: nn.Module,
    inputs: tuple['Batch', ...],
    output: 'Batch',
    variables: Mapping[str, str],
) -> Sample:
    """Return Aurora predictions under their dataset variable names."""
    prediction = output.surf_vars | output.atmos_vars
    return {source: prediction[name] for source, name in variables.items()}
