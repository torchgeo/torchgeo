# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Aurora models."""

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from torch import nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import Sample, lazy_import

if TYPE_CHECKING:
    from aurora import Batch

type _VariableMapping = Mapping[
    str, tuple[Literal['surf_vars', 'atmos_vars', 'static_vars'], str]
]

_DEFAULT_VARIABLE_MAPPING: _VariableMapping = {
    '2m_temperature': ('surf_vars', '2t'),
    '10m_u_component_of_wind': ('surf_vars', '10u'),
    '10m_v_component_of_wind': ('surf_vars', '10v'),
    'mean_sea_level_pressure': ('surf_vars', 'msl'),
    'land_sea_mask': ('static_vars', 'lsm'),
    'soil_type': ('static_vars', 'slt'),
    'geopotential_at_surface': ('static_vars', 'z'),
    'temperature': ('atmos_vars', 't'),
    'u_component_of_wind': ('atmos_vars', 'u'),
    'v_component_of_wind': ('atmos_vars', 'v'),
    'specific_humidity': ('atmos_vars', 'q'),
    'geopotential': ('atmos_vars', 'z'),
}

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
    weights: Aurora_Weights | None = None,
    *args: Any,
    variable_mapping: _VariableMapping | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Aurora model.

    Inputs and predictions use physical units. Aurora normalizes inputs internally
    and restores physical units before returning predictions.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    This model requires the following additional library to be installed:

    * `microsoft-aurora <https://pypi.org/project/microsoft-aurora/>`_ to load the models.

    .. versionadded:: 0.8

    .. versionadded:: 0.11
       The *variable_mapping* parameter.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to ``aurora.Aurora``
        variable_mapping: Complete mapping from dataset names to
            ``(category, Aurora name)`` pairs. Categories are ``'surf_vars'``,
            ``'atmos_vars'``, and ``'static_vars'``. For example,
            ``{'temperature': ('atmos_vars', 't')}``. Include every model input
            variable, including variables whose names are unchanged. If unset,
            use the standard ERA5 variables loaded by
            :class:`~torchgeo.datasets.WeatherBench2`. A supplied mapping replaces
            the default mapping in full.
            Weather dictionaries use surface inputs of shape ``(B, T, H, W)``,
            atmospheric inputs of shape ``(B, T, L, H, W)``, and shared static
            inputs of shape ``(H, W)``. Times are Unix nanoseconds, with the last
            input time used for each batch element. Predictions use dataset names
            and retain one time step. Static fields are not returned as predictions.
        **kwargs: Additional keyword arguments to pass to ``aurora.Aurora``

    Returns:
        An Aurora model.
    """
    aurora = lazy_import('aurora')
    if variable_mapping is None:
        variable_mapping = _DEFAULT_VARIABLE_MAPPING

    if weights is None:
        model = aurora.Aurora(*args, **kwargs)
    else:
        model = getattr(aurora, weights.meta['model'])(*args, **kwargs)
        model.load_checkpoint(
            repo=weights.meta['hf_repo'], name=weights.meta['filename']
        )

    model.register_forward_pre_hook(
        partial(_to_aurora_batch, variable_mapping=variable_mapping)
    )
    model.register_forward_hook(
        partial(_from_aurora_batch, variable_mapping=variable_mapping)
    )
    return cast(nn.Module, model)


def _to_aurora_batch(
    module: nn.Module, inputs: tuple[Sample], variable_mapping: _VariableMapping
) -> 'Batch':
    """Convert a weather dictionary to Aurora's batch object."""
    batch = inputs[0]
    aurora = lazy_import('aurora')
    variables: dict[str, Sample] = {
        'surf_vars': {},
        'atmos_vars': {},
        'static_vars': {},
    }
    for source, (category, name) in variable_mapping.items():
        variables[category][name] = batch[source]
    metadata = aurora.Metadata(
        lat=batch['latitude'][0],
        lon=batch['longitude'][0],
        atmos_levels=tuple(batch['level'][0].tolist()),
        time=tuple(
            datetime.fromtimestamp(time / 1e9, tz=UTC)
            for time in batch['time'][:, -1].tolist()
        ),
    )
    return aurora.Batch(**variables, metadata=metadata)


def _from_aurora_batch(
    module: nn.Module,
    inputs: tuple['Batch'],
    output: 'Batch',
    variable_mapping: _VariableMapping,
) -> Sample:
    """Return Aurora predictions under their dataset variable names."""
    return {
        source: getattr(output, category)[name]
        for source, (category, name) in variable_mapping.items()
        if category != 'static_vars'
    }
