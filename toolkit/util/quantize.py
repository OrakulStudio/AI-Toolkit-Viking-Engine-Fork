from fnmatch import fnmatch
from typing import List, Optional, Union, TYPE_CHECKING
import torch

from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from optimum.quanto import freeze
from tqdm import tqdm
from toolkit.print import print_acc
import os

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
]

torchao_qtypes = {
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "float8": Float8WeightOnlyConfig(),
    # [ORAKUL-STUDIO] Нативный FP8 для Ada (4090)
    # [ORAKUL-STUDIO] Чистый конфиг без лишних аргументов
    "bf8": Float8WeightOnlyConfig(weight_dtype=torch.float8_e5m2),
}

class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]

def get_qtype(qtype_val: Union[str, qtype]) -> qtype:
    if qtype_val in torchao_qtypes:
        return aotype(qtype_val)
    if isinstance(qtype_val, str):
        return qtypes[qtype_val]
    else:
        return qtype_val

def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(fnmatch(name, pattern) for pattern in include):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        try:
            if m.__class__.__name__ in Q_MODULES:
                continue
            if isinstance(weights, aotype):
                # [ORAKUL-STUDIO] Квантуем на CPU, чтобы не фрагментировать VRAM на больших рангах
                m.to(device='cpu', dtype=torch.bfloat16)
                torchao_quantize_(m, weights.config)
            else:
                _quantize_submodule(model, name, m, weights=weights, activations=activations, optimizer=optimizer)
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")

def quantize_model(base_model: "BaseModel", model_to_quantize: torch.nn.Module):
    from toolkit.dequantize import patch_dequantization_on_save
    patch_dequantization_on_save(model_to_quantize)

    q_name = base_model.model_config.qtype
    quantization_type = get_qtype(q_name)

    if q_name == 'bf8' and isinstance(quantization_type, aotype):
        quantization_type.config.weight_dtype = torch.float8_e5m2
        base_model.print_and_status_update(">>> [ORAKUL-STUDIO] BF8 NATIVE (E5M2) DETECTED - CPU PRE-QUANT MODE")

    # 1. Текстовые энкодеры
    te_list = ['text_encoder', 'text_encoder_2']
    for te_name in te_list:
        te_obj = getattr(base_model, te_name, None)
        if te_obj and isinstance(te_obj, torch.nn.Module):
            base_model.print_and_status_update(f" - quantizing {te_name}")
            quantize(te_obj, weights=quantization_type)
            freeze(te_obj)
            if base_model.model_config.low_vram:
                te_obj.to("cpu")

    # 2. Трансформер (Flux)
    all_blocks = []
    transformer_block_names = base_model.get_transformer_block_names()
    for name in transformer_block_names:
        block_list = getattr(model_to_quantize, name, None)
        if block_list is not None:
            all_blocks += list(block_list)
    
    base_model.print_and_status_update(f" - quantizing {len(all_blocks)} transformer blocks on CPU")
    for block in tqdm(all_blocks, desc="Quantizing by NIKINOV ROMAN RTX 4090"):
        # Квантуем строго на CPU перед забросом в VRAM
        quantize(block, weights=quantization_type)
        freeze(block)
        
        # Если не стоит жесткий Low VRAM, подготавливаем блок к GPU
        if not base_model.model_config.low_vram:
             block.to(base_model.device_torch)

    base_model.print_and_status_update(" - quantizing extras")
    quantize(model_to_quantize, weights=quantization_type)
    freeze(model_to_quantize)
    if not base_model.model_config.low_vram:
        model_to_quantize.to(base_model.device_torch)