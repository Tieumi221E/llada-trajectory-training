from __future__ import annotations

from pathlib import Path
from typing import Optional

import json

import torch
import torch.nn as nn
from transformers import AutoModel  # type: ignore

try:
    import safetensors.torch as st  # type: ignore
except ImportError:  # pragma: no cover - runtime guard
    st = None  # type: ignore

PEFT_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from peft import PeftModel  # type: ignore

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime guard
    PeftModel = None  # type: ignore


def _get_submodule(root: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    current: torch.nn.Module = root
    for attr in path.split("."):
        if not attr:
            continue
        if attr.isdigit():
            current = current[int(attr)]  # type: ignore[index]
        else:
            if not hasattr(current, attr):
                return None
            current = getattr(current, attr)
    return current


def _apply_lora_delta(
    module: torch.nn.Module,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    alpha: float,
) -> None:
    if not isinstance(module, nn.Linear):
        return
    r = tensor_a.shape[0]
    if r == 0:
        return
    scale = alpha / float(r)
    delta = torch.matmul(tensor_b, tensor_a) * scale
    module.weight.data += delta.to(module.weight.data.dtype)


def _try_apply_safetensors_lora(
    model: torch.nn.Module, adapter_dir: Path, device: torch.device
) -> bool:
    if st is None:
        return False

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        return False

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    alpha = float(cfg.get("lora_alpha", 32.0))

    state = st.load_file(str(weights_path))
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            grouped.setdefault(base, {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            base = key[: -len(".lora_B.weight")]
            grouped.setdefault(base, {})["B"] = value

    applied, skipped = 0, 0
    for path_key, tensors in grouped.items():
        if "A" not in tensors or "B" not in tensors:
            continue
        module_path = path_key
        for prefix in ("base_model.model.", "base_model."):
            if module_path.startswith(prefix):
                module_path = module_path[len(prefix) :]
                break
        module = _get_submodule(model, module_path)
        if module is None:
            skipped += 1
            continue
        _apply_lora_delta(
            module, tensors["A"].to(device), tensors["B"].to(device), alpha
        )
        applied += 1

    if applied == 0:
        raise RuntimeError(
            f"LoRA adapter loaded but 0/{applied + skipped} modules matched the model. "
            f"Check key prefixes in {adapter_dir}/adapter_model.safetensors."
        )
    print(f"Applied LoRA to {applied} modules (skipped {skipped}) from {adapter_dir}")
    return True


def load_diffusion_model(
    model_name: str,
    *,
    device: torch.device,
    lora_path: str = "",
    merge_lora: bool = False,
) -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    lora_path = lora_path.strip()
    if not lora_path:
        return model.eval()

    adapter_dir = Path(lora_path)
    if _try_apply_safetensors_lora(model, adapter_dir, device):
        return model.eval()

    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "LoRA support requires either safetensors weights or the peft package installed."
        )

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    if merge_lora and hasattr(peft_model, "merge_and_unload"):
        peft_model = peft_model.merge_and_unload()
        peft_model = peft_model.to(device)
    return peft_model.eval()


__all__ = ["load_diffusion_model"]
