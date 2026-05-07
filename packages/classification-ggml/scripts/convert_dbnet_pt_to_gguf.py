#!/usr/bin/env python3
"""
Skeleton converter for DocTR db_mobilenet_v3_large DBNet (.pt) -> GGUF.

This script is intentionally minimal and focuses on conversion plumbing:
1. Instantiate a DocTR db_mobilenet_v3_large model.
2. Load a PyTorch checkpoint into the model, or load DocTR's pretrained weights.
3. Map tensor names to GGUF names.
4. Write metadata and tensors to a GGUF file.

You still need to align the metadata/tensor naming with the runtime that will
consume this GGUF file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from doctr.models import detection as doctr_detection


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFEndian, GGUFWriter, LlamaFileType  # noqa: E402


ARCH_NAME = "dbnet"
MODEL_ARCH = "db_mobilenet_v3_large"
logger = logging.getLogger("convert_dbnet_pt_to_gguf")


def get_model_factory() -> Any:
    return doctr_detection.__dict__[MODEL_ARCH]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Convert a DocTR {MODEL_ARCH} .pt checkpoint to a GGUF file.")
    parser.add_argument(
        "--model",
        type=Path,
        help=f"Path to DocTR {MODEL_ARCH} .pt checkpoint. If omitted, DocTR pretrained weights are used.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="Path to output .gguf file. Defaults to <model stem>-{ftype}.gguf, or db_mobilenet_v3_large-{ftype}.gguf.",
    )
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "auto"],
        default="f16",
        help="Target tensor storage type",
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--bigendian", action="store_true", help="Write big-endian GGUF")
    parser.add_argument("--dry-run", action="store_true", help="Do not write GGUF file")
    return parser.parse_args()


def maybe_strip_prefix(name: str) -> str:
    prefixes = ("module.", "model.", "net.")
    for pfx in prefixes:
        if name.startswith(pfx):
            return name[len(pfx) :]
    return name


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        raw_state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            raw_state = checkpoint["state_dict"]
        else:
            raw_state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if isinstance(value, torch.Tensor):
            state_dict[maybe_strip_prefix(key)] = value
    if not state_dict:
        raise ValueError("No tensors found in checkpoint/state_dict.")
    return state_dict


def infer_dbnet_hparams(state_dict: dict[str, torch.Tensor]) -> dict[str, int | str]:
    """
    Best-effort metadata inference from tensor shapes.
    """
    num_classes = -1
    head_chans = -1

    # prob_head.6 is final transposed conv: [num_classes, head_chans/4, 2, 2]
    if "prob_head.6.weight" in state_dict:
        w = state_dict["prob_head.6.weight"]
        if w.ndim == 4:
            num_classes = int(w.shape[0])
            head_chans = int(w.shape[1] * 4)

    hparams: dict[str, int | str] = {
        "detection": MODEL_ARCH,
        "num_classes": num_classes,
        "head_chans": head_chans,
    }

    # Backbone metadata (best-effort), with extra fields for MobileNet-based DBNet.
    feat_stage_ids: set[int] = set()
    for name in state_dict:
        if not name.startswith("feat_extractor."):
            continue
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            feat_stage_ids.add(int(parts[1]))

    if feat_stage_ids:
        hparams["backbone.stage_count"] = max(feat_stage_ids) + 1

    # FPN input channels correspond to the channels of extracted backbone feature maps.
    fpn_feature_channels: list[int] = []
    idx = 0
    while True:
        key = f"fpn.in_branches.{idx}.0.weight"
        if key not in state_dict:
            break
        w = state_dict[key]
        if w.ndim >= 2:
            fpn_feature_channels.append(int(w.shape[1]))
        idx += 1

    if fpn_feature_channels:
        hparams["backbone.feature_count"] = len(fpn_feature_channels)
        hparams["backbone.feature_channels"] = ",".join(str(v) for v in fpn_feature_channels)

    hparams["backbone.arch"] = "mobilenet_v3_large"
    # DBNet MobileNetV3-Large feature taps from doctr implementation.
    hparams["backbone.feature_layers"] = "3,6,12,16"

    stem_key = "feat_extractor.0.0.weight"
    if stem_key in state_dict and state_dict[stem_key].ndim >= 1:
        hparams["backbone.stem_out_channels"] = int(state_dict[stem_key].shape[0])

    last_key = "feat_extractor.16.0.weight"
    if last_key in state_dict and state_dict[last_key].ndim >= 1:
        hparams["backbone.last_out_channels"] = int(state_dict[last_key].shape[0])

    return hparams


def _to_int_or_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def map_tensor_name(pt_name: str) -> str:
    """
    Map PyTorch tensor names to GGUF tensor names.

    TODO:
    - Replace this placeholder mapping with the exact naming expected by your
      DBNet runtime.
    - If you introduce blocks/layers in a runtime-specific layout, encode that
      numbering here.
    """
    mapped = pt_name
    mapped = mapped.replace("feat_extractor.", "features.")
    mapped = mapped.replace("fpn.", "dbnet.fpn.")
    mapped = mapped.replace("prob_head.", "dbnet.prob_head.")
    mapped = mapped.replace("thresh_head.", "dbnet.thresh_head.")
    return mapped


def remap_batchnorm_tensor_name(
    pt_name: str,
    gguf_name: str,
    key_op_lookup: dict[str, tuple[str, str, str | None]],
) -> str:
    """
    Export BatchNorm affine params with ggml-style names:
      - weight -> scale
      - bias   -> shift
    """
    resolved = key_op_lookup.get(pt_name)
    if resolved is None:
        return gguf_name

    op_name, field_name, _ = resolved
    if op_name not in {"BatchNorm2d", "BatchNorm1d", "BatchNorm3d"}:
        return gguf_name

    if field_name == "weight":
        return gguf_name.removesuffix(".weight") + ".scale"
    if field_name == "bias":
        return gguf_name.removesuffix(".bias") + ".shift"
    return gguf_name


def build_state_key_op_lookup(model: torch.nn.Module) -> dict[str, tuple[str, str, str | None]]:
    """
    Build an exact state-dict key -> (module_type, field_name, op_details) lookup from the loaded model.
    """
    lookup: dict[str, tuple[str, str, str | None]] = {}
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        op_details: str | None = None
        if isinstance(module, torch.nn.Conv2d):
            is_depthwise = module.groups == module.in_channels
            op_details = (
                f"depthwise={str(is_depthwise).lower()} groups={module.groups} in_channels={module.in_channels}"
            )
        for field_name, _param in module.named_parameters(recurse=False):
            key = f"{module_name}.{field_name}" if module_name else field_name
            lookup[key] = (module_type, field_name, op_details)
        for field_name, _buffer in module.named_buffers(recurse=False):
            key = f"{module_name}.{field_name}" if module_name else field_name
            lookup[key] = (module_type, field_name, op_details)

    return lookup


def to_numpy_for_gguf(tensor: torch.Tensor, outtype: str) -> np.ndarray:
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()

    arr = t.numpy()

    if outtype == "f16" and arr.dtype in (np.float32, np.float64):
        arr = arr.astype(np.float16)
    elif outtype == "f32" and arr.dtype in (np.float16, np.float64):
        arr = arr.astype(np.float32)

    return np.ascontiguousarray(arr)


def add_metadata(writer: GGUFWriter, hparams: dict[str, Any], outtype: str) -> None:
    logger.info("Input hparams: %s", hparams)

    detection = str(hparams.get("detection", MODEL_ARCH))
    num_classes = _to_int_or_default(hparams.get("num_classes"), -1)
    head_channels = _to_int_or_default(hparams.get("head_chans"), -1)

    name = f"DBNet-{MODEL_ARCH}"
    description = f"DocTR {MODEL_ARCH} checkpoint converted from PyTorch (.pt) to GGUF (skeleton converter)."
    architecture = ARCH_NAME
    file_type = LlamaFileType.MOSTLY_F16 if outtype == "f16" else LlamaFileType.ALL_F32

    writer.add_name(name)
    writer.add_description(description)
    writer.add_string("general.architecture", architecture)
    writer.add_string("dbnet.detection", detection)
    writer.add_uint32("dbnet.num_classes", num_classes)
    writer.add_uint32("dbnet.head_channels", head_channels)
    writer.add_file_type(file_type)

    # Dump full hparams dictionary, using mobilenet namespace for MobileNet-specific keys.
    for key, value in sorted(hparams.items()):
        if key.startswith("backbone."):
            writer.add_string(f"mobilenet.hparams.{key.removeprefix('backbone.')}", str(value))
        else:
            writer.add_string(f"dbnet.hparams.{key}", str(value))

    logger.info("GGUF metadata:")
    logger.info("  general.name = %s", name)
    logger.info("  general.description = %s", description)
    logger.info("  general.architecture = %s", architecture)
    logger.info("  dbnet.detection = %s", detection)
    logger.info("  dbnet.num_classes = %d", num_classes)
    logger.info("  dbnet.head_channels = %d", head_channels)
    logger.info("  general.file_type = %s", file_type.name)
    logger.info("GGUF hparams dump:")
    for key, value in sorted(hparams.items()):
        if key.startswith("backbone."):
            logger.info("  mobilenet.hparams.%s = %s", key.removeprefix("backbone."), value)
        else:
            logger.info("  dbnet.hparams.%s = %s", key, value)

    # TODO: add all runtime-required DBNet hyperparameters here:
    # - input size policy
    # - post-processing thresholds
    # - FPN layer settings, deformable conv flag, etc.


def add_tensors(
    writer: GGUFWriter,
    state_dict: dict[str, torch.Tensor],
    outtype: str,
    key_op_lookup: dict[str, tuple[str, str, str | None]] | None = None,
) -> tuple[int, int]:
    written = 0
    skipped = 0
    unresolved = 0
    lookup = key_op_lookup or {}
    for pt_name, tensor in state_dict.items():
        if pt_name.endswith("num_batches_tracked"):
            skipped += 1
            continue

        gguf_name = map_tensor_name(pt_name)
        gguf_name = remap_batchnorm_tensor_name(pt_name, gguf_name, lookup)
        gguf_arr = to_numpy_for_gguf(tensor, outtype)
        writer.add_tensor(gguf_name, gguf_arr)
        resolved = lookup.get(pt_name)
        if resolved is None:
            module_name, _, field_name = pt_name.rpartition(".")
            logger.info("Ported tensor: %s -> %s [op=UNKNOWN field=%s module=%s]", pt_name, gguf_name, field_name, module_name)
            unresolved += 1
        else:
            op_name, field_name, op_details = resolved
            if op_details is None:
                logger.info("Ported tensor: %s -> %s [op=%s field=%s]", pt_name, gguf_name, op_name, field_name)
            else:
                logger.info(
                    "Ported tensor: %s -> %s [op=%s field=%s %s]",
                    pt_name,
                    gguf_name,
                    op_name,
                    field_name,
                    op_details,
                )
        written += 1
    if unresolved:
        logger.warning("Could not resolve operation type for %d tensors via PyTorch module API.", unresolved)
    return written, skipped


def instantiate_model(pretrained: bool) -> torch.nn.Module:
    model_factory = get_model_factory()
    try:
        return model_factory(pretrained=pretrained, pretrained_backbone=False)
    except TypeError:
        return model_factory(pretrained=pretrained)


def checkpoint_hparams(checkpoint: Any, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("hparams"), dict):
        hparams = dict(checkpoint["hparams"])
        checkpoint_detection = hparams.get("detection")
        if checkpoint_detection not in (None, MODEL_ARCH):
            raise ValueError(f"Expected {MODEL_ARCH} checkpoint hparams, got detection={checkpoint_detection!r}")
        hparams["detection"] = MODEL_ARCH
        logger.info("Using checkpoint hparams directly.")
        return hparams

    logger.info("Checkpoint hparams missing; using inferred hparams.")
    return dict(infer_dbnet_hparams(state_dict))


def load_model(checkpoint_path: Path | None) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    pretrained = checkpoint_path is None
    if pretrained:
        logger.info("No checkpoint provided; loading DocTR pretrained %s weights.", MODEL_ARCH)
    else:
        logger.info("Instantiating %s without pretrained weights.", MODEL_ARCH)

    model = instantiate_model(pretrained=pretrained)
    checkpoint: Any = None

    if checkpoint_path is not None:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        checkpoint_state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(checkpoint_state_dict, strict=True)
        logger.info("Loaded checkpoint state_dict into %s with strict=True.", MODEL_ARCH)

    model.eval()
    state_dict = dict(model.state_dict())
    hparams = checkpoint_hparams(checkpoint, state_dict)
    logger.info(
        "Detected metadata: detection=%s num_classes=%s head_chans=%s",
        hparams.get("detection"),
        hparams.get("num_classes"),
        hparams.get("head_chans"),
    )
    return model, state_dict, hparams


def write_gguf(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    hparams: dict[str, Any],
    output_type: LlamaFileType,
    fname_out: Path,
    *,
    is_big_endian: bool = False,
    dry_run: bool = False,
) -> Path:
    outtype_name = "f16" if output_type == LlamaFileType.MOSTLY_F16 else "f32"
    fname_out = Path(str(fname_out).format(ftype=outtype_name))

    if dry_run:
        logger.info("Dry run: printing sample tensor plan")
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i == 20:
                logger.info("...")
                break
            logger.info("%-64s shape=%s dtype=%s", name, tuple(tensor.shape), tensor.dtype)
        logger.info("Dry run complete")
        return fname_out

    fname_out.parent.mkdir(parents=True, exist_ok=True)

    endian = GGUFEndian.BIG if is_big_endian else GGUFEndian.LITTLE
    writer = GGUFWriter(path=str(fname_out), arch=ARCH_NAME, endianess=endian)
    add_metadata(writer, hparams, outtype_name)
    key_op_lookup = build_state_key_op_lookup(model)
    if key_op_lookup:
        logger.info("Resolved operation types for %d state-dict entries.", len(key_op_lookup))
    else:
        logger.warning("Operation-type resolution unavailable (no model lookup).")
    written, skipped = add_tensors(writer, state_dict, outtype_name, key_op_lookup)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("Done. wrote=%d skipped=%d output=%s", written, skipped, fname_out)
    return fname_out


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.model is not None and not args.model.is_file():
        logger.error(f"Error: {args.model} is not a file")
        sys.exit(1)

    ftype_map: dict[str, LlamaFileType] = {
        "f32": LlamaFileType.ALL_F32,
        "f16": LlamaFileType.MOSTLY_F16,
        "auto": LlamaFileType.MOSTLY_F16,
    }

    if args.outfile is not None:
        fname_out = args.outfile
    elif args.model is not None:
        fname_out = args.model.with_name(f"{args.model.stem}-{{ftype}}.gguf")
    else:
        fname_out = Path(f"{MODEL_ARCH}-{{ftype}}.gguf")

    model_source = args.model.name if args.model is not None else f"{MODEL_ARCH} pretrained"
    logger.info("Loading model: %s", model_source)

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model, state_dict, hparams = load_model(args.model)
        logger.info("Exporting model...")
        output_path = write_gguf(
            model,
            state_dict,
            hparams,
            output_type,
            fname_out,
            is_big_endian=args.bigendian,
            dry_run=args.dry_run,
        )
        logger.info("Model successfully exported to %s", output_path)


if __name__ == "__main__":
    main()
