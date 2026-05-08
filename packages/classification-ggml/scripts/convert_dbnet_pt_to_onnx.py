#!/usr/bin/env python3
# Copyright (C) 2021-2026, Mindee.
#
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
"""Export DocTR DBNet detection models to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from onnxconverter_common import float16
from torch import nn
from torch.export import Dim

from doctr.models import detection


DEFAULT_ARCH = "db_mobilenet_v3_large"
DEFAULT_OUTPUT = Path("output") / f"{DEFAULT_ARCH}.onnx"
DEFAULT_QUANTIZATION = "none"
QUANTIZATION_TYPES = {
    "none": "none",
    "fp16": "fp16",
    "float16": "fp16",
}
CHECKSUM_RTOL = 1e-4
CHECKSUM_ATOL = 1e-2


class LogitsWrapper(nn.Module):
    """Return only the logits tensor from an exportable DBNet model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.model(input_tensor)
        return output["logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a DocTR DBNet detection model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        choices=["db_resnet34", "db_resnet50", DEFAULT_ARCH],
        help="DBNet architecture to export.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the generated ONNX file.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Export randomly initialized weights instead of pretrained weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size used during export.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Dummy input height used during export.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Dummy input width used during export.",
    )
    parser.add_argument(
        "--quantization",
        default=DEFAULT_QUANTIZATION,
        choices=sorted(QUANTIZATION_TYPES),
        help="ONNX post-export weight conversion mode.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Mark the batch axis as dynamic.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX model validation after export.",
    )
    return parser.parse_args()


def make_verification_input(args: argparse.Namespace) -> torch.Tensor:
    input_shape = (args.batch_size, 3, args.height, args.width)
    return torch.ones(input_shape, dtype=torch.float32)


def checksum(values: torch.Tensor | np.ndarray) -> float:
    if isinstance(values, torch.Tensor):
        return float(values.detach().cpu().to(dtype=torch.float64).sum().item())
    return float(np.asarray(values, dtype=np.float64).sum())


def verify_exported_model(
    wrapped_model: nn.Module,
    output_path: Path,
    verification_input: torch.Tensor,
) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_output = wrapped_model(verification_input)
    torch_checksum = checksum(torch_output)

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_outputs = session.run(None, {"input": verification_input.cpu().numpy()})
    if len(onnx_outputs) != 1:
        raise RuntimeError(f"Expected 1 ONNX output, got {len(onnx_outputs)}")

    onnx_checksum = checksum(onnx_outputs[0])
    print(f"PyTorch checksum: {torch_checksum}")
    print(f"ONNX checksum: {onnx_checksum}")

    if not np.isclose(
        torch_checksum,
        onnx_checksum,
        rtol=CHECKSUM_RTOL,
        atol=CHECKSUM_ATOL,
    ):
        raise RuntimeError(
            "ONNX conversion checksum mismatch: "
            f"PyTorch={torch_checksum}, ONNX={onnx_checksum}"
        )

    print("ONNX conversion succeeded: checksums match")


def export_dbnet_onnx(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be greater than zero")

    model_factory = detection.__dict__[args.arch]
    model = model_factory(pretrained=not args.no_pretrained, exportable=True)
    model.eval()

    wrapped_model = LogitsWrapper(model).eval()
    dummy_input = make_verification_input(args)

    dynamic_shapes = None
    if args.dynamic_batch:
        dynamic_shapes = {
            "input_tensor": {
                0: Dim("batch", min=1),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantization_type = QUANTIZATION_TYPES[args.quantization]
    export_path = args.output
    if quantization_type == "fp16":
        with NamedTemporaryFile(
            dir=args.output.parent,
            prefix=f"{args.output.stem}.fp32.",
            suffix=args.output.suffix,
            delete=False,
        ) as temporary_file:
            export_path = Path(temporary_file.name)

    torch.onnx.export(
        model=wrapped_model,
        args=(dummy_input,),
        f=export_path,
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )

    if not args.skip_check:
        import onnx

        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        verify_exported_model(wrapped_model, export_path, dummy_input)

    if quantization_type == "fp16":
        import onnx

        fp32_model = onnx.load(export_path)
        fp16_model = float16.convert_float_to_float16(fp32_model)
        onnx.save(fp16_model, args.output)
        export_path.unlink(missing_ok=True)
        if not args.skip_check:
            converted_model = onnx.load(args.output)
            onnx.checker.check_model(converted_model)

    return args.output


def main() -> int:
    args = parse_args()
    output_path = export_dbnet_onnx(args)
    print(f"Exported {args.arch} ({args.quantization}) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
