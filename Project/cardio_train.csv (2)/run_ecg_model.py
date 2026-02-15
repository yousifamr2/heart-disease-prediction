import argparse
import importlib
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install with: python -m pip install torch"
    ) from exc


def _load_signal(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Unsupported input format. Use .npy or .csv/.txt")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        # [length] -> [batch, channels, length]
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        # Heuristic: treat smaller dimension as channels.
        if arr.shape[0] <= arr.shape[1]:
            arr = arr[None, :, :]
        else:
            arr = arr.T[None, :, :]
    elif arr.ndim == 3:
        # Assume [batch, channels, length]
        pass
    else:
        raise ValueError(f"Unexpected input shape: {arr.shape}")

    return torch.from_numpy(arr)


def _import_class(path: str):
    if ":" not in path:
        raise ValueError("Use --model-class in format module:ClassName")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_model(model_path: Path, model_class: str | None, model_kwargs: dict, device: str):
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model, "torchscript"
    except Exception:
        pass

    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj, "pytorch-module"

    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", obj)
        if model_class is None:
            raise ValueError(
                "Model appears to be a state_dict. Provide --model-class."
            )
        cls = _import_class(model_class)
        model = cls(**model_kwargs)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, "state_dict"

    raise ValueError("Unsupported model format.")


def main():
    parser = argparse.ArgumentParser(description="Run ECG model inference.")
    parser.add_argument(
        "--model-path",
        default="iter_0004200.pt",
        help="Path to .pt model file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ECG data (.npy or .csv/.txt).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (cpu/cuda). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help="Model class path for state_dict: module:ClassName",
    )
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help="JSON kwargs for model class init.",
    )
    parser.add_argument(
        "--save-output",
        default=None,
        help="Optional path to save output .npy",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = json.loads(args.model_kwargs)

    model, model_kind = _load_model(model_path, args.model_class, model_kwargs, device)
    x = _load_signal(input_path).to(device)

    with torch.no_grad():
        y = model(x)

    if isinstance(y, (list, tuple)):
        y = y[0]

    y_np = y.detach().cpu().numpy()
    print(f"Model kind: {model_kind}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y_np.shape)}")

    if y_np.ndim >= 2 and y_np.shape[1] <= 10:
        probs = torch.softmax(torch.from_numpy(y_np), dim=1).numpy()
        preds = probs.argmax(axis=1)
        print(f"Pred class: {preds}")

    if args.save_output:
        np.save(args.save_output, y_np)
        print(f"Saved output to: {args.save_output}")


if __name__ == "__main__":
    main()
