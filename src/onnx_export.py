from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def _patch_numpy_for_tf2onnx() -> None:
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "object"):
        np.object = object


def export_keras_to_onnx(keras_model, sample_input: np.ndarray, output_path: Path) -> tuple[bool, str]:
    try:
        _patch_numpy_for_tf2onnx()
        import tf2onnx
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_features = int(sample_input.shape[1])
        try:
            _ = tf2onnx.convert.from_keras(
                keras_model,
                input_signature=(
                    tf.TensorSpec([None, n_features], tf.float32, name="features"),
                ),
                output_path=str(output_path),
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                return True, "ok_direct"
        except Exception as direct_exc:
            direct_error = repr(direct_exc)

        temp_saved_model_dir = output_path.parent / "_temp_saved_model"
        if temp_saved_model_dir.exists():
            import shutil

            shutil.rmtree(temp_saved_model_dir, ignore_errors=True)

        tf.saved_model.save(keras_model, str(temp_saved_model_dir))

        cli_cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(temp_saved_model_dir),
            "--output",
            str(output_path),
            "--opset",
            "13",
        ]
        cli_proc = subprocess.run(
            cli_cmd,
            capture_output=True,
            text=True,
        )
        if cli_proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True, "ok_savedmodel_cli"
        cli_error = cli_proc.stderr or cli_proc.stdout

        temp_model_path = output_path.parent / "_temp_lattice_model.keras"
        keras_model.save(temp_model_path)

        worker = Path(__file__).resolve().parent / "onnx_worker.py"
        cmd = [
            sys.executable,
            str(worker),
            str(temp_model_path),
            str(output_path),
            str(n_features),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True, "ok_worker"
        worker_error = proc.stderr or proc.stdout
        if "direct_error" in locals():
            return False, (
                f"direct_failed: {direct_error} | "
                f"savedmodel_cli_failed: {cli_error} | "
                f"worker_failed: {worker_error}"
            )
        return False, f"savedmodel_cli_failed: {cli_error} | worker_failed: {worker_error}"
    except Exception as exc:
        return False, repr(exc)


def run_onnx_parity_smoke(
    native_model,
    sample_input: np.ndarray,
    onnx_path: Path,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    if not onnx_path.exists():
        return {
            "ok": False,
            "status": "onnx_missing",
            "rows": 0,
        }

    try:
        import onnxruntime as ort
    except Exception as exc:
        return {
            "ok": False,
            "status": f"onnxruntime_unavailable:{type(exc).__name__}",
            "rows": int(sample_input.shape[0]),
        }

    try:
        X = np.asarray(sample_input, dtype=np.float32)
        native = np.asarray(native_model.predict_proba(X)[:, 1], dtype=float).reshape(-1)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_out = session.run([output_name], {input_name: X})[0]
        onnx_arr = np.asarray(onnx_out)
        if onnx_arr.ndim == 2 and onnx_arr.shape[1] >= 2:
            onnx_scores = onnx_arr[:, 1].reshape(-1)
        else:
            onnx_scores = onnx_arr.reshape(-1)

        n = int(min(len(native), len(onnx_scores)))
        if n == 0:
            return {
                "ok": False,
                "status": "no_predictions",
                "rows": 0,
            }

        abs_err = np.abs(native[:n] - onnx_scores[:n])
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))
        return {
            "ok": bool(max_abs_err <= float(tolerance)),
            "status": "ok" if max_abs_err <= float(tolerance) else "parity_exceeds_tolerance",
            "rows": int(n),
            "tolerance": float(tolerance),
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": f"parity_failed:{type(exc).__name__}",
            "rows": int(sample_input.shape[0]),
        }
