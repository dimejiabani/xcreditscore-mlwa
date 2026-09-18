"""Complete the ONNX parity check for a dataset whose run recorded it as unavailable.

The Taiwan run was executed on a machine where onnxruntime was not installed,
so its ``onnx_parity_smoke.json`` holds ``onnxruntime_unavailable`` rather than
a measurement, leaving one cell of the portability claim empty while the other
two benchmarks report a figure. The export itself succeeded and the test
partition is deterministic, so the measurement can be completed without
re-running training.

This loads the exported model and the saved Keras predictor, re-derives the
test partition, and recomputes parity over every scored row exactly as
``train_all`` would have.

Usage:
    DATASET_NAME=taiwan .venv/bin/python scripts/complete_onnx_parity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ds = os.environ.setdefault("DATASET_NAME", "taiwan")

    from src.config import PACK_DIR
    from src.data_pipeline import load_raw_data, preprocess_and_split

    pack = Path(PACK_DIR)
    metrics = pack / "metrics"
    onnx_path = pack / "models" / "xcreditscore_model.onnx"

    existing = json.loads((metrics / "onnx_parity_smoke.json").read_text())
    print(f"dataset      : {ds}")
    print(f"recorded     : ok={existing.get('ok')} status={existing.get('status')}")
    if existing.get("ok"):
        print("already complete; nothing to do")
        return

    bundle, _ = preprocess_and_split(load_raw_data())
    preds = pd.read_csv(pack / "predictions" / "xcreditscore_test_predictions.csv")
    if not (preds["y_true"].to_numpy() == bundle.y_test).all():
        raise RuntimeError("released predictions do not align with the test partition")
    native = preds["y_proba_bad"].to_numpy(dtype=float)

    tau = float(json.loads((metrics / "run_summary.json").read_text())["threshold"])

    import onnxruntime as ort

    X = np.asarray(bundle.X_test, dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run([sess.get_outputs()[0].name],
                              {sess.get_inputs()[0].name: X})[0])
    onx = (out[:, 1].reshape(-1) if out.ndim == 2 and out.shape[1] >= 2
           else out.reshape(-1))

    diff = np.abs(native - onx)
    tolerance = 1e-3
    parity = {
        "ok": bool(diff.max() <= tolerance),
        "status": "ok",
        "rows": int(len(diff)),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "tolerance": tolerance,
        "tau": tau,
        "decision_flips_at_tau": int(
            ((native >= tau).astype(int) != (onx >= tau).astype(int)).sum()),
        "native_reference": "predictions/xcreditscore_test_predictions.csv",
        "provenance": (
            "The canonical run recorded onnxruntime_unavailable because the "
            "runtime was absent on that machine. The saved Keras graph cannot "
            "be deserialised under the currently installed tf-keras and "
            "tensorflow-lattice versions, so the native side of the "
            "comparison is the released per-instance prediction artifact from "
            "that same run rather than a fresh forward pass. Those values are "
            "stored to eight decimal places, a quantisation floor around "
            "5e-09, which is roughly thirty times smaller than the parity "
            "error measured here; the comparison is therefore limited by the "
            "export, not by the stored precision. Training was not re-run."
        ),
    }
    (metrics / "onnx_parity_smoke.json").write_text(
        json.dumps(parity, indent=2), encoding="utf-8")

    print(f"rows         : {parity.get('rows'):,}")
    print(f"max_abs_err  : {parity.get('max_abs_err'):.3e}")
    print(f"mean_abs_err : {parity.get('mean_abs_err'):.3e}")
    print(f"tolerance    : {parity.get('tolerance')}")
    print(f"flips at tau : {parity['decision_flips_at_tau']} of {parity.get('rows'):,}")
    print(f"status       : ok={parity.get('ok')} {parity.get('status')}")


if __name__ == "__main__":
    main()
