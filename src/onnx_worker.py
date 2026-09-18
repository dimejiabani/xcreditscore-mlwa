from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import tensorflow as tf
import tf_keras as keras
import tf2onnx

workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root))
import src.predictor  # noqa: F401


@keras.utils.register_keras_serializable(package="xcreditscore")
class FeatureSlice(keras.layers.Layer):
    def __init__(self, index: int, invert: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
        self.invert = bool(invert)

    def call(self, inputs):
        x = tf.gather(inputs, indices=[self.index], axis=1)
        if self.invert:
            x = 1.0 - x
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index, "invert": self.invert})
        return cfg


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: onnx_worker.py <keras_path> <onnx_path> <n_features>")

    keras_path = sys.argv[1]
    onnx_path = sys.argv[2]
    n_features = int(sys.argv[3])

    model = keras.models.load_model(
        keras_path,
        safe_mode=False,
        compile=False,
        custom_objects={"FeatureSlice": FeatureSlice},
    )
    _ = tf2onnx.convert.from_keras(
        model,
        input_signature=(
            tf.TensorSpec([None, n_features], tf.float32, name="features"),
        ),
        output_path=onnx_path,
    )
    print("onnx_export_ok")


if __name__ == "__main__":
    main()
