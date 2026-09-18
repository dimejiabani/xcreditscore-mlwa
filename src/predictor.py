from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    LATTICE_FIXED_GROUPS,
    LATTICE_GROUPING_MODE,
    LATTICE_MAX_GROUP_DIM,
    MONOTONIC_CONSTRAINTS,
    THRESHOLD,
)


@dataclass
class PredictorResult:
    model: object
    metrics: Dict[str, float]
    y_proba: np.ndarray
    y_pred: np.ndarray
    lattice_groups: List[List[str]]
    model_family: str
    fallback_reason: str


class KerasLatticeWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y1 = self.model.predict(X, verbose=0).reshape(-1)
        y1 = np.clip(y1, 1e-6, 1 - 1e-6)
        y0 = 1.0 - y1
        return np.column_stack([y0, y1])


def _register_serializable(keras):
    @keras.utils.register_keras_serializable(package="xcreditscore")
    class FeatureSlice(keras.layers.Layer):
        def __init__(self, index: int, invert: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.index = int(index)
            self.invert = bool(invert)

        def call(self, inputs):
            import tensorflow as tf

            x = tf.gather(inputs, indices=[self.index], axis=1)
            if self.invert:
                x = 1.0 - x
            return x

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"index": self.index, "invert": self.invert})
            return cfg

    return FeatureSlice


def _mono_to_str(val: int) -> str:
    if val != 0:
        return "increasing"
    return "none"


def build_lattice_groups(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    grouping_mode: str = LATTICE_GROUPING_MODE,
    max_group_dim: int = LATTICE_MAX_GROUP_DIM,
    fixed_groups: tuple[tuple[str, ...], ...] = LATTICE_FIXED_GROUPS,
) -> List[List[str]]:
    max_group_dim = max(1, int(max_group_dim))

    if grouping_mode == "fixed":
        if not fixed_groups:
            raise ValueError("LATTICE_GROUPING_MODE='fixed' requires LATTICE_FIXED_GROUPS")

        feature_set = set(feature_names)
        grouped: List[List[str]] = []
        used = set()
        for g in fixed_groups:
            group = [name for name in g if name in feature_set and name not in used]
            if not group:
                continue
            if len(group) > max_group_dim:
                for i in range(0, len(group), max_group_dim):
                    grouped.append(group[i : i + max_group_dim])
            else:
                grouped.append(group)
            used.update(group)

        # Any unassigned features are appended as singleton groups.
        for name in feature_names:
            if name not in used:
                grouped.append([name])
        return grouped

    if grouping_mode != "correlation":
        raise ValueError(f"Unsupported grouping mode: {grouping_mode}")

    corr_scores = []
    y_centered = y_train - np.mean(y_train)
    for i, name in enumerate(feature_names):
        x = X_train[:, i]
        x_centered = x - np.mean(x)
        denom = (np.std(x_centered) * np.std(y_centered)) + 1e-8
        score = abs(float(np.mean(x_centered * y_centered) / denom))
        corr_scores.append((name, score))

    ranked = [name for name, _ in sorted(corr_scores, key=lambda x: x[1], reverse=True)]

    groups: List[List[str]] = []
    for i in range(0, len(ranked), max_group_dim):
        groups.append(ranked[i : i + max_group_dim])
    return groups


def build_predictor(feature_names: List[str]) -> HistGradientBoostingClassifier:
    monotonic_cst = [MONOTONIC_CONSTRAINTS.get(name, 0) for name in feature_names]
    model = HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=25,
        monotonic_cst=monotonic_cst,
        random_state=42,
    )
    return model


def build_lattice_predictor(
    feature_names: List[str],
    lattice_groups: List[List[str]],
    calibration_keypoints: int = 10,
    lattice_size: int = 3,
    learning_rate: float = 0.005,
) -> KerasLatticeWrapper:
    import tensorflow as tf
    import tf_keras as keras
    import tensorflow_lattice as tfl
    FeatureSlice = _register_serializable(keras)

    n_features = len(feature_names)
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    inputs = keras.Input(shape=(n_features,), name="features")

    calibrated = []
    for i, name in enumerate(feature_names):
        mono = _mono_to_str(MONOTONIC_CONSTRAINTS.get(name, 0))
        pwl = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0.0, 1.0, num=calibration_keypoints),
            output_min=0.0,
            output_max=1.0,
            monotonicity=mono,
            name=f"cal_{name}",
        )
        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        feature_slice = FeatureSlice(index=i, invert=direction < 0, name=f"slice_{name}")(inputs)
        calibrated.append(pwl(feature_slice))

    lattice_outputs: List[tf.Tensor] = []
    for g_idx, group in enumerate(lattice_groups):
        group_tensors = [calibrated[feature_to_idx[name]] for name in group]
        concat = keras.layers.Concatenate(name=f"group_concat_{g_idx}")(group_tensors)
        group_mono = [_mono_to_str(MONOTONIC_CONSTRAINTS.get(name, 0)) for name in group]
        lattice = tfl.layers.Lattice(
            lattice_sizes=[lattice_size] * len(group),
            monotonicities=group_mono,
            interpolation="simplex",
            output_min=0.0,
            output_max=1.0,
            name=f"lattice_{g_idx}",
        )
        lattice_outputs.append(lattice(concat))

    if len(lattice_outputs) > 1:
        merged = keras.layers.Concatenate(name="lattice_merged")(lattice_outputs)
    else:
        merged = lattice_outputs[0]

    # Non-negative combiner weights preserve monotonic behavior from lattice outputs.
    output = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_constraint=keras.constraints.NonNeg(),
        name="pd_output",
    )(merged)
    model = keras.Model(inputs=inputs, outputs=output, name="xcreditscore_lattice")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )
    return KerasLatticeWrapper(model)


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    decision_threshold: float = THRESHOLD,
) -> Dict[str, float]:
    y_pred = (y_proba >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    epochs: int = 30,
    batch_size: int = 256,
    calibration_keypoints: int = 10,
    lattice_size: int = 3,
    learning_rate: float = 0.005,
    early_stopping_patience: int = 5,
    grouping_mode: str = LATTICE_GROUPING_MODE,
    max_group_dim: int = LATTICE_MAX_GROUP_DIM,
    fixed_groups: tuple[tuple[str, ...], ...] = LATTICE_FIXED_GROUPS,
    decision_threshold: float = THRESHOLD,
) -> PredictorResult:
    np.random.seed(42)
    try:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tensorflow as tf

        tf.random.set_seed(42)
    except Exception:
        pass

    lattice_groups = build_lattice_groups(
        X_train,
        y_train,
        feature_names,
        grouping_mode=grouping_mode,
        max_group_dim=max_group_dim,
        fixed_groups=fixed_groups,
    )

    model_family = "tensorflow_lattice"
    fallback_reason = ""
    try:
        model = build_lattice_predictor(
            feature_names,
            lattice_groups,
            calibration_keypoints=calibration_keypoints,
            lattice_size=lattice_size,
            learning_rate=learning_rate,
        )
        callbacks = []
        if early_stopping_patience > 0:
            import tf_keras as keras

            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        model.model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    except Exception as exc:
        model_family = "fallback_hist_gradient_boosting"
        fallback_reason = repr(exc)
        model = build_predictor(feature_names)
        model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= decision_threshold).astype(int)
    metrics = evaluate_predictions(y_test, y_test_proba, decision_threshold=decision_threshold)

    return PredictorResult(
        model=model,
        metrics=metrics,
        y_proba=y_test_proba,
        y_pred=y_test_pred,
        lattice_groups=lattice_groups,
        model_family=model_family,
        fallback_reason=fallback_reason,
    )


def save_predictor(model: object, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, KerasLatticeWrapper):
        model.model.save(output_dir / "xcreditscore_lattice.keras")
    else:
        import joblib

        joblib.dump(model, output_dir / "xcreditscore_model.joblib")
