from __future__ import annotations

from pathlib import Path

from src.reproducibility import build_reproducibility_manifest, write_reproducibility_manifest


def test_build_and_write_reproducibility_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    artifacts = workspace / "artifacts" / "heloc_pack"
    metrics = artifacts / "metrics"
    workspace.mkdir(parents=True, exist_ok=True)

    data_path = workspace / "data.csv"
    data_path.write_text("a,b\n1,2\n", encoding="utf-8")

    manifest = build_reproducibility_manifest(
        workspace_root=workspace,
        data_path=data_path,
        artifact_root=artifacts,
        config_snapshot={"run_profile": "fast", "threshold": 0.36},
    )

    assert "config_hash_sha256" in manifest
    assert manifest["dataset"]["sha256"] is not None
    assert "dependencies" in manifest and "numpy" in manifest["dependencies"]

    out_path = write_reproducibility_manifest(metrics, manifest)
    assert out_path.exists()
