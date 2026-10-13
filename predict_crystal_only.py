"""Release CLI around the original crystal-only CGCNN prediction kernel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from crystal_cgcnn_smoke import CrystalGraphCache
from crystal_dos_privileged import OfficialCgcnnEncoder
from crystal_dos_privileged_v3 import PrivilegedCrystalDosModelV3
from official_cgcnn_iid import build_atom_feature_lookup
from outer_target_blind_cgcnn_predict import predict_target_blind_cgcnn


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-model-py", type=Path, default=root / "cgcnn_model.py")
    parser.add_argument("--atom-init-json", type=Path, default=root / "atom_init.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "crystal_dos_low_label_loco_checkpoint_v1":
        raise ValueError("expected a checkpoint from the matched low-label runner")
    normalizers = checkpoint["normalizers"]
    model = PrivilegedCrystalDosModelV3(OfficialCgcnnEncoder(args.official_model_py))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device)
    atom_lookup, _ = build_atom_feature_lookup(args.atom_init_json)
    material_ids = np.load(args.graph_cache_dir / "material_ids.npy", allow_pickle=False)
    cache = CrystalGraphCache(args.graph_cache_dir, expected_material_ids=material_ids)
    try:
        prediction = predict_target_blind_cgcnn(
            model, cache, atom_lookup,
            normalizers["target_mean"].numpy(),
            normalizers["target_scale"].numpy(),
            batch_size=args.batch_size, device=args.device,
        )
    finally:
        cache.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id", "k_prediction_gpa", "g_prediction_gpa"])
        writer.writerows((str(mid), float(row[0]), float(row[1]))
                         for mid, row in zip(material_ids, prediction, strict=True))


if __name__ == "__main__":
    main()
