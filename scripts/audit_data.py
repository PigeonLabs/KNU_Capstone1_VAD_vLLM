"""Audit all sequences, compare archive inventories, emit safe derived labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import numpy as np
from PIL import Image
from ipad_vad.data import frames_in_order, evaluation_labels, validate_labels


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/stage00"))
    p.add_argument("--derived", type=Path, default=Path("artifacts/labels_strict"))
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows, mismatches, totals, splits = [], [], defaultdict(int), {}
    with ZipFile(args.archive) as archive:
        infos = archive.infolist()
        mapping = {}
        counts = defaultdict(int)
        for info in infos:
            if "/IPAD_dataset/" in "/" + info.filename:
                key = ("/" + info.filename).split("/IPAD_dataset/", 1)[1]
                mapping[key] = info
                if key.endswith(".jpg"):
                    counts[key.rsplit("/", 1)[0]] += 1
        for scene in sorted(args.data_root.iterdir()):
            if not scene.is_dir():
                continue
            train_ids = sorted(p.name for p in (scene / "training/frames").iterdir() if p.is_dir())
            # Deterministic whole-video split, fixed before looking at test outcomes.
            ranked = sorted(train_ids, key=lambda s: digest(f"42/{scene.name}/{s}".encode()))
            ncal = max(2, round(len(ranked) * .2))
            splits[scene.name] = {"fit": sorted(ranked[ncal:]), "calibration": sorted(ranked[:ncal])}
            for partition in ("training", "testing"):
                for seq in sorted((scene / partition / "frames").iterdir()):
                    if not seq.is_dir():
                        continue
                    files = frames_in_order(seq)
                    rel = seq.relative_to(args.data_root).as_posix()
                    row = {"scene": scene.name, "partition": partition, "sequence": seq.name,
                           "relative_directory": rel, "frames": len(files),
                           "first_frame_id": 0, "last_frame_id": len(files)-1,
                           "time_unit": "source_frame_index", "fps": None, "timestamp_seconds": None,
                           "cycle_boundaries": None, "phase_ground_truth": None,
                           "lexicographic_order_differs": sorted(files) != files,
                           "filenames_sha256": digest("\n".join(f.name for f in files).encode()),
                           "archive_frames": counts[rel], "archive_sample_content_equal": True}
                    if counts[rel] != len(files):
                        raise ValueError(f"Archive frame count mismatch: {rel}")
                    dimensions = set()
                    for idx in sorted(set([0, len(files)//2, len(files)-1])):
                        content = files[idx].read_bytes()
                        if content != archive.read(mapping[f"{rel}/{files[idx].name}"]):
                            raise ValueError(f"Archive sample mismatch: {files[idx]}")
                        with Image.open(io.BytesIO(content)) as image:
                            image.load(); dimensions.add(image.size)
                    row["sample_dimensions"] = sorted(dimensions)
                    if partition == "testing":
                        label_path = scene / "test_label" / f"{int(seq.name):03d}.npy"
                        raw = label_path.read_bytes()
                        label_key = label_path.relative_to(args.data_root).as_posix()
                        if raw != archive.read(mapping[label_key]):
                            raise ValueError(f"Archive label mismatch: {label_key}")
                        labels = validate_labels(np.load(io.BytesIO(raw), allow_pickle=False))
                        out = evaluation_labels(labels, len(files))
                        row.update(label_file=label_key, label_sha256=digest(raw), label_count=len(labels),
                                   alignment_status="length_matched_index_mapping" if len(labels)==len(files) else "unresolved_quarantined",
                                   valid_evaluation_frames=int((out>=0).sum()),
                                   original_positive_labels=int(labels.sum()),
                                   anomaly_frames_in_strict_evaluation=int((out==1).sum()),
                                   anomaly_intervals_source_label_indices=np.flatnonzero(np.diff(np.r_[0, labels, 0])).reshape(-1,2).tolist(),
                                   archive_label_equal=True)
                        out_path = args.derived / scene.name / f"{seq.name}.npy"
                        out_path.parent.mkdir(parents=True, exist_ok=True); np.save(out_path, out)
                        if len(labels) != len(files):
                            mismatches.append(row.copy())
                        totals["strict_evaluation_frames"] += int((out>=0).sum())
                        totals["quarantined_frames"] += int((out<0).sum())
                        totals["strict_positive_frames"] += int((out==1).sum())
                    totals[partition+"_sequences"] += 1
                    totals[partition+"_frames"] += len(files)
                    totals["lexicographic_order_differences"] += int(row["lexicographic_order_differs"])
                    rows.append(row)
                    print(f"{rel}: {len(files)} frames", flush=True)
    summary = {"stage": "00", "status": "safe_data_interface_complete_exact_relabeling_unresolved",
               "totals": dict(totals), "unresolved_sequences": len(mismatches),
               "fps_status": "not_available_in_supplied_frame_archive_or_official_loader",
               "official_source_commit": "22764cbeeda3946303d236babdd2664fd6241b91",
               "official_loader_note": "index = frame_name * 200 // sequence_length is a relative position proxy, not semantic phase GT or cycle annotation",
               "archive": {"basename": args.archive.name, "bytes": args.archive.stat().st_size, "entries": len(infos)},
               "checks": {"all_frame_names_and_counts": True, "all_test_label_bytes_compared_to_archive": True,
                          "decoded_images": "first/middle/last per sequence only", "archive_frame_content": "first/middle/last per sequence only"},
               "policy": "Original files preserved. Equal-length labels use frame index correspondence. Entire mismatched sequences receive -1 unknown labels in primary evaluation. No truncation, stretching, inferred FPS, or test-based phase generation."}
    for name, obj in [("summary", summary), ("manifest", rows), ("mismatches", mismatches), ("splits", splits)]:
        (args.out / f"{name}.json").write_text(json.dumps(obj, indent=2)+"\n")
    with (args.out / "sequences.csv").open("w") as f:
        fields = ["scene","partition","sequence","frames","label_count","alignment_status","valid_evaluation_frames","fps","time_unit"]
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
