import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

NORMALIZED_AEROSOL_TYPES = (
    "continental",
    "maritime",
    "urban",
    "desert",
)

NORMALIZED_ATM_PROFILES = (
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
)

CONTINUOUS_FIELDS = (
    "sza_deg",
    "vza_deg",
    "raa_deg",
    "aod550",
    "cwv_cm",
    "o3_cm",
    "elev_km",
)


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_range(value: str) -> Tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid range `{value}`. Expected `min,max`.")
    lo = float(parts[0].strip())
    hi = float(parts[1].strip())
    if hi < lo:
        raise ValueError(f"Invalid range `{value}`: max < min.")
    return lo, hi


def normalize_label(value: object, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Missing required normalized label for `{field_name}`.")
    return text


def lhs_unit(n: int, dims: int, rng: random.Random) -> List[List[float]]:
    if n <= 0:
        return []
    cols: List[List[float]] = []
    for _ in range(dims):
        perm = list(range(n))
        rng.shuffle(perm)
        col = [((perm[i] + rng.random()) / n) for i in range(n)]
        cols.append(col)
    return [[cols[d][i] for d in range(dims)] for i in range(n)]


def scale_unit(x: float, lo: float, hi: float) -> float:
    return lo + x * (hi - lo)


def state_id_from_state(state: Dict[str, object], include_solver: bool, include_nstr: bool) -> str:
    payload: Dict[str, object] = {
        "sza_deg": state["sza_deg"],
        "vza_deg": state["vza_deg"],
        "raa_deg": state["raa_deg"],
        "aod550": state["aod550"],
        "cwv_cm": state["cwv_cm"],
        "o3_cm": state["o3_cm"],
        "elev_km": state["elev_km"],
        "aerosol_type": state["aerosol_type"],
        "atm_profile": state["atm_profile"],
    }
    if include_solver:
        payload["solver"] = state["solver"]
    if include_nstr:
        payload["nstr"] = state["nstr"]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_split_by_index(n: int, seed: int) -> List[str]:
    rng = random.Random(seed + 1)
    order = list(range(n))
    rng.shuffle(order)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split_by_index = ["test"] * n
    for idx in order[:n_train]:
        split_by_index[idx] = "train"
    for idx in order[n_train:n_train + n_val]:
        split_by_index[idx] = "val"
    return split_by_index


def init_summary_counters() -> Dict[str, object]:
    split_counts = {"train": 0, "val": 0, "test": 0}
    aerosol_counts = {a: 0 for a in NORMALIZED_AEROSOL_TYPES}
    atm_counts = {a: 0 for a in NORMALIZED_ATM_PROFILES}
    combo_counts: Dict[str, int] = {}
    min_max: Dict[str, Dict[str, float]] = {
        field: {"min": float("inf"), "max": float("-inf")} for field in CONTINUOUS_FIELDS
    }
    return {
        "split_counts": split_counts,
        "aerosol_counts": aerosol_counts,
        "atm_counts": atm_counts,
        "combo_counts": combo_counts,
        "min_max": min_max,
    }


def update_summary_counters(counters: Dict[str, object], row: Dict[str, object]) -> None:
    split_counts = counters["split_counts"]
    aerosol_counts = counters["aerosol_counts"]
    atm_counts = counters["atm_counts"]
    combo_counts = counters["combo_counts"]
    min_max = counters["min_max"]

    split = str(row["split"])
    split_counts[split] = split_counts.get(split, 0) + 1

    aerosol = str(row["aerosol_type"])
    profile = str(row["atm_profile"])
    aerosol_counts[aerosol] = aerosol_counts.get(aerosol, 0) + 1
    atm_counts[profile] = atm_counts.get(profile, 0) + 1
    combo_key = f"{aerosol}|{profile}"
    combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1

    for field in CONTINUOUS_FIELDS:
        v = float(row[field])
        min_max[field]["min"] = min(min_max[field]["min"], v)
        min_max[field]["max"] = max(min_max[field]["max"], v)


def finalize_summary(
    n_states: int,
    counters: Dict[str, object],
    include_solver: bool,
    include_nstr: bool,
) -> Dict[str, object]:
    min_max = counters["min_max"]
    for field in CONTINUOUS_FIELDS:
        if min_max[field]["min"] == float("inf"):
            min_max[field] = {"min": None, "max": None}

    return {
        "unique_states": n_states,
        "split_counts": counters["split_counts"],
        "counts_per_aerosol_type": counters["aerosol_counts"],
        "counts_per_atm_profile": counters["atm_counts"],
        "counts_per_aerosol_atm_combination": counters["combo_counts"],
        "continuous_min_max": min_max,
        "state_id_includes_solver": bool(include_solver),
        "state_id_includes_nstr": bool(include_nstr),
    }


def print_example_commands() -> None:
    print("Example commands:")
    print("")
    print("# Smoke manifest (200 states)")
    print(
        "python data_generator/generate_shared_states.py "
        "--n_states 200 --output_dir data/shared_states_smoke_200"
    )
    print("")
    print("# Medium manifest (1000 states)")
    print(
        "python data_generator/generate_shared_states.py "
        "--n_states 1000 --output_dir data/shared_states_medium_1k"
    )
    print("")
    print("# Full manifest (100000 states)")
    print(
        "python data_generator/generate_shared_states.py "
        "--n_states 100000 --output_dir data/shared_states_100k"
    )
    print("")
    print("# libRadtran from manifest (smoke 3 bands)")
    print(
        "python data_generator/generate_libradtran_dataset.py "
        "--states_manifest data/shared_states_smoke_200/state_manifest.jsonl "
        "--bands B2,B3,B4 --output_dir data/generated_libradtran_smoke_200_3b"
    )
    print("")
    print("# 6S from manifest (smoke 3 bands)")
    print(
        "python data_generator/generate_6s_dataset.py "
        "--states_manifest data/shared_states_smoke_200/state_manifest.jsonl "
        "--bands B2,B3,B4 --output_dir data/generated_6s_smoke_200_3b"
    )


def build_parser() -> argparse.ArgumentParser:
    description = (
        "Generate a shared normalized state manifest for RTM pipelines.\n\n"
        "This script only generates states; it does not run libRadtran or 6S.\n"
        "Intended workflow:\n"
        "  1) python data_generator/generate_shared_states.py ...\n"
        "  2) python data_generator/generate_libradtran_dataset.py --states_manifest <manifest> ...\n"
        "  3) python data_generator/generate_6s_dataset.py --states_manifest <manifest> ..."
    )
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n_states", type=int, default=100000)
    p.add_argument("--output_dir", type=str, default="data/shared_states")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sza_range", type=str, default="0,70")
    p.add_argument("--vza_range", type=str, default="0,30")
    p.add_argument("--raa_range", type=str, default="0,180")
    p.add_argument("--aod_range", type=str, default="0.02,1.5")
    p.add_argument("--cwv_range", type=str, default="0.2,6.0")
    p.add_argument("--o3_range", type=str, default="0.2,0.45")
    p.add_argument("--elev_range", type=str, default="0.0,3.0")
    p.add_argument("--aerosol_types", type=str, default="continental,maritime,urban,desert")
    p.add_argument(
        "--atm_profiles",
        type=str,
        default="tropical,midlatitude_summer,midlatitude_winter,subarctic_summer,subarctic_winter",
    )
    p.add_argument("--solvers", type=str, default="disort")
    p.add_argument("--nstr_values", type=str, default="8")
    p.add_argument("--print_example_commands", action="store_true")
    return p


def main(argv: List[str] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.n_states <= 0:
        raise ValueError("--n_states must be a positive integer.")

    if args.print_example_commands:
        print_example_commands()

    aerosol_types = [normalize_label(x, "aerosol_type") for x in parse_csv_list(args.aerosol_types)]
    atm_profiles = [normalize_label(x, "atm_profile") for x in parse_csv_list(args.atm_profiles)]
    solvers = [x.strip() for x in parse_csv_list(args.solvers)]
    nstr_values = parse_int_list(args.nstr_values)

    unsupported_aerosol = sorted(set(aerosol_types) - set(NORMALIZED_AEROSOL_TYPES))
    unsupported_profiles = sorted(set(atm_profiles) - set(NORMALIZED_ATM_PROFILES))
    if unsupported_aerosol:
        raise ValueError(
            f"Unsupported aerosol_types: {unsupported_aerosol}. Supported: {list(NORMALIZED_AEROSOL_TYPES)}"
        )
    if unsupported_profiles:
        raise ValueError(
            f"Unsupported atm_profiles: {unsupported_profiles}. Supported: {list(NORMALIZED_ATM_PROFILES)}"
        )
    if not solvers:
        raise ValueError("--solvers must include at least one value.")
    if not nstr_values:
        raise ValueError("--nstr_values must include at least one value.")

    sza_range = parse_range(args.sza_range)
    vza_range = parse_range(args.vza_range)
    raa_range = parse_range(args.raa_range)
    aod_range = parse_range(args.aod_range)
    cwv_range = parse_range(args.cwv_range)
    o3_range = parse_range(args.o3_range)
    elev_range = parse_range(args.elev_range)

    rng = random.Random(args.seed)
    bucket_keys: List[Tuple[str, str]] = [(a, p) for a in aerosol_types for p in atm_profiles]
    if not bucket_keys:
        raise ValueError("No aerosol_type × atm_profile buckets available.")

    include_solver = len(set(solvers)) > 1
    include_nstr = len(set(nstr_values)) > 1

    n_buckets = len(bucket_keys)
    base = args.n_states // n_buckets
    remainder = args.n_states % n_buckets
    bucket_targets = {bucket_keys[i]: base + (1 if i < remainder else 0) for i in range(n_buckets)}

    seen_ids = set()
    solver_counter = 0
    nstr_counter = 0
    flush_every = 500
    progress_every = 5000

    ranges = [sza_range, vza_range, raa_range, aod_range, cwv_range, o3_range, elev_range]
    split_by_index = build_split_by_index(args.n_states, seed=args.seed)
    row_index = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "state_manifest.jsonl"
    summary_path = output_dir / "summary.json"
    counters = init_summary_counters()

    with manifest_path.open("w", encoding="utf-8") as f:
        for aerosol, profile in bucket_keys:
            target = bucket_targets[(aerosol, profile)]
            created = 0
            attempt = 0
            while created < target:
                need = target - created
                batch = max(need * 2, 64)
                design = lhs_unit(batch, 7, rng)
                for row in design:
                    solver = solvers[solver_counter % len(solvers)]
                    nstr = nstr_values[nstr_counter % len(nstr_values)]
                    solver_counter += 1
                    nstr_counter += 1
                    state = {
                        "sza_deg": scale_unit(row[0], *ranges[0]),
                        "vza_deg": scale_unit(row[1], *ranges[1]),
                        "raa_deg": scale_unit(row[2], *ranges[2]),
                        "aod550": scale_unit(row[3], *ranges[3]),
                        "cwv_cm": scale_unit(row[4], *ranges[4]),
                        "o3_cm": scale_unit(row[5], *ranges[5]),
                        "elev_km": scale_unit(row[6], *ranges[6]),
                        "aerosol_type": aerosol,
                        "atm_profile": profile,
                        "solver": solver,
                        "nstr": int(nstr),
                    }
                    sid = state_id_from_state(state, include_solver=include_solver, include_nstr=include_nstr)
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    split = split_by_index[row_index]
                    row_obj = {
                        **state,
                        "state_id": sid,
                        "split": split,
                        "state_id_includes_solver": include_solver,
                        "state_id_includes_nstr": include_nstr,
                    }
                    f.write(json.dumps(row_obj) + "\n")
                    row_index += 1
                    created += 1
                    update_summary_counters(counters, row_obj)
                    if (row_index % flush_every) == 0:
                        f.flush()
                    if (row_index % progress_every) == 0:
                        print(f"Progress: {row_index}/{args.n_states} states written")
                    if created >= target:
                        break
                attempt += 1
                if attempt > 50 and created < target:
                    raise RuntimeError(
                        f"Failed to fill bucket {(aerosol, profile)} with unique states after many attempts."
                    )
        f.flush()

    if row_index != args.n_states:
        raise RuntimeError(f"Generated {row_index} states but expected {args.n_states}.")
    if len(seen_ids) != row_index:
        raise ValueError("Duplicate state_id values detected in generated manifest.")

    summary = finalize_summary(
        n_states=row_index,
        counters=counters,
        include_solver=include_solver,
        include_nstr=include_nstr,
    )
    summary["states_manifest"] = str(manifest_path.resolve())
    summary["seed"] = int(args.seed)
    summary["n_states_requested"] = int(args.n_states)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
