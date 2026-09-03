import json
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).parent
MARGO_ROOT = THIS_DIR.parent

VALIDATOR = THIS_DIR / "manifest_validator.py"
MANIFEST = THIS_DIR / "dataset_manifest.jsonl"
SPLIT_SUMMARY = THIS_DIR / "split_summary.json"


def run_validator(manifest_path: Path, extra_args: list[str] | None = None) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--manifest",
        str(manifest_path),
        "--margo-root",
        str(MARGO_ROOT),
        "--split-summary",
        str(SPLIT_SUMMARY),
        "--split-policy",
        str(THIS_DIR / "split_policy.json"),
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def load_manifest_lines() -> list[dict]:
    lines = MANIFEST.read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def write_manifest_lines(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def pick_one(rows, predicate):
    for r in rows:
        if predicate(r):
            return r
    raise RuntimeError("no candidate found")


def main() -> None:
    base_rows = load_manifest_lines()
    assert len(base_rows) == 2500

    base_rc, base_out = run_validator(MANIFEST)
    if base_rc != 0:
        raise RuntimeError("Base manifest failed before mutation tests:\n" + base_out)

    EXPECTED_ERRORS = {
        "delete_meta_train_row": "Manifest does not match dataset tree exactly",
        "duplicate_one_row": "Duplicate graph_id in manifest",
        "mutate_graph_id": "graph_id does not match path index",
        "swap_support_query": "stratified assignment mismatch",
        "mutate_distribution_id": "distribution_id != path distribution_id",
        "mutate_relative_path_traversal": "Manifest does not match dataset tree exactly",
        "mutate_raw_sha256": "raw_sha256 mismatch",
        "mutate_canonical_graph_sha256": "canonical_graph_sha256 mismatch",
    }

    mutations: list[tuple[str, callable]] = []

    # 1) delete one meta_train row => dataset scan mismatch / missing row => FAIL
    def mut_delete_one():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        rows.remove(victim)
        return rows

    mutations.append(("delete_meta_train_row", mut_delete_one))

    # 2) duplicate one row => duplicate graph_id/raw_sha/canonical_sha => FAIL
    def mut_duplicate_one():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        rows.append(dict(victim))
        return rows

    mutations.append(("duplicate_one_row", mut_duplicate_one))

    # 3) change graph_id string (no other fields) => path/id consistency FAIL
    def mut_graph_id():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["graph_id"].startswith("dist_1_graph_"))
        victim = dict(victim)
        victim["graph_id"] = victim["graph_id"] + "_X"
        # replace first matching instance
        for i, r in enumerate(rows):
            if r["graph_id"] == victim["graph_id"].replace("_X", ""):
                rows[i] = victim
                break
        return rows

    mutations.append(("mutate_graph_id", mut_graph_id))

    # 4) swap support/query roles inside one held-out distribution => assignment mismatch FAIL
    def mut_swap_support_query():
        rows = list(base_rows)
        # pick a held-out validation distribution
        heldout = pick_one(rows, lambda r: r["role"] == "validation_support")["distribution_id"]
        support_row = pick_one(rows, lambda r: r["distribution_id"] == heldout and r["role"] == "validation_support")
        query_row = pick_one(rows, lambda r: r["distribution_id"] == heldout and r["role"] == "validation_query")
        support_row2 = dict(support_row)
        query_row2 = dict(query_row)
        support_row2["role"] = "validation_query"
        query_row2["role"] = "validation_support"
        for i, r in enumerate(rows):
            if r["graph_id"] == support_row["graph_id"]:
                rows[i] = support_row2
            elif r["graph_id"] == query_row["graph_id"]:
                rows[i] = query_row2
        return rows

    mutations.append(("swap_support_query", mut_swap_support_query))

    # 5) change distribution_id but keep graph_id/path => path/id consistency FAIL
    def mut_distribution_id():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        victim2 = dict(victim)
        victim2["distribution_id"] = victim2["distribution_id"] + 1
        for i, r in enumerate(rows):
            if r["graph_id"] == victim["graph_id"]:
                rows[i] = victim2
                break
        return rows

    mutations.append(("mutate_distribution_id", mut_distribution_id))

    # 6) relative_path traversal => safe_join must fail => FAIL
    def mut_relative_path():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        victim2 = dict(victim)
        victim2["relative_path"] = "../" + victim2["relative_path"]
        for i, r in enumerate(rows):
            if r["graph_id"] == victim["graph_id"]:
                rows[i] = victim2
                break
        return rows

    mutations.append(("mutate_relative_path_traversal", mut_relative_path))

    # 7) mutate raw_sha256 => FAIL (raw bytes hash mismatch)
    def mut_raw_sha256():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        victim2 = dict(victim)
        victim2["raw_sha256"] = victim2["raw_sha256"][:-1] + ("0" if victim2["raw_sha256"][-1] != "0" else "1")
        for i, r in enumerate(rows):
            if r["graph_id"] == victim["graph_id"]:
                rows[i] = victim2
                break
        return rows

    mutations.append(("mutate_raw_sha256", mut_raw_sha256))

    # 8) mutate canonical_graph_sha256 => FAIL (canonical hash mismatch)
    def mut_canonical_sha256():
        rows = list(base_rows)
        victim = pick_one(rows, lambda r: r["role"] == "meta_train")
        victim2 = dict(victim)
        victim2["canonical_graph_sha256"] = victim2["canonical_graph_sha256"][:-1] + (
            "0" if victim2["canonical_graph_sha256"][-1] != "0" else "1"
        )
        for i, r in enumerate(rows):
            if r["graph_id"] == victim["graph_id"]:
                rows[i] = victim2
                break
        return rows

    mutations.append(("mutate_canonical_graph_sha256", mut_canonical_sha256))

    # Run
    for name, mut in mutations:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            mutated_path = td_path / f"manifest_mutated_{name}.jsonl"
            rows = mut()
            write_manifest_lines(mutated_path, rows)

            rc, out = run_validator(mutated_path)
            if rc == 0:
                raise RuntimeError(f"Mutation {name} DID NOT FAIL (rc=0)")
            expected = EXPECTED_ERRORS.get(name)
            if expected and expected not in out:
                raise RuntimeError(f"Mutation {name} failed for wrong reason. expected='{expected}' out='{out[:500]}'")
            print(f"OK fail: {name} (rc={rc})")

    # Path-traversal must also fail via safe_join when dataset-set scan is skipped.
    with tempfile.TemporaryDirectory() as td:
        mutated_path = Path(td) / "manifest_mutated_relative_path_traversal_nonstrict.jsonl"
        write_manifest_lines(mutated_path, mut_relative_path())
        rc, out = run_validator(mutated_path, extra_args=["--no-strict-dataset-scan"])
        if rc == 0:
            raise RuntimeError("Mutation mutate_relative_path_traversal_nonstrict DID NOT FAIL (rc=0)")
        if "relative_path" not in out:
            raise RuntimeError(
                "Mutation mutate_relative_path_traversal_nonstrict failed for wrong reason. "
                f"expected relative_path/schema failure out='{out[:500]}'"
            )
        print(f"OK fail: mutate_relative_path_traversal_nonstrict (rc={rc})")

    print("ALL mutation tests OK (all mutants failed as required).")


if __name__ == "__main__":
    main()

