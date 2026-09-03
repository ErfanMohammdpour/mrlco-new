import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


NODE_LINE_RE = re.compile(r'^\s*(\d+)\s*\[([^\]]*)\]\s*$', re.MULTILINE)
EDGE_LINE_RE = re.compile(r'^\s*(\d+)\s*->\s*(\d+)\s*\[([^\]]*)\]\s*$', re.MULTILINE)

SIZE_RE = re.compile(r'size\s*=\s*"?([0-9]+)"?')
EXPECT_RE = re.compile(r'expect_size\s*=\s*"?([0-9]+)"?')

PATH_RE = re.compile(r".*/offload_random20_(\d+)/random\.20\.(\d+)\.gv$")

META_RE = re.compile(
    r'--ccr\s*([0-9.]+).*?--fat\s*([0-9.]+).*?--regular\s*([0-9.]+).*?--density\s*([0-9.]+).*?'
    r'--mindata\s*([0-9]+).*?--maxdata\s*([0-9]+)',
    re.IGNORECASE,
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def eol_normalize(raw_bytes: bytes) -> bytes:
    """Deterministic EOL normalization for MARGO-DATA-v1 `raw_sha256`.

    1. CRLF (0D 0A) -> LF (0A)
    2. remaining CR (0D) -> LF (0A)
    3. no other byte transformation is permitted
    """
    return raw_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def stable_canonical_graph_sha(canon_obj: dict) -> str:
    canon_ser = json.dumps(canon_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(canon_ser.encode("utf-8"))


def parse_gv_file(gv_text: str):
    # Header meta is in a comment line like:
    # // ./daggen --dot -n 20 --ccr 0.5 --fat 0.5 --regular 0.5 --density 0.6 --mindata ... --maxdata ...
    meta_line = None
    for ln in gv_text.splitlines()[:25]:
        if ln.startswith("//") and "--ccr" in ln and "--fat" in ln and "--density" in ln:
            meta_line = ln
            break
    if not meta_line:
        m = META_RE.search(gv_text)
        if not m:
            raise ValueError("Failed to find generator metadata in .gv header")
        meta = m
    else:
        meta = META_RE.search(meta_line)
        if not meta:
            raise ValueError("Failed to parse generator metadata from header comment")

    generator_ccr = float(meta.group(1))
    generator_fat = float(meta.group(2))
    generator_regular = float(meta.group(3))
    generator_density = float(meta.group(4))
    generator_min_data_bytes = int(meta.group(5))
    generator_max_data_bytes = int(meta.group(6))

    # Parse nodes + edges
    nodes = {}  # tid -> (size, expect_size)
    edges_records = []  # (src, dst, edge_output_bytes)

    # Important: Graphviz syntax here is line-based; each node/edge is one line in provided corpus.
    for ln in gv_text.splitlines():
        ln = ln.strip("\n")
        if not ln.strip() or ln.strip().startswith("//"):
            continue

        mn = NODE_LINE_RE.match(ln)
        if mn:
            tid = int(mn.group(1))
            attr = mn.group(2)
            sm = SIZE_RE.search(attr)
            em = EXPECT_RE.search(attr)
            if sm and em:
                if tid in nodes:
                    raise ValueError(f"Duplicate task ID in .gv: {tid}")
                nodes[tid] = (int(sm.group(1)), int(em.group(1)))
            continue

        me = EDGE_LINE_RE.match(ln)
        if me:
            src = int(me.group(1))
            dst = int(me.group(2))
            attr = me.group(3)
            sm = SIZE_RE.search(attr)
            if sm:
                w = int(sm.group(1))
                edges_records.append((src, dst, w))
            continue

    if not nodes:
        raise ValueError("No nodes parsed from .gv")

    for (a, b, w) in edges_records:
        if a not in nodes or b not in nodes:
            raise ValueError(f"Edge references missing node: {a}->{b}")

    unique_edges = list(set(edges_records))  # dedup exact identical (src, dst, edge_output_bytes)
    edge_record_count = len(edges_records)
    unique_edge_count = len(unique_edges)
    duplicate_edge_record_count = edge_record_count - unique_edge_count

    # Canonical conflict check:
    # if the same (src, dst) appears with two different edge_output_bytes => fail.
    by_pair = defaultdict(set)  # (src,dst)->set(edge_output_bytes)
    for (a, b, w) in edges_records:
        by_pair[(a, b)].add(w)
    for pair, ws in by_pair.items():
        if len(ws) > 1:
            raise ValueError(f"Conflicting duplicate edges found for pair={pair} values={sorted(ws)[:5]}")

    # Degrees + DAG check on unique edges
    # Use stable node ordering by task_id to replicate canonical external_input logic.
    task_ids = sorted(nodes.keys())
    tid_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    n = len(task_ids)

    indeg = {tid: 0 for tid in task_ids}
    outdeg = {tid: 0 for tid in task_ids}
    adj = [[] for _ in range(n)]
    indeg2 = [0] * n
    for a, b, w in unique_edges:
        indeg[b] = indeg.get(b, 0) + 1
        outdeg[a] = outdeg.get(a, 0) + 1
        ia = tid_to_idx[a]
        ib = tid_to_idx[b]
        adj[ia].append(ib)
        indeg2[ib] += 1

    max_indegree_unique = max(indeg.values()) if indeg else 0
    max_outdegree_unique = max(outdeg.values()) if outdeg else 0

    # Kahn topological sort (unique edges)
    stack = [i for i in range(n) if indeg2[i] == 0]
    seen = 0
    indeg2_work = indeg2[:]
    while stack:
        u = stack.pop()
        seen += 1
        for v in adj[u]:
            indeg2_work[v] -= 1
            if indeg2_work[v] == 0:
                stack.append(v)
    is_dag = (seen == n)

    # Canonical graph hash object:
    roots = {tid for tid, val in indeg.items() if val == 0}
    canon_nodes = []
    for tid in task_ids:
        size, expect_size = nodes[tid]
        ext = size if tid in roots else 0
        canon_nodes.append(
            {
                "task_id": tid,
                "compute_workload_bytes": size,
                "task_output_bytes": expect_size,
                "external_input_bytes": ext,
            }
        )

    canon_edges = []
    for a, b, w in sorted(unique_edges, key=lambda t: (t[0], t[1], t[2])):
        canon_edges.append({"src_task_id": a, "dst_task_id": b, "edge_output_bytes": w})

    canon_obj = {
        "canonical_schema": "MARGO-GRAPH-v1",
        "nodes": canon_nodes,
        "edges": canon_edges,
    }
    canon_sha = stable_canonical_graph_sha(canon_obj)

    return {
        "generator_ccr": generator_ccr,
        "generator_fat": generator_fat,
        "generator_regular": generator_regular,
        "generator_density": generator_density,
        "generator_min_data_bytes": generator_min_data_bytes,
        "generator_max_data_bytes": generator_max_data_bytes,
        "node_count": n,
        "edge_record_count": edge_record_count,
        "unique_edge_count": unique_edge_count,
        "duplicate_edge_record_count": duplicate_edge_record_count,
        "is_dag": is_dag,
        "max_indegree_unique": max_indegree_unique,
        "max_outdegree_unique": max_outdegree_unique,
        "canonical_graph_sha256": canon_sha,
    }


def safe_join(margo_root: Path, relative_path: str) -> Path:
    # Basic path traversal protection (no absolute, no .. components)
    rp = Path(relative_path)
    if rp.is_absolute():
        raise ValueError("relative_path must be relative")
    if ".." in rp.parts:
        raise ValueError("relative_path must not contain ..")
    abs_path = (margo_root / rp).resolve()
    if not str(abs_path).startswith(str(margo_root.resolve()) + str(Path("/"))):
        raise ValueError("relative_path escapes margo_root")
    if not abs_path.exists():
        raise FileNotFoundError(str(abs_path))
    return abs_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="dataset_manifest.jsonl (or draft) path")
    ap.add_argument("--margo-root", required=False, default=None, help="Directory containing env/... (default: inferred)")
    ap.add_argument("--schema", required=False, default=None, help="Optional JSON schema path")
    ap.add_argument(
        "--split-summary",
        required=False,
        default=None,
        help="derived split_summary.json; MUST match split_policy + observed role_counts",
    )
    ap.add_argument(
        "--split-policy",
        required=False,
        default=None,
        help="split_policy.json path (normative input: split_seed + split sets + rule names)",
    )
    ap.add_argument(
        "--mode",
        required=False,
        default="draft",
        choices=["draft", "final"],
        help="draft: sidecar optional; final: sidecar MUST match",
    )
    ap.add_argument(
        "--no-strict-dataset-scan",
        action="store_true",
        default=False,
        help="Skip full dataset tree scan (not recommended for Phase 0/ADR-004 acceptance).",
    )
    args = ap.parse_args()

    spec_dir = Path(__file__).parent.resolve()
    manifest_path = Path(args.manifest).resolve()
    if args.margo_root:
        margo_root = Path(args.margo_root).resolve()
    else:
        margo_root = manifest_path.parent.parent.resolve()

    if args.mode == "final":
        sidecar_path = Path(str(manifest_path) + ".sha256").resolve()
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Sidecar hash missing in final mode: {sidecar_path}")
        expected_sidecar = sidecar_path.read_text().strip()
        actual_manifest_sha = sha256_bytes(manifest_path.read_bytes())
        if actual_manifest_sha != expected_sidecar:
            raise ValueError(f"Manifest sidecar mismatch: expected={expected_sidecar} actual={actual_manifest_sha}")

    # JSON Schema MUST be applied (safety). If caller doesn't pass --schema,
    # infer default from spec/.
    default_schema_path = spec_dir / "schemas" / "dataset-manifest.schema.json"
    schema_path = (Path(args.schema) if args.schema else default_schema_path).resolve()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text())
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for manifest validation") from exc

    schema_validator = lambda row: jsonschema.validate(instance=row, schema=schema)

    # split_policy is normative input: split_seed + distribution IDs + rule names.
    if args.split_policy:
        split_policy_path = Path(args.split_policy).resolve()
    else:
        split_policy_path = (spec_dir / "split_policy.json").resolve()
    if not split_policy_path.exists():
        raise FileNotFoundError(f"split_policy not found: {split_policy_path}")

    split_policy = json.loads(split_policy_path.read_text())
    if "split_seed" not in split_policy:
        raise ValueError(f"split_policy missing split_seed: {split_policy_path}")

    split_seed = str(split_policy["split_seed"])
    policy_split_version = str(split_policy["split_version"])
    support_count_policy = int(split_policy["support_count"])
    query_count_policy = int(split_policy["query_count"])
    n_total_policy = support_count_policy + query_count_policy

    if split_policy.get("distribution_rule") != "latin_grid_holdout_v1":
        raise ValueError(
            f"distribution_rule mismatch: {split_policy.get('distribution_rule')} != latin_grid_holdout_v1"
        )
    if split_policy.get("graph_rule") != "stratified_sha256_rank_v1":
        raise ValueError(
            f"graph_rule mismatch: {split_policy.get('graph_rule')} != stratified_sha256_rank_v1"
        )

    expected_validation_distribution_ids = set(split_policy["validation_distribution_ids"])
    expected_meta_test_distribution_ids = set(split_policy["meta_test_distribution_ids"])
    expected_meta_train_distribution_ids = set(split_policy["meta_train_distribution_ids"])

    if args.split_summary:
        split_summary_path = Path(args.split_summary).resolve()
    else:
        split_summary_path = (spec_dir / "split_summary.json").resolve()
        if args.mode != "final" and not split_summary_path.exists():
            split_summary_path = (spec_dir / "split_summary.draft.json").resolve()
    if not split_summary_path.exists():
        raise FileNotFoundError(f"split_summary not found: {split_summary_path}")
    if args.mode == "final" and split_summary_path.name.endswith(".draft.json"):
        raise ValueError("final mode forbids draft split_summary")
    split_summary = json.loads(split_summary_path.read_text())

    heldout_support_counts = defaultdict(int)
    heldout_query_counts = defaultdict(int)
    dist_roles = defaultdict(set)
    dist_graphs_seen = defaultdict(lambda: defaultdict(set))  # dist -> role -> set(graph_id)
    dist_rows = defaultdict(list)  # dist -> list[manifest_row]
    dist_gen_fat = {}  # dist -> generator_fat
    dist_gen_density = {}  # dist -> generator_density
    seen_graph_ids = set()
    seen_relative_paths = set()
    seen_raw_sha256 = set()
    seen_canonical_sha256 = set()
    manifest_relative_paths = set()

    graph_id_to_dist = {}
    errors = 0

    # Preload manifest rows (needed for split-level checks).
    rows = []
    for ln in manifest_path.read_text().splitlines():
        row = json.loads(ln)
        rows.append(row)
        manifest_relative_paths.add(row["relative_path"])

    if not args.no_strict_dataset_scan:
        # Scan dataset tree and require manifest covers it exactly.
        if not manifest_relative_paths:
            raise ValueError("Empty manifest")
        sample_rel = next(iter(manifest_relative_paths))
        if "/offload_random20_" not in sample_rel:
            raise ValueError(f"Unexpected relative_path format: {sample_rel}")
        dataset_base_rel = sample_rel.split("/offload_random20_")[0]
        dataset_base_dir = (margo_root / dataset_base_rel).resolve()
        dataset_paths = set(
            p.resolve().relative_to(margo_root).as_posix()
            for p in dataset_base_dir.rglob("*.gv")
        )
        if dataset_paths != manifest_relative_paths:
            missing = dataset_paths - manifest_relative_paths
            extra = manifest_relative_paths - dataset_paths
            raise ValueError(
                "Manifest does not match dataset tree exactly. "
                f"missing={len(missing)} extra={len(extra)}"
            )

    EXPECTED_SPEC_VERSION = "MARGO-SPEC-v0.1"
    EXPECTED_MANIFEST_VERSION = "MARGO-DATA-v1"
    EXPECTED_SPLIT_VERSION = policy_split_version

    observed_spec_versions = {r["spec_version"] for r in rows}
    if observed_spec_versions != {EXPECTED_SPEC_VERSION}:
        raise ValueError(f"spec_version mismatch: observed={sorted(observed_spec_versions)} expected={EXPECTED_SPEC_VERSION}")

    observed_manifest_versions = {r["manifest_version"] for r in rows}
    if observed_manifest_versions != {EXPECTED_MANIFEST_VERSION}:
        raise ValueError(
            f"manifest_version mismatch: observed={sorted(observed_manifest_versions)} expected={EXPECTED_MANIFEST_VERSION}"
        )

    observed_split_versions = {r["split_version"] for r in rows}
    if observed_split_versions != {EXPECTED_SPLIT_VERSION}:
        raise ValueError(
            f"split_version mismatch: observed={sorted(observed_split_versions)} expected={EXPECTED_SPLIT_VERSION}"
        )

    for r in rows:
        if not r.get("is_dag", False):
            raise ValueError(f"Graph is cyclic: {r['graph_id']}")

    # Validate each row + compute/compare derived quantities
    for row in rows:
        schema_validator(row)

        graph_id = row["graph_id"]
        rel = row["relative_path"]

        if graph_id in seen_graph_ids:
            raise ValueError(f"Duplicate graph_id in manifest: {graph_id}")
        if rel in seen_relative_paths:
            raise ValueError(f"Duplicate relative_path in manifest: {rel}")
        if row["raw_sha256"] in seen_raw_sha256:
            raise ValueError(f"Duplicate raw_sha256 in manifest: {row['raw_sha256']}")
        if row["canonical_graph_sha256"] in seen_canonical_sha256:
            raise ValueError(f"Duplicate canonical_graph_sha256 in manifest: {row['canonical_graph_sha256']}")

        seen_graph_ids.add(graph_id)
        seen_relative_paths.add(rel)
        seen_raw_sha256.add(row["raw_sha256"])
        seen_canonical_sha256.add(row["canonical_graph_sha256"])

        m = PATH_RE.match(rel)
        if not m:
            raise ValueError(f"relative_path does not match expected pattern: {rel}")
        did_from_path = int(m.group(1))
        graph_idx_from_path = int(m.group(2))
        expected_graph_id = f"dist_{did_from_path}_graph_{graph_idx_from_path}"

        if did_from_path != row["distribution_id"]:
            raise ValueError(
                f"distribution_id != path distribution_id. graph_id={graph_id} row={row['distribution_id']} path={did_from_path}"
            )
        if expected_graph_id != graph_id:
            raise ValueError(
                f"graph_id does not match path index. graph_id={graph_id} expected={expected_graph_id}"
            )

        fp = safe_join(margo_root, rel)
        raw_bytes = fp.read_bytes()
        raw_norm = eol_normalize(raw_bytes)
        raw_sha = sha256_bytes(raw_norm)
        if raw_sha != row["raw_sha256"]:
            raise ValueError(f"raw_sha256 mismatch graph_id={row['graph_id']}")

        # parse
        try:
            gv_text = raw_norm.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid UTF-8 in graph file: {fp}") from exc

        parsed = parse_gv_file(gv_text)
        if parsed["canonical_graph_sha256"] != row["canonical_graph_sha256"]:
            raise ValueError(f"canonical_graph_sha256 mismatch graph_id={row['graph_id']}")

        # cross-check numeric fields
        for k_map in [
            ("generator_ccr", "generator_ccr"),
            ("generator_fat", "generator_fat"),
            ("generator_density", "generator_density"),
            ("generator_regular", "generator_regular"),
            ("generator_min_data_bytes", "generator_min_data_bytes"),
            ("generator_max_data_bytes", "generator_max_data_bytes"),
        ]:
            pk, mk = k_map
            if abs(parsed[pk] - row[mk]) > 1e-9:
                raise ValueError(f"{mk} mismatch graph_id={row['graph_id']} parsed={parsed[pk]} manifest={row[mk]}")

        for mk in ["node_count", "edge_record_count", "unique_edge_count", "duplicate_edge_record_count", "is_dag",
                   "max_indegree_unique", "max_outdegree_unique"]:
            if parsed[mk] != row[mk]:
                raise ValueError(f"{mk} mismatch graph_id={row['graph_id']} parsed={parsed[mk]} manifest={row[mk]}")

        # distribution split checks
        did = row["distribution_id"]
        role = row["role"]
        dist_rows[did].append(row)
        dist_roles[did].add(role)
        dist_graphs_seen[did][role].add(row["graph_id"])
        graph_id_to_dist[row["graph_id"]] = did

        if did not in dist_gen_fat:
            dist_gen_fat[did] = row["generator_fat"]
        elif abs(dist_gen_fat[did] - row["generator_fat"]) > 1e-9:
            raise ValueError(f"Inconsistent generator_fat inside distribution did={did}")

        if did not in dist_gen_density:
            dist_gen_density[did] = row["generator_density"]
        elif abs(dist_gen_density[did] - row["generator_density"]) > 1e-9:
            raise ValueError(f"Inconsistent generator_density inside distribution did={did}")

        if role in ("validation_support", "meta_test_support"):
            heldout_support_counts[did] += 1
        if role in ("validation_query", "meta_test_query"):
            heldout_query_counts[did] += 1

    # ---------------------------
    # Split hard checks (ADR-004 rules)
    # ---------------------------

    def role_set_to_category(role_set: set) -> str:
        if role_set == {"meta_train"}:
            return "meta_train"
        if role_set == {"validation_support", "validation_query"}:
            return "validation"
        if role_set == {"meta_test_support", "meta_test_query"}:
            return "meta_test"
        if role_set == {"excluded"}:
            return "excluded"
        raise ValueError(f"Unexpected role set for distribution: {sorted(role_set)}")

    def grid_index(v: float) -> int:
        vr = round(float(v) + 1e-12, 1)
        vals = [0.4, 0.5, 0.6, 0.7, 0.8]
        if vr not in vals:
            raise ValueError(f"grid_index: value out of allowed set: {v}")
        return vals.index(vr)

    dist_categories = {}  # did -> category
    for did, role_set in dist_roles.items():
        dist_categories[did] = role_set_to_category(role_set)

    # latin_grid_holdout_v1: fat_index == density_index => validation
    # fat_index == (density_index + 2) mod 5 => meta_test
    # remaining => meta_train
    validation_dists = set()
    meta_test_dists = set()
    meta_train_dists = set()
    excluded_dists = set()

    for did in sorted(dist_categories.keys()):
        fat_i = grid_index(dist_gen_fat[did])
        dens_i = grid_index(dist_gen_density[did])
        if fat_i == dens_i:
            expected = "validation"
        elif fat_i == (dens_i + 2) % 5:
            expected = "meta_test"
        else:
            expected = "meta_train"

        actual = dist_categories[did]
        if actual != expected:
            raise ValueError(f"latin_grid_holdout_v1 mismatch dist={did}: expected={expected} actual={actual}")

        if actual == "validation":
            validation_dists.add(did)
        elif actual == "meta_test":
            meta_test_dists.add(did)
        elif actual == "meta_train":
            meta_train_dists.add(did)
        else:
            excluded_dists.add(did)

    if validation_dists != expected_validation_distribution_ids:
        raise ValueError("validation_distribution_ids mismatch with split_policy")
    if meta_test_dists != expected_meta_test_distribution_ids:
        raise ValueError("meta_test_distribution_ids mismatch with split_policy")
    if meta_train_dists != expected_meta_train_distribution_ids:
        raise ValueError("meta_train_distribution_ids mismatch with split_policy")

    observed_role_counts = defaultdict(int)
    for r in rows:
        observed_role_counts[r["role"]] += 1

    if str(split_summary.get("split_version")) != policy_split_version:
        raise ValueError("split_summary.split_version mismatch with split_policy")
    if str(split_summary.get("split_seed")) != split_seed:
        raise ValueError("split_summary.split_seed mismatch with split_policy")
    if set(split_summary.get("validation_distribution_ids", [])) != expected_validation_distribution_ids:
        raise ValueError("split_summary.validation_distribution_ids mismatch with split_policy")
    if set(split_summary.get("meta_test_distribution_ids", [])) != expected_meta_test_distribution_ids:
        raise ValueError("split_summary.meta_test_distribution_ids mismatch with split_policy")
    if set(split_summary.get("meta_train_distribution_ids", [])) != expected_meta_train_distribution_ids:
        raise ValueError("split_summary.meta_train_distribution_ids mismatch with split_policy")
    summary_role_counts = split_summary.get("role_counts")
    if summary_role_counts is not None:
        summary_norm = {str(k): int(v) for k, v in dict(summary_role_counts).items()}
        observed_norm = {str(k): int(v) for k, v in dict(observed_role_counts).items()}
        if summary_norm != observed_norm:
            raise ValueError(
                f"split_summary.role_counts mismatch: summary={summary_norm} observed={observed_norm}"
            )

    heldout_dists = validation_dists | meta_test_dists

    # stratified_sha256_rank_v1 re-check (exact assignment)
    split_version = rows[0]["split_version"]

    def assignment_hash(did: int, rel: str, raw_sha: str) -> str:
        s = (
            split_version
            + "\0"
            + split_seed
            + "\0"
            + str(did)
            + "\0"
            + rel
            + "\0"
            + raw_sha
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    SUPPORT_BY_CAT = {"validation": ("validation_support", "validation_query"), "meta_test": ("meta_test_support", "meta_test_query")}

    for did in sorted(heldout_dists):
        cat = dist_categories[did]
        support_role, query_role = SUPPORT_BY_CAT[cat]
        dist_row_list = dist_rows[did]
        if len(dist_row_list) != n_total_policy:
            raise ValueError(f"held-out dist size wrong: dist={did} count={len(dist_row_list)} expected={n_total_policy}")

        manifest_support_set = dist_graphs_seen[did][support_role]
        manifest_query_set = dist_graphs_seen[did][query_role]
        if len(manifest_support_set) != support_count_policy:
            raise ValueError(
                f"manifest support count wrong: dist={did} support={len(manifest_support_set)} expected={support_count_policy}"
            )
        if len(manifest_query_set) != query_count_policy:
            raise ValueError(
                f"manifest query count wrong: dist={did} query={len(manifest_query_set)} expected={query_count_policy}"
            )
        if manifest_support_set & manifest_query_set:
            raise ValueError(f"support/query overlap in manifest: dist={did}")

        strata = defaultdict(list)  # ccr_str -> rows
        for r in dist_row_list:
            ccr = round(float(r["generator_ccr"]) + 1e-12, 1)
            strata[str(ccr)].append(r)

        # expected support quotas by proportional largest remainder
        n_total = n_total_policy
        support_needed = support_count_policy
        q = {}
        floors = {}
        remainders = {}
        sum_floor = 0
        for c_str, lst in strata.items():
            n_c = len(lst)
            q_c = support_needed * n_c / n_total
            fl = int(math.floor(q_c + 1e-12))
            q[c_str] = q_c
            floors[c_str] = fl
            remainders[c_str] = q_c - fl
            sum_floor += fl

        remaining = support_needed - sum_floor
        if remaining < 0:
            raise ValueError(f"Negative remaining support slots dist={did}: remaining={remaining}")

        order = sorted(remainders.keys(), key=lambda c: (-remainders[c], float(c)))
        extra = {c: 0 for c in remainders.keys()}
        for c in order[:remaining]:
            extra[c] += 1

        expected_support_set = set()
        # pick support inside each stratum by assignment_hash ordering
        for c_str, lst in strata.items():
            k = floors[c_str] + extra.get(c_str, 0)
            lst_sorted = sorted(lst, key=lambda r: assignment_hash(did, r["relative_path"], r["raw_sha256"]))
            support_rows = lst_sorted[:k]
            for r in support_rows:
                expected_support_set.add(r["graph_id"])

        if expected_support_set != manifest_support_set:
            missing = expected_support_set - manifest_support_set
            extra_wrong = manifest_support_set - expected_support_set
            raise ValueError(f"stratified assignment mismatch dist={did} missing={len(missing)} extra_wrong={len(extra_wrong)}")

    print("OK manifest validation passed")


if __name__ == "__main__":
    main()

