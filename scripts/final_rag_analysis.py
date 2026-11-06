"""Materialize adjudicated final RAG grades and reproduce aggregate analysis.
No inference requests, corpus writes, or changes to the frozen evaluator.
"""
import ast
import contextlib
import csv
import hashlib
import io
import json
import re
import runpy
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8-sig"))
def rows(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
def save(p, obj):
    (ROOT / p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def quantile(a, p=.5):
    return float(np.quantile(a, p)) if len(a) else None

plan = read("eval/final-evaluation-plan.json")
decision_file = "eval/final-review-decisions.json"
decisions = read(decision_file)
assert decisions["pending_count"] == 0
grading = {(d["question_id"], d["answer_sha256"]): d for d in decisions["decisions"]}
assert len(grading) == 321
questions = rows(plan["dataset"])
qmap = {q["id"]: q for q in questions}
corpus = read("eval/db/baseline/corpus.json")
cmap = {d["id"]: d for d in corpus}
assert len(questions) == 60 and len(corpus) == 501
assert len({d["metadata"]["title"] for d in corpus}) == 495
validation = {"frozen_hashes": {}, "runs": {}, "reference_quotes_checked": 0}
for p, k in [(plan["script"], "script_sha256"), (plan["dataset"], "dataset_sha256"),
             (plan["grading_rubric"], "grading_rubric_sha256"),
             ("dataset/gym_exercises.csv", "csv_sha256"), (plan["script_snapshot"], "script_sha256")]:
    assert sha(p) == plan[k], p
    validation["frozen_hashes"][p] = sha(p)
for q in questions:
    for e in q["reference_evidence"]:
        doc = cmap[e["source_id"]]
        if e["field"] == "metadata":
            assert json.loads(e["quote"]) == doc["metadata"]
        elif e["field"] == "full_document":
            assert e["quote"] == doc["text"]
        else:
            fields = dict(l[1:].split("] ", 1) for l in doc["text"].splitlines() if l.startswith("["))
            assert e["quote"] in fields[e["field"]]
        validation["reference_quotes_checked"] += 1
module = runpy.run_path(str(ROOT / plan["script"]), run_name="frozen_eval_analysis")
assert module["load_corpus"]("baseline") == corpus, "Preprocessing/corpus mismatch"
validation["current_preprocessing_reproduces_frozen_501_documents"] = True
for profile, name in [("baseline", "BASELINE_SYSTEM"), ("grounded", "GROUNDED_SYSTEM")]:
    assert hashlib.sha256(module[name].encode()).hexdigest() == plan["prompt_sha256"][profile]

allruns = {}
reviews_by_run = {}
seen_decisions = set()
for cond in plan["conditions"]:
    rid = cond["run_id"]
    folder = Path("eval/runs") / rid
    cfg = read(folder / "config.json")
    rr = rows(folder / "answers.jsonl")
    expected = {(q["id"], m, r) for q in questions for m in cond["methods"] for r in range(3)}
    keys = [(r["question_id"], r["method"], r["repeat"]) for r in rr]
    assert len(rr) == len(set(keys)) == 360 and set(keys) == expected
    assert all(r["status"] == "ok" for r in rr)
    expected_cfg = {"model": plan["model"], "embedding_model": plan["embedding_model"],
        "base_url": cond["base_url"], "prompt_profile": cond["profile"],
        "system_prompt_sha256": plan["prompt_sha256"][cond["profile"]],
        "options": plan["generation_options"], "think": False, "seed": 42,
        "repeats": 3, "methods": cond["methods"], "k": 5, "fetch_k": 20,
        "lambda_mult": .5, "planned_attempts": 360, "dataset_sha256": plan["dataset_sha256"],
        "script_sha256": plan["script_sha256"], "corpus": read("eval/db/baseline/manifest.json"),
        "retrieval_metric_version": plan["retrieval_metric_version"]}
    assert all(cfg.get(k) == v for k, v in expected_cfg.items())
    tags = {m["name"]: m["digest"] for m in cfg["ollama_tags"]["models"]}
    assert all(tags[m["name"]] == m["digest"] for m in plan["model_tags_at_last_development_run"]["models"])
    source_review = ROOT / folder / "review.csv"
    backup = ROOT / folder / "review.unscored.csv"
    if not backup.exists():
        with source_review.open(encoding="utf-8-sig", newline="") as f:
            initial = list(csv.DictReader(f))
        assert all(not r[k].strip() for r in initial for k in ["correct", "unsupported", "abstained"])
        shutil.copyfile(source_review, backup)
    with backup.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        original_review = list(reader)
    assert len(original_review) == 360
    review_map = {r["result_id"]: r for r in original_review}
    audit = []
    for r in rr:
        q = qmap[r["question_id"]]
        h = hashlib.sha256(r["answer"].encode()).hexdigest()
        d = grading[(q["id"], h)]
        assert d["answer"] == r["answer"] and not d["pending"]
        seen_decisions.add((q["id"], h))
        r["grade"] = {k: d[k] for k in ["correct", "unsupported", "abstained"]}
        for k, v in r["grade"].items():
            review_map[r["result_id"]][k] = str(v)
        note = d["notes"]
        docs = r["retrieved_documents"]
        for doc in docs:
            original = cmap[doc["metadata"]["source_id"]]
            assert doc["text"] == original["text"] and doc["metadata"] == original["metadata"]
        groups = q.get("relevant_groups", [[i] for i in q["source_ids"]])
        flags = [bool(set(g) & set(r["retrieved_ids"])) for g in groups]
        expected_hit = int(any(flags)) if flags and r["method"] != "no_rag" else None
        expected_recall = sum(flags) / len(flags) if flags and r["method"] != "no_rag" else None
        assert (r["hit_at_k"], r["recall_at_k"]) == (expected_hit, expected_recall)
        app_matches = list(re.finditer(r"\[근거\s*([\d,\s]+)\]", r["answer"]))
        numbers = [int(n) for m in app_matches for n in re.findall(r"\d+", m[1])]
        bracket_contents = re.findall(r"\[근거\s*([^\]]+)\]", r["answer"])
        malformed = [v for v in bracket_contents if v.strip() != "없음" and not re.fullmatch(r"[\d,\s]+", v)]
        range_style = bool(re.search(r"\[근거\s*\d+\]\s*[~\-–]\s*\[근거\s*\d+\]", r["answer"]))
        invalid = sorted({n for n in numbers if n < 1 or n > len(docs)})
        target_ids = set(q["source_ids"]) | {s for group in groups for s in group}
        cited = [{"number": n, "source_id": docs[n-1]["metadata"]["source_id"],
                  "title": docs[n-1]["metadata"]["title"],
                  "in_fixed_target_evidence": docs[n-1]["metadata"]["source_id"] in target_ids}
                 for n in sorted(set(numbers)) if 1 <= n <= len(docs)]
        missing = [i for i, group in enumerate(groups) if not set(group) & set(r["retrieved_ids"])]
        context_notes = []
        if r["method"] == "rag" and missing:
            context_notes.append("Missing fixed required evidence groups: " + str(missing))
        if q["id"] == "t043" and r["method"] == "rag":
            context_notes.append("Triceps dip facts exist in corpus but no dip document was retrieved; citing row documents does not support a dip claim.")
        if q["id"] in ("t048", "t055", "t060") and r["method"] == "rag":
            context_notes.append("Target exercise exists in corpus but is absent from the actual five documents. Provided-context absence is distinct from a global CSV-absence claim.")
        if q["id"] == "t013" and d["variant"] == 0 and r["method"] == "rag":
            context_notes.append("Citation 1 is Smith shoulder press: its generic Smith stability statement matches, while lower-chest target must come from citation 2.")
        if q["id"] == "t045" and r["method"] == "rag":
            context_notes.append("Citations to skullcrusher variants support the shared triceps claim; do not infer variant-specific effects for the exact target.")
        if q["id"] == "t039" and d["variant"] == 0 and r["method"] == "rag":
            context_notes.append("Citations 1, 2, 4 all describe lowering/tracking behind or over the head; shared position statement is supported.")
        if invalid:
            context_notes.append("Citation numbers outside actual context: " + str(invalid))
        if r["method"] == "no_rag" and (numbers or malformed):
            context_notes.append("No documents supplied: numeric/template evidence citations are fabricated.")
        if malformed or range_style:
            context_notes.append("Range/template citation syntax does not represent the full intended set in the app parser.")
        prompt_leak = "확인된 사실에만 실제 사용한 문서의" in r["answer"]
        if prompt_leak:
            context_notes.append("Grounded system instruction echoed into answer; record separately from factual correctness.")
        if context_notes:
            note += " | " + " | ".join(context_notes)
        review_map[r["result_id"]]["notes"] = note
        audit.append({"result_id": r["result_id"], "question_id": q["id"], "method": r["method"],
            "repeat": r["repeat"], "answer_sha256": h, "decision_variant": d["variant"],
            **r["grade"], "status": "adjudicated", "pending": False, "notes": note,
            "required_facts": q["required_facts"], "reference_evidence": q["reference_evidence"],
            "actual_retrieved_ids": r["retrieved_ids"], "citation_numbers_app_parser": numbers,
            "citation_out_of_range": invalid, "citation_malformed_brackets": malformed,
            "citation_range_style": range_style, "cited_documents": cited,
            "semantic_citation_review": "Manual source/content findings are in notes; target-ID membership alone is not a claim-entailment test.",
            "system_instruction_echo": prompt_leak})
    with source_review.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(original_review)
    assert len(audit) == 360 and not any(a["pending"] for a in audit)
    save(folder / "agent-review.json", {"review_date": "2026-10-29", "reviewer": decisions["reviewer"],
        "human_reviewed": False, "external_medical_validation": False, "rubric_sha256": plan["grading_rubric_sha256"],
        "decisions_sha256": sha(decision_file), "pending_count": 0,
        "counts": {"attempts": len(audit), "adjudicated": len(audit)}, "reviews": audit})
    with contextlib.redirect_stdout(io.StringIO()):
        module["summarize"](SimpleNamespace(run_id=rid))
    summary = read(folder / "summary.json")
    for method in cond["methods"]:
        subset = [r for r in rr if r["method"] == method]
        assert summary[method]["accuracy_all_attempts"] == sum(r["grade"]["correct"] for r in subset) / 180
        assert summary[method]["unsupported_answer_rate_successful"] == sum(r["grade"]["unsupported"] for r in subset) / 180
    validation["runs"][rid] = {"attempts": len(rr), "unique_expected_attempts": len(set(keys)),
        "missing": 0, "unexpected": 0, "error_count": 0, "truncation_count": 0,
        "last_result_id": rr[-1]["result_id"], "answers_sha256": sha(folder / "answers.jsonl"),
        "config_sha256": sha(folder / "config.json"), "unscored_review_sha256": sha(folder / "review.unscored.csv"),
        "graded_review_sha256": sha(folder / "review.csv"), "config_matches_plan": True,
        "all_retrieved_documents_match_corpus": True, "all_logged_retrieval_metrics_match": True,
        "started_utc": cfg["started_utc"], "pending_count": 0}
    allruns[rid] = rr
    reviews_by_run[rid] = audit
assert seen_decisions == set(grading)

groups_out = {}
for rid, rr in allruns.items():
    for method in ["no_rag", "rag"]:
        selected = [r for r in rr if r["method"] == method]
        audit = [a for a in reviews_by_run[rid] if a["method"] == method]
        group = dict(read(Path("eval/runs") / rid / "summary.json")[method])
        group.update({"correct_count": sum(r["grade"]["correct"] for r in selected),
            "unsupported_count": sum(r["grade"]["unsupported"] for r in selected),
            "abstained_count": sum(r["grade"]["abstained"] for r in selected),
            "median_load_s": quantile([r["load_s"] for r in selected]),
            "median_prompt_eval_s": quantile([r["prompt_eval_s"] for r in selected]),
            "median_decode_s": quantile([r["decode_s"] for r in selected]),
            "total_output_tokens": sum(r["output_tokens"] for r in selected),
            "citation_out_of_range_answers": sum(bool(a["citation_out_of_range"]) for a in audit),
            "citation_malformed_or_range_answers": sum(bool(a["citation_malformed_brackets"] or a["citation_range_style"]) for a in audit),
            "fabricated_citation_answers_no_context": sum(bool(re.search(r"\[근거\s*\d", r["answer"])) and method == "no_rag" for r in selected),
            "system_instruction_echo_answers": sum(a["system_instruction_echo"] for a in audit)})
        group["categories"] = {}
        for category in ["description", "metadata", "comparison", "unanswerable"]:
            subset = [r for r in selected if qmap[r["question_id"]]["category"] == category]
            group["categories"][category] = {"attempts": len(subset),
                "correct": sum(r["grade"]["correct"] for r in subset),
                "unsupported": sum(r["grade"]["unsupported"] for r in subset),
                "abstained": sum(r["grade"]["abstained"] for r in subset)}
        group["answerable_correct"] = sum(r["grade"]["correct"] for r in selected if not r["expected_abstain"])
        group["unanswerable_correct"] = sum(r["grade"]["correct"] for r in selected if r["expected_abstain"])
        groups_out[rid + "/" + method] = group
cluster_names = sorted(set(plan["question_clusters"].values()))
cluster_counts = np.array([sum(plan["question_clusters"][q["id"]] == c for q in questions) * 3 for c in cluster_names])
rng = np.random.default_rng(20261029)
bootstrap_weights = rng.multinomial(len(cluster_names), [1/len(cluster_names)] * len(cluster_names), size=10000)
denominators = bootstrap_weights @ cluster_counts

def compare(left, right, label):
    arun, amethod = left.split("/")
    brun, bmethod = right.split("/")
    a = {(r["question_id"], r["repeat"]): r for r in allruns[arun] if r["method"] == amethod}
    b = {(r["question_id"], r["repeat"]): r for r in allruns[brun] if r["method"] == bmethod}
    assert a.keys() == b.keys() and len(a) == 180
    pairs = [(a[k], b[k]) for k in sorted(a)]
    diffs = np.array([sum(y["grade"]["correct"]-x["grade"]["correct"] for x,y in pairs
                     if plan["question_clusters"][x["question_id"]] == c) for c in cluster_names])
    distribution = 100 * (bootstrap_weights @ diffs) / denominators
    same = [(x, y) for x, y in pairs if x["answer"] == y["answer"]]
    ga, gb = groups_out[left], groups_out[right]
    ret_same = sum(x["retrieved_ids"] == y["retrieved_ids"] for x,y in pairs)
    return {"label": label, "left": left, "right": right, "paired_attempts": 180,
        "accuracy_change_pp": 100*(gb["accuracy_all_attempts"]-ga["accuracy_all_attempts"]),
        "accuracy_change_cluster_bootstrap_95ci_pp": [quantile(distribution, .025), quantile(distribution, .975)],
        "unsupported_change_pp": 100*(gb["unsupported_answer_rate_successful"]-ga["unsupported_answer_rate_successful"]),
        "median_latency_reduction_pct": 100*(1-gb["median_total_s_successful"]/ga["median_total_s_successful"]),
        "p95_latency_reduction_pct": 100*(1-gb["p95_total_s_successful"]/ga["p95_total_s_successful"]),
        "median_paired_latency_delta_s": quantile([y["total_s"]-x["total_s"] for x,y in pairs]),
        "correct_improved_pairs": sum(x["grade"]["correct"] < y["grade"]["correct"] for x,y in pairs),
        "correct_regressed_pairs": sum(x["grade"]["correct"] > y["grade"]["correct"] for x,y in pairs),
        "identical_answer_pairs": len(same), "identical_retrieved_id_order_pairs": ret_same,
        "same_answer_median_left_s": quantile([x["total_s"] for x,y in same]),
        "same_answer_median_right_s": quantile([y["total_s"] for x,y in same]),
        "same_answer_output_token_pairs": sum(x["output_tokens"] == y["output_tokens"] for x,y in same)}

comparisons = []
for rid in allruns:
    comparisons.append(compare(rid+"/no_rag", rid+"/rag", "RAG effect within fixed condition"))
for method in ["no_rag", "rag"]:
    comparisons.append(compare("final-001/"+method, "final-002/"+method, "Connection address change"))
    comparisons.append(compare("final-002/"+method, "final-003/"+method, "Response policy change"))
    comparisons.append(compare("final-001/"+method, "final-003/"+method, "Combined changes"))

retrieval = []
for q in questions:
    selected = [r for rr in allruns.values() for r in rr if r["question_id"] == q["id"] and r["method"] == "rag"]
    assert len({tuple(r["retrieved_ids"]) for r in selected}) == 1
    r = selected[0]
    retrieval.append({"question_id": q["id"], "answerable": not q["expected_abstain"],
        "retrieved_ids": r["retrieved_ids"], "hit": r["hit_at_k"], "recall": r["recall_at_k"],
        "fully_retrieved": r["recall_at_k"] == 1 if not q["expected_abstain"] else None})
question_scores = []
for q in questions:
    s = {"question_id": q["id"], "category": q["category"], "question": q["question"], "conditions": {}}
    for label, rr in allruns.items():
        for method in ["no_rag", "rag"]:
            subset = [r for r in rr if r["question_id"] == q["id"] and r["method"] == method]
            s["conditions"][label+"/"+method] = {"correct": sum(r["grade"]["correct"] for r in subset),
                 "unsupported": sum(r["grade"]["unsupported"] for r in subset), "attempts": 3}
    question_scores.append(s)

result = {"review_date": "2026-10-29", "scope": "Local frozen 501-document CSV-RAG follow-up evaluation",
    "questions": 60, "repeats": 3, "total_attempts": 1080, "distinct_adjudicated_question_answers": 321,
    "pending_count": 0, "human_reviewed": False, "external_medical_validation": False,
    "groups": groups_out, "comparisons": comparisons,
    "bootstrap": {"resamples": 10000, "seed": 20261029, "clusters": 35,
      "unit": "Resample source-connected groups; retain all questions and three matched repeats within each group",
      "interval": "95% percentile; descriptive uncertainty on this designed test set, not independent expert validation"},
    "retrieval_by_question": retrieval, "question_scores": question_scores, "validation": validation,
    "decisions_sha256": sha(decision_file),
    "limitations": ["Designed 60-question set with 40 exercises and 35 connected source groups.",
      "Three repeats are not 180 independent questions per method.",
      "A was run on the previous day; fixed condition order, cache and GPU load confound causal latency attribution.",
      "Prompt latency includes output-length changes; source citations are not automatic entailment evidence.",
      "Agent created/reviewed the rubric and outputs; no independent human, medical or complete blind validation.",
      "Only local CSV search and generation; excludes app UI, query analysis, Wikipedia, routines and former GMS service.",
      "Changes were evaluated in the harness and have not been applied to app.py."]}
save("eval/final-comparison.json", result)
save("eval/final-validation.json", validation)
progress = read("eval/final-progress.json")
progress["recorded_date"] = "2026-10-29"
for rid, detail in validation["runs"].items():
    progress[rid] = {"status": "completed", "user_confirmed_normal_exit": True,
       "planned_attempts": 360, "recorded_attempts": 360, "unique_expected_attempts": 360,
       "error_count": 0, "truncation_count": 0, "last_result_id": detail["last_result_id"],
       "answers_sha256": detail["answers_sha256"], "grading_status": "completed",
       "graded_attempts": 360, "pending_count": 0, "do_not_rerun": True}
progress.update({"next_actions": [], "frozen_inputs_verified": True, "test_answer_content_reviewed": True,
    "grading_status": "completed", "comparison": "eval/final-comparison.json",
    "report": "eval/final-report.md"})
save("eval/final-progress.json", progress)
print(json.dumps({"groups": groups_out, "comparisons": comparisons,
  "reference_quotes_checked": validation["reference_quotes_checked"], "pending_count": 0}, ensure_ascii=True))
