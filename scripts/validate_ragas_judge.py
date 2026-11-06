"""Source-grounded judge validation. Never writes original scores or datasets."""
from __future__ import annotations
import argparse, asyncio, csv, hashlib, importlib.metadata as metadata
import json, math, os, statistics, sys, time, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import local_ragas_eval as original
os.environ.setdefault("RAGAS_DO_NOT_TRACK","true")
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def rows(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def protected():
    paths=["eval/dev.jsonl","eval/test.jsonl","eval/grading-rubric.md","eval/final-evaluation-plan.json",
           "scripts/local_ragas_eval.py","dataset/gym_exercises.csv","eval/db/baseline/corpus.json",
           "eval/ragas/final-003.samples.jsonl","eval/ragas/final-003-judge-001/config.json",
           "eval/ragas/final-003-judge-001/scores.jsonl"]
    paths += [f"eval/runs/final-00{i}/answers.jsonl" for i in (1,2,3)]
    return {p:sha(ROOT/p) for p in paths}
def prepare(folder):
    cfg=read(ROOT/"eval/ragas/final-003-judge-001/config.json")
    assert sha(ROOT/"scripts/local_ragas_eval.py")==cfg["script_sha256"]
    qs={q["id"]:q for q in rows(ROOT/"eval/dev.jsonl")}
    docs={d["id"]:d for d in read(ROOT/"eval/db/baseline/corpus.json")}
    specs=read(ROOT/"eval/ragas/judge-validation-spec.json")
    with (ROOT/"dataset/gym_exercises.csv").open(encoding="utf-8-sig",newline="") as f:
        csvrows={f"exercise-{r['']}":r for r in csv.DictReader(f)}
    tasks=[]; cases=[]; evidence=[]
    for s in specs:
        q=qs[s["qid"]]; assert not q["expected_abstain"]
        doc=docs[q["source_ids"][0]]; cr=csvrows[doc["id"]]
        for ck,mk in [("Title","title"),("Equipment","equipment"),("Level","level"),("BodyPart","body_part"),("Type","type")]:
            assert cr[ck].strip()==doc["metadata"][mk]
        assert cr["Desc"].strip() in doc["text"]
        for check in s.get("source_checks",[]):
            assert check["quote"] in cr[check["field"]], (s["qid"],check)
        evidence.append({"qid":q["id"],"question":q["question"],"reference":q["reference_answer"],
                         "required_facts":q["required_facts"],"corpus_document":doc,"csv_row":cr,
                         "source_checks":s.get("source_checks",[]),"gold_facts":s["gold_facts"]})
        for variant,response in [("exact",q["reference_answer"]),("equivalent",s["equivalent"]),
                                 ("partial",s["partial"]),("wrong",s["wrong"])]:
            full=variant in ("exact","equivalent")
            expected=[0.90,1.0] if full else ([0.0,0.10] if variant=="wrong" else s["partial_interval"])
            c={"case_id":f"{q['id']}:{variant}","qid":q["id"],"variant":variant,
               "response":response,"reference":q["reference_answer"],"user_input":q["question"],
               "retrieved_contexts":[doc["text"]],"source_ids":q["source_ids"],
               "gold_basis":s["gold_facts"],"variant_note":s["notes"].get(variant,"")}
            cases.append(c)
            repeats=3 if q["id"] in ("d004","d016") else 1
            metrics=["factual_f1"]
            if q["id"]=="d004": metrics+=["factual_precision","factual_recall"]
            for metric in metrics:
                interval=expected
                if metric=="factual_precision" and variant=="partial": interval=[0.90,1.0]
                if metric=="factual_recall" and variant=="partial": interval=[0.35,0.65]
                for r in range(repeats):
                    tasks.append({"task_id":f"{c['case_id']}:{metric}:{r}","case_id":c["case_id"],
                        "qid":q["id"],"variant":variant,"metric":metric,"repeat":r,
                        "inputs":{"response":response,"reference":q["reference_answer"]},
                        "expected_interval":interval,"assertion":"source_grounded_fact_overlap"})
        if q["id"] not in ("d004","d016"): continue
        # Same facts with an explicit subject: a diagnostic condition, not a changed final reference.
        for r in range(3):
            tasks.append({"task_id":f"{q['id']}:explicit_reference:factual_f1:{r}",
                "case_id":f"{q['id']}:explicit_reference","qid":q["id"],"variant":"explicit_reference",
                "metric":"factual_f1","repeat":r,
                "inputs":{"response":s["equivalent"],"reference":s["explicit_reference"]},
                "expected_interval":[0.90,1.0],"assertion":"subject_clarification_diagnostic"})
        noisy=docs["exercise-676"]["text"]
        ctxfull=[doc["text"]]; ctxnone=[noisy]
        # A literal source excerpt, with no invented fact; isolates context coverage.
        assert s["partial_context"] in doc["text"]
        ctxpartial=[f"[Title] {cr['Title']}\n{s['partial_context']}"]
        ctxsets={"clean":ctxfull,"unrelated":ctxnone,"partial":ctxpartial,
                 "noise_first":[noisy,doc["text"]],"relevant_first":[doc["text"],noisy]}
        ground=[("context_precision","clean",[1,1]),("context_precision","unrelated",[0,0]),
                ("context_precision","noise_first",[.5,.5]),("context_precision","relevant_first",[1,1]),
                ("context_recall","clean",[1,1]),("context_recall","unrelated",[0,0]),
                ("context_recall","partial",[.35,.65]),
                ("faithfulness","clean",[.9,1]),("faithfulness","wrong",[0,.1]),
                ("faithfulness","unrelated",[0,.1])]
        for metric,variant,interval in ground:
            inputs={"user_input":q["question"],"retrieved_contexts":ctxsets.get(variant,ctxfull)}
            inputs["response" if metric=="faithfulness" else "reference"]=s["wrong"] if variant=="wrong" else s["equivalent"] if metric=="faithfulness" else q["reference_answer"]
            for r in range(3):
                tasks.append({"task_id":f"{q['id']}:{variant}:{metric}:{r}",
                    "case_id":f"{q['id']}:{variant}","qid":q["id"],"variant":variant,
                    "metric":metric,"repeat":r,"inputs":inputs,"expected_interval":interval,
                    "assertion":"controlled_context_perturbation"})
        for variant,response in [("correct",s["equivalent"]),("wrong",s["wrong"]),("unrelated",s["unrelated"])]:
            for r in range(3):
                tasks.append({"task_id":f"{q['id']}:{variant}:answer_relevancy:{r}",
                    "case_id":f"{q['id']}:{variant}","qid":q["id"],"variant":variant,
                    "metric":"answer_relevancy","repeat":r,
                    "inputs":{"user_input":q["question"],"response":response},
                    "expected_interval":None,"assertion":"correct_vs_unrelated_ranking_only"})
    assert len({t["task_id"] for t in tasks})==len(tasks)
    plan={"purpose":"Judge validity and stability checks; not a new chatbot benchmark",
        "created_utc":datetime.now(timezone.utc).isoformat(),"original_config":cfg,
        "protected_sha256_before":protected(),"questions":list(s["qid"] for s in specs),
        "development_only":True,"independent_human_expert_review":False,
        "synthetic_variants":True,"metric_input_cache":False,"concurrency":1,
        "criteria_frozen_before_calls":True,
        "threshold_basis":"Project diagnostic gates, not Ragas certification thresholds",
        "gates":{"all_tasks_successful":True,"fact_full_and_wrong_all_within_interval":True,
            "context_endpoint_and_ranking_all_within_interval":True,
            "repeat_max_range":0.10,"relevancy_correct_minus_unrelated_min":0.10},
        "partial_case_intervals_are_diagnostic_only":True,
        "limits":["Small controlled development sample","Gold labels authored by agent from CSV, not independent experts",
                  "Same local judge as generator","No model/prompt selection based on final test scores"],
        "task_count":len(tasks)}
    folder.mkdir(parents=True,exist_ok=False)
    dump(folder/"plan.json",plan); dump(folder/"evidence.json",evidence); dump(folder/"cases.json",cases)
    dump(folder/"tasks.json",tasks)
    dump(folder/"manifest.json",{"plan_sha256":sha(folder/"plan.json"),"tasks_sha256":sha(folder/"tasks.json"),
                               "script_sha256":sha(__file__),"spec_sha256":sha(ROOT/"eval/ragas/judge-validation-spec.json")})
    print(json.dumps({"prepared":str(folder),"tasks":len(tasks),"metrics":dict(Counter(t["metric"] for t in tasks))}),flush=True)

async def run(folder):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import FactualCorrectness,AnswerRelevancy,ContextPrecisionWithReference,ContextRecall,Faithfulness
    plan=read(folder/"plan.json"); cfg=plan["original_config"]; manifest=read(folder/"manifest.json")
    assert sha(folder/"plan.json")==manifest["plan_sha256"]
    assert sha(folder/"tasks.json")==manifest["tasks_sha256"]
    assert sha(__file__)==manifest["script_sha256"]
    assert protected()==plan["protected_sha256_before"]
    assert {p:metadata.version(p) for p in cfg["versions"]}==cfg["versions"]
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=10) as f: tags=json.load(f)
    assert {m["name"]:m["digest"] for m in tags["models"]}==cfg["ollama_model_digests"]
    client=AsyncOpenAI(api_key="ollama",base_url=cfg["base_url"],timeout=60,max_retries=0)
    llm=llm_factory(cfg["judge_model"],provider="openai",client=client,temperature=cfg["temperature"],
         top_p=cfg["top_p"],seed=cfg["seed"],max_tokens=cfg["max_tokens"],
         reasoning_effort=cfg["reasoning_effort"],system_prompt=cfg["system_prompt"])
    scorers={f"factual_{mode}":FactualCorrectness(llm=llm,mode=mode,atomicity=cfg["atomicity"],coverage=cfg["coverage"])
             for mode in ("precision","recall","f1")}
    emb=embedding_factory("openai",model=cfg["embedding_model"],client=client,interface="modern")
    scorers.update(answer_relevancy=AnswerRelevancy(llm=llm,embeddings=emb,strictness=cfg["strictness"]),
        context_precision=ContextPrecisionWithReference(llm=llm),context_recall=ContextRecall(llm=llm),
        faithfulness=Faithfulness(llm=llm))
    steps=[]; original_agenerate=llm.agenerate
    async def traced(prompt,response_model):
        step={"step":len(steps)+1,"response_model":response_model.__name__,"prompt":prompt,
              "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
        steps.append(step); started=time.perf_counter()
        try:
            result=await original_agenerate(prompt,response_model)
            step.update(status="ok",result=result.model_dump(mode="json"))
            return result
        except BaseException as e:
            step.update(status="error",error=f"{type(e).__name__}: {e}")
            raise
        finally: step["seconds"]=time.perf_counter()-started
    llm.agenerate=traced
    tasks=read(folder/"tasks.json")
    output=folder/"scores.jsonl"
    prior=rows(output) if output.exists() else []
    # Do not replace failed calls or cherry-pick retries; resume only unseen task IDs.
    done={r["task_id"] for r in prior}
    try:
        with output.open("a",encoding="utf-8") as f:
            for i,task in enumerate(tasks,1):
                if task["task_id"] in done: continue
                steps.clear(); started=time.perf_counter()
                record={k:task[k] for k in ("task_id","qid","variant","metric","repeat","expected_interval","assertion")}
                record["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
                try:
                    result=await asyncio.wait_for(scorers[task["metric"]].ascore(**task["inputs"]),timeout=150)
                    score=float(result.value)
                    if not math.isfinite(score): raise ValueError("Non-finite score")
                    bounds=task["expected_interval"]
                    record.update(status="ok",score=score,within_expected=None if bounds is None else bounds[0]-1e-6<=score<=bounds[1]+1e-6)
                except Exception as e:
                    record.update(status="error",score=None,within_expected=False,error=f"{type(e).__name__}: {e}")
                record.update(steps=list(steps),seconds=time.perf_counter()-started,reused_from=None)
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n"); f.flush()
                print(f"{i}/{len(tasks)} {record['task_id']} {record['status']} {record['score']} expected={record['within_expected']}",flush=True)
    finally:
        await client.close()
    analyze(folder)

def analyze(folder):
    plan=read(folder/"plan.json"); tasks=read(folder/"tasks.json"); rs=rows(folder/"scores.jsonl")
    assert len({r["task_id"] for r in rs})==len(rs)
    assert set(r["task_id"] for r in rs)<=set(t["task_id"] for t in tasks)
    groups=defaultdict(list)
    for r in rs: groups[(r["qid"],r["variant"],r["metric"])].append(r)
    repeats=[]; pairs=[]
    for key,items in groups.items():
        values=[r["score"] for r in items if r["status"]=="ok"]
        if len(items)>=3:
            repeats.append({"qid":key[0],"variant":key[1],"metric":key[2],
                "attempts":len(items),"valid":len(values),"scores":[r["score"] for r in items],
                "range":max(values)-min(values) if values else None,
                "exact_score_agreement":len(set(values))==1 and len(values)==len(items),
                "interval_judgment_agreement":len(set(r["within_expected"] for r in items))==1,
                "all_successful":len(values)==len(items)})
    for qid in ("d004","d016"):
        for r in range(3):
            lookup={(x["variant"],x["repeat"]):x for x in rs if x["qid"]==qid and x["metric"]=="answer_relevancy"}
            a,b=lookup.get(("correct",r)),lookup.get(("unrelated",r))
            if a and b:
                delta=a["score"]-b["score"] if a["status"]==b["status"]=="ok" else None
                pairs.append({"qid":qid,"repeat":r,"correct":a["score"],"unrelated":b["score"],
                    "delta":delta,"pass":delta is not None and delta>=0.10})
    bymetric={}
    for metric in sorted({t["metric"] for t in tasks}):
        items=[r for r in rs if r["metric"]==metric]
        bounded=[r for r in items if r["expected_interval"] is not None]
        bymetric[metric]={"attempts":len(items),"ok":sum(r["status"]=="ok" for r in items),
            "errors":sum(r["status"]=="error" for r in items),"bounded_checks":len(bounded),
            "within_expected":sum(r["within_expected"] is True for r in bounded)}
    fact_gates=[r for r in rs if r["metric"].startswith("factual_") and r["variant"] in ("exact","equivalent","wrong")]
    ctx_gates=[r for r in rs if r["assertion"]=="controlled_context_perturbation" and r["variant"]!="partial"]
    violations=[r for r in rs if r["status"]!="ok" or r["within_expected"] is False]
    complete=len(rs)==len(tasks); preservation=protected()==plan["protected_sha256_before"]
    gates={"all_tasks_processed":complete,"all_calls_successful":all(r["status"]=="ok" for r in rs) and complete,
        "facts_full_and_wrong":all(r["within_expected"] is True for r in fact_gates) and complete,
        "context_endpoint_and_ranking":all(r["within_expected"] is True for r in ctx_gates) and complete,
        "repeat_ranges_at_most_0_10":all(r["all_successful"] and r["range"]<=0.100001 for r in repeats) and complete,
        "answer_relevancy_correct_over_unrelated":len(pairs)==6 and all(p["pass"] for p in pairs),
        "protected_files_unchanged":preservation}
    summary={"processed":len(rs),"planned":len(tasks),"statuses":dict(Counter(r["status"] for r in rs)),
        "metrics":bymetric,"gates":gates,"validation_passed":all(gates.values()),"repeat_groups":repeats,
        "answer_relevancy_pairs":pairs,"violations":[{k:r[k] for k in ("task_id","score","expected_interval","status")} for r in violations],
        "limits":plan["limits"],"independent_human_expert_validation":False,"original_results_replaced":False}
    dump(folder/"summary.json",summary)
    dump(folder/"disagreements.json",[{"task":next(t for t in tasks if t["task_id"]==r["task_id"]),"result":r} for r in violations])
    print(json.dumps({"complete":complete,"statuses":summary["statuses"],"gates":gates,"summary":str(folder/"summary.json")}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","run","analyze"])
    parser.add_argument("--out",default="eval/ragas/judge-validation-001")
    args=parser.parse_args(); folder=ROOT/args.out
    if args.command=="prepare": prepare(folder)
    elif args.command=="run": asyncio.run(run(folder))
    else: analyze(folder)
