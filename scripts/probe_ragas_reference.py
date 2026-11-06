"""Bounded reference-subject diagnostic; never replaces original scores."""
import asyncio,hashlib,json,math,os,pathlib,sys,time
os.environ.setdefault("RAGAS_DO_NOT_TRACK","true")
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_ragas_judge import read,rows,sha,dump,protected
async def main():
 from openai import AsyncOpenAI
 from ragas.llms import llm_factory
 from ragas.metrics.collections import FactualCorrectness,ContextRecall
 folder=ROOT/"eval/ragas/judge-validation-001/reference-probe"
 plan=read(folder/"plan.json"); cfg=read(ROOT/"eval/ragas/judge-validation-001/plan.json")["original_config"]
 assert sha(folder/"inputs.json")==plan["inputs_sha256"]
 assert sha(__file__)==plan["script_sha256"]
 assert protected()==plan["protected_sha256_before"]
 client=AsyncOpenAI(api_key="ollama",base_url=cfg["base_url"],timeout=60,max_retries=0)
 llm=llm_factory(cfg["judge_model"],provider="openai",client=client,
   **{k:cfg[k] for k in ("temperature","top_p","seed","max_tokens","reasoning_effort","system_prompt")})
 scorers={"factual_f1":FactualCorrectness(llm=llm,mode="f1",atomicity=cfg["atomicity"],coverage=cfg["coverage"]),
          "context_recall":ContextRecall(llm=llm)}
 original=llm.agenerate; steps=[]
 async def traced(prompt,response_model):
  step={"schema":response_model.__name__,"prompt":prompt}; steps.append(step)
  try:
   result=await original(prompt,response_model); step.update(status="ok",result=result.model_dump(mode="json")); return result
  except BaseException as e:
   step.update(status="error",error=f"{type(e).__name__}: {e}"); raise
 llm.agenerate=traced
 tasks=read(folder/"inputs.json"); resultfile=folder/"scores.jsonl"
 prior=rows(resultfile) if resultfile.exists() else []; done={r["task_id"] for r in prior}
 try:
  with resultfile.open("a",encoding="utf-8") as f:
   for i,t in enumerate(tasks,1):
    if t["task_id"] in done: continue
    steps.clear(); r={k:v for k,v in t.items() if k!="inputs"}; started=time.perf_counter()
    try:
     value=float((await asyncio.wait_for(scorers[t["metric"]].ascore(**t["inputs"]),timeout=150)).value)
     if not math.isfinite(value): raise ValueError("non-finite score")
     lo,hi=t["expected_interval"]
     r.update(status="ok",score=value,within_expected=lo-1e-6<=value<=hi+1e-6)
    except Exception as e:
     r.update(status="error",score=None,within_expected=False,error=f"{type(e).__name__}: {e}")
    r.update(steps=list(steps),seconds=time.perf_counter()-started)
    f.write(json.dumps(r,ensure_ascii=False)+"\n"); f.flush()
    print(f"{i}/{len(tasks)} {r['task_id']} {r['status']} {r['score']} expected={r['within_expected']}",flush=True)
 finally: await client.close()
 rs=rows(resultfile)
 dump(folder/"summary.json",{"kind":"adaptive_development_reference_diagnostic_not_independent_validation",
  "planned":len(tasks),"processed":len(rs),"ok":sum(r["status"]=="ok" for r in rs),
  "within_expected":sum(r["within_expected"] for r in rs),
  "all_checks_pass":len(rs)==len(tasks) and all(r["within_expected"] for r in rs),
  "protected_files_unchanged":protected()==plan["protected_sha256_before"],
  "failures":[{k:r.get(k) for k in ("task_id","score","expected_interval","error")} for r in rs if not r["within_expected"]]})
 print(json.dumps(read(folder/"summary.json")),flush=True)
if __name__=="__main__": asyncio.run(main())
