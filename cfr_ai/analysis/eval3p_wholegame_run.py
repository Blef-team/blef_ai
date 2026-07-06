"""Run the PROVEN single-process whole-game eval (cfr_ai.analysis.eval3p_wholegame) for
one CFR base, so it can be invoked once for shipped (cfr_ai/p3/outputs) and once
for 2x (cfr_ai/p3_2x/outputs) and the CFR first-out rates compared. No
multiprocessing — same single-process path that produced the original result.

  python -m cfr_ai.analysis.eval3p_wholegame_run cfr_ai/p3/outputs 150
"""
import json
import sys
import time

import cfr_ai.analysis.eval3p_wholegame as wg
from conservative_bayesian_ai.agent import determine_action as cons
from nfsp_ai.production_agent import load_agent, determine_action as nfsp

load_agent(
    model_path="nfsp_ai/artifacts/nfsp_inference_24.pt",
    card_embedding_path="nfsp_ai/artifacts/card_embedding_pretrain_24.pt",
    history_embedding_path="nfsp_ai/artifacts/history_embedding_pretrain_24.pt",
    greedy=False,
)

base = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
wg.P3_OUT = base                      # the run() CFR lambda reads this module global
t0 = time.time()
res = wg.run(cons, nfsp, n_per_perm=n)
print(f"BASE {base}  n_per_perm={n}  ({time.time()-t0:.0f}s)")
print(wg._fmt(res))
print("RATES " + json.dumps(res["first_out_rate"]))
