import sys,json,torch
sys.path.insert(0,'/work/src'); sys.path.insert(0,'/work/scripts')
from sol_execbench.core import Definition, Workload
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
P='/work/data/SOL-ExecBench/benchmark/L2/015_audio_sinusoidal_position_embedding_with_conv_projection'
d=Definition(**json.loads(open(P+'/definition.json').read()))
wl=[Workload(**json.loads(l)) for l in open('/work/artifacts/05/workloads/L2/015_audio_sinusoidal_position_embedding_with_conv_projection/workload.jsonl') if l.strip()]
import reference as R, kernel as K
for w in wl:
    set_seed(0)
    ins=gen_inputs(d,w,'cuda')
    a=R.run(*ins); b=K.run(*ins)
    c,ex=compute_error_stats(b,a,w.tolerance)
    print(w.axes['batch_size'],w.axes['time_dim'],"exceeds",ex,"abs",c.max_absolute_error,"rel",c.max_relative_error,"tol",w.tolerance.max_atol, "refmax", a.abs().max().item())
