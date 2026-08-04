import json
p='/work/artifacts/05/workloads/SOL-ExecBench/L2/015_audio_sinusoidal_position_embedding_with_conv_projection/workload.jsonl'
import glob
g=glob.glob('/work/artifacts/05/workloads/**/*015_audio*/workload.jsonl',recursive=True)
print(g)
for line in open(g[0]):
    d=json.loads(line); print(json.dumps(d)[:1200]); break
