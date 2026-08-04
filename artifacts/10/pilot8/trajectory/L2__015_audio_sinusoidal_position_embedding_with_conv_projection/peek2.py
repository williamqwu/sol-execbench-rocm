import glob,os,json
p='/work/artifacts/05/workloads'
for f in sorted(glob.glob(p+'/**',recursive=True))[:40]: print(f)
