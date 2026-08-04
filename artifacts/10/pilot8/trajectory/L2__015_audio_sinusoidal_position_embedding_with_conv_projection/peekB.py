s=open('/work/src/sol_execbench/core/bench/io.py').read()
i=s.find('def _generate_heuristic_tensor')
print(s[i-4000:i+3000])
