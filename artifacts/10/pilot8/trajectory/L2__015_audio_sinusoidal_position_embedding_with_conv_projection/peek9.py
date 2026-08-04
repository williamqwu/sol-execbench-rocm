s=open('/work/src/sol_execbench/core/bench/io.py').read()
i=s.find('def gen_inputs')
print(s[i-3000:i+4000])
