from lt import *
import reference
B,S=32,4096
atol=3.3900898571863226e-06; rtol=1.1920928955078125e-07
inp=gen(B,S)
r1=reference.run(**inp)
r2=reference.run(**inp)
compare(r2,r1,atol,rtol,"ref vs ref (determinism)")
print("ref time", bench(lambda: reference.run(**inp)))
