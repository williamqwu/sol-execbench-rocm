import math

CU = 256


def cost(M, bm, ntile, per):
    """waves needed * work-per-tile-row; lower is better."""
    tiles = math.ceil(M / bm) * ntile
    return math.ceil(tiles / CU) * per


# gemm1: N tiles = I/128, candidates bm 64 (nw8 ns3) and 128 (nw8 ns2)
def cfg1(M, I):
    nt = I // 128
    c64 = cost(M, 64, nt, 64)
    c128 = cost(M, 128, nt, 128)
    return 64 if c64 < c128 else 128


# gemm2: N tiles = H/128, bm 64 (nw8) vs 128 (nw4)
def cfg2(M, H):
    nt = H // 128
    return 64 if math.ceil(M / 64) * nt <= 2 * CU else 128


G1 = {384: 64, 640: 64, 896: 64, 1024: 64, 1152: 128, 1536: 128, 1792: 128,
      1920: 128, 2048: 128, 2176: 64, 2432: 64, 2816: 64, 3072: 64, 3584: 128,
      3712: 128, 4096: 128}
G2 = {384: 64, 640: 64, 896: 64, 1024: 64, 1152: 64, 1536: 128, 1792: 128,
      1920: 128, 2048: 128, 2176: 128, 2432: 128, 2816: 128, 3072: 128,
      3584: 128, 3712: 128, 4096: 128}
ok = True
for M in G1:
    a, b = cfg1(M, 2048), cfg2(M, 3584)
    m1, m2 = ("ok" if a == G1[M] else "MISS"), ("ok" if b == G2[M] else "MISS")
    ok &= (a == G1[M]) and (b == G2[M])
    print(f"M={M:5d} g1 model={a:4d} meas={G1[M]:4d} {m1:4s} | "
          f"g2 model={b:4d} meas={G2[M]:4d} {m2}")
print("ALL MATCH" if ok else "MISMATCH")
