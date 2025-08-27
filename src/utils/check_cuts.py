from astropy.table import Table
import glob

base = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering"
tracers = ["BGS_BRIGHT","ELG_LOPnotqso","LRG","QSO"]
# cuts = dict(RA_min=0, RA_max=360, DEC_min=-90, DEC_max=90)  # si hay datos, no hay error
cuts = dict(RA_min=110, RA_max=300, DEC_min=-80, DEC_max=48, zmin=0.3, zmax=0.9) #! funciona para ngc1
cuts = dict(RA_min=180, RA_max=260, DEC_min=30, DEC_max=40, zmin=0.3, zmax=0.9) #! funciona para ngc2

for tr in tracers:
    t = Table.read(f"{base}/{tr}_N_clustering.dat.fits")
    m = ((t["RA"]>cuts["RA_min"]) & (t["RA"]<cuts["RA_max"]) &
         (t["DEC"]>cuts["DEC_min"]) & (t["DEC"]<cuts["DEC_max"]) &
         (t["Z"]>cuts["zmin"]) & (t["Z"]<cuts["zmax"]))
    print(tr, "->", int(m.sum()), "items in that RA/DEC/Z")