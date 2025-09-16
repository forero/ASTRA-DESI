from astropy.table import Table, vstack
import glob

base = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering"
tracers = ["BGS_BRIGHT","ELG_LOPnotqso","LRG","QSO"]
ngc1 = dict(RA_min=110, RA_max=260, DEC_min=-10, DEC_max=33, zmin=0.4, zmax=0.9)
ngc2 = dict(RA_min=180, RA_max=260, DEC_min=30, DEC_max=40, zmin=0.4, zmax=0.9)

for tr in tracers:
    a = Table.read(f"{base}/{tr}_S_clustering.dat.fits")
    b = Table.read(f"{base}/{tr}_N_clustering.dat.fits")
    t = vstack([a,b])
     
    m = ((t["RA"]>ngc1["RA_min"]) & (t["RA"]<ngc1["RA_max"]) &
         (t["DEC"]>ngc1["DEC_min"]) & (t["DEC"]<ngc1["DEC_max"]))
    print(tr, "->", int(m.sum()), "items in that RA/DEC/Z for zone ngc1")
    
    m = ((t["RA"]>ngc2["RA_min"]) & (t["RA"]<ngc2["RA_max"]) &
         (t["DEC"]>ngc2["DEC_min"]) & (t["DEC"]<ngc2["DEC_max"]))
    print(tr, "->", int(m.sum()), "items in that RA/DEC/Z for zone ngc2")
    print('----')