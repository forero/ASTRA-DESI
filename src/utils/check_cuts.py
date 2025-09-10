from astropy.table import Table
import glob

base = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering"
tracers = ["BGS_BRIGHT","ELG_LOPnotqso","LRG","QSO"]
ngc1 = dict(RA_min=110, RA_max=260, DEC_min=-10, DEC_max=34, zmin=0.4, zmax=0.9)
ngc2 = dict(RA_min=180, RA_max=260, DEC_min=30, DEC_max=40, zmin=0.4, zmax=0.9)

for tr in tracers:
    t = Table.read(f"{base}/{tr}_N_clustering.dat.fits")
    m = ((t["RA"]>ngc1["RA_min"]) & (t["RA"]<ngc1["RA_max"]) &
         (t["DEC"]>ngc1["DEC_min"]) & (t["DEC"]<ngc1["DEC_max"]) &
         (t["Z"]>ngc1["zmin"]) & (t["Z"]<ngc1["zmax"]))
    print(tr, "->", int(m.sum()), "items in that RA/DEC/Z for zone ngc1")
    
    t = Table.read(f"{base}/{tr}_N_clustering.dat.fits")
    m = ((t["RA"]>ngc2["RA_min"]) & (t["RA"]<ngc2["RA_max"]) &
         (t["DEC"]>ngc2["DEC_min"]) & (t["DEC"]<ngc2["DEC_max"]) &
         (t["Z"]>ngc2["zmin"]) & (t["Z"]<ngc2["zmax"]))
    print(tr, "->", int(m.sum()), "items in that RA/DEC/Z for zone ngc2")
    print('----')

'''
cuts = dict(RA_min=60, RA_max=230, DEC_min=-70, DEC_max=38, zmin=0.4, zmax=0.9) en ngc1
     BGS_BRIGHT -> 1745 items in that RA/DEC/Z
     ELG_LOPnotqso -> 1475 items in that RA/DEC/Z
     LRG -> 11921 items in that RA/DEC/Z
     QSO -> 358 items in that RA/DEC/Z

# cuts = dict(RA_min=140, RA_max=220, DEC_min=30, DEC_max=40, zmin=0.4, zmax=0.9)
     BGS_BRIGHT -> 1525 items in that RA/DEC/Z
     ELG_LOPnotqso -> 1197 items in that RA/DEC/Z
     LRG -> 9624 items in that RA/DEC/Z
     QSO -> 304 items in that RA/DEC/Z
'''