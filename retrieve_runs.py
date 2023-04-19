from pathlib import Path
import os
from tqdm import tqdm

SCRATCH = os.environ.get("SCRATCH", ".")
runpath = Path(SCRATCH) / "npe_nsf/lz96/runs"
destpath = Path(SCRATCH) / "res_maf_lz"
#destpath.mkdir(parents = True, exist_ok = True)

fd = filter(os.path.isdir, [os.path.join(runpath,f) for f in os.listdir(runpath)])
fd = list(fd)
for d in tqdm(fd):
    files = [os.path.join(d,f) for f in os.listdir(d)]
    if len(files) == 0:
        continue
    f = files[-1]
    with open(f, "rb") as src:
        data = src.read()
    dir_file = f.split("/")[-2:]
    dp = Path(destpath) / dir_file[0]
    dp.mkdir(parents = True, exist_ok = True)
    
    with open(Path(dp) / dir_file[1], "wb") as dest:
        dest.write(data)
