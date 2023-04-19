from pathlib import Path
import os

SCRATCH = os.environ.get("SCRATCH", ".")
runpath = Path(SCRATCH) / "npe_conv/lz96/run"
fd = filter(os.path.isdir, [os.path.join(runpath,f) for f in os.listdir(runpath)])
print(fd)