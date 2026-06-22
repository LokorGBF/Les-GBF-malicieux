import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "hitran-api",
    "numpy",
    "matplotlib"
])

print("Installation terminée.")