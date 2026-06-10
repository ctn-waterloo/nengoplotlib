"""Download example dependency files for the connectome notebook.

Run once from the examples/ directory before opening the notebook:

    python examples/download_example_data.py
"""

import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent

FILES = {
    "forage.py": "https://raw.githubusercontent.com/kpc-simone/nengo_learning_examples/refs/heads/master/grid/forage.py",
    "grid.py":   "https://raw.githubusercontent.com/tcstewar/nengo_learning_examples/refs/heads/master/grid/grid.py",
}

for name, url in FILES.items():
    dest = HERE / name
    if dest.exists():
        print(f"  {name} already present, skipping.")
    else:
        print(f"  Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")

print("done.")
