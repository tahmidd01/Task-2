"""Execute the analysis notebook with the same interpreter as the pipeline."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_notebook(root):
    path = root / "notebooks" / "Unemployment_Analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runtime = root / "tmp" / "jupyter"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
    connection = runtime / f"kernel-{os.getpid()}.json"
    kernels = Path(sys.prefix) / "share" / "jupyter" / "kernels"
    specs = KernelSpecManager(kernel_dirs=[str(kernels)])
    manager = KernelManager(kernel_name="python3", connection_file=str(connection), kernel_spec_manager=specs)
    client = NotebookClient(notebook, km=manager, timeout=180)
    client.execute(cwd=str(root), cleanup_kc=True)
    nbformat.validate(notebook)
    temporary = Path(path).with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(f"Executed notebook saved to {path}", flush=True)

