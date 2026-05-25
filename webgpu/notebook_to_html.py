"""Convert a Jupyter notebook to standalone HTML with WebGPU scenes.

Usage:
    python -m webgpu.notebook_to_html notebook.ipynb [-o output.html]
"""

import argparse
import glob
import os
import subprocess
import sys


def _find_lavapipe_icd():
    """Find lavapipe ICD json path."""
    candidates = [
        "/usr/share/vulkan/icd.d/lvp_icd.json",
        "/usr/share/vulkan/icd.d/lvp_icd.x86_64-linux-gnu.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _has_real_gpu():
    """Check if a real GPU render node is available."""
    return bool(glob.glob("/dev/dri/renderD*"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert notebook to standalone HTML with WebGPU 3D scenes."
    )
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument("-o", "--output", help="Output HTML path (default: same name .html)")
    parser.add_argument("--timeout", type=int, default=120, help="Cell execution timeout in seconds")
    parser.add_argument("--kernel", default=None, help="Jupyter kernel name")
    parser.add_argument("--lazy-load", action="store_true", default=False,
                        help="Embed screenshot previews that load the WebGPU scene on click")
    args = parser.parse_args()

    if not os.path.exists(args.notebook):
        print(f"Error: {args.notebook} not found", file=sys.stderr)
        sys.exit(1)

    # Build environment
    env = os.environ.copy()
    env["WEBGPU_EXPORTING"] = "1"
    if args.lazy_load:
        env["WEBGPU_LAZY_LOAD"] = "1"

    if not _has_real_gpu():
        icd = _find_lavapipe_icd()
        if icd:
            env["VK_ICD_FILENAMES"] = icd
            env["MESA_VK_DEVICE_SELECT"] = "lvp"
        else:
            print(
                "Warning: No GPU found and lavapipe ICD not installed.\n"
                "Install with: sudo apt-get install mesa-vulkan-drivers",
                file=sys.stderr,
            )

    # Build nbconvert command
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--execute", "--to", "html",
        f"--ExecutePreprocessor.timeout={args.timeout}",
    ]
    if args.kernel:
        cmd.append(f"--ExecutePreprocessor.kernel_name={args.kernel}")
    if args.output:
        cmd.extend(["--output", args.output])
    cmd.append(args.notebook)

    print(f"Converting {args.notebook}...")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
