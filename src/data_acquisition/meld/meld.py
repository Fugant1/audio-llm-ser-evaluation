("""Helper to run the MELD dataset preparation script.

This module exposes `run_meld_script()` which runs the `meld.sh` script
located in the same directory as this file. It invokes the script via
`bash` and returns the subprocess CompletedProcess for inspection.
""")

from pathlib import Path
import subprocess
import sys
from typing import Optional


def run_meld_script(script_path: Optional[Path] = None,
					cwd: Optional[Path] = None,
					capture_output: bool = True) -> subprocess.CompletedProcess:
	"""Run the `meld.sh` script.

	Args:
		script_path: Optional path to the script. If None, uses `meld.sh`
			located next to this file.
		cwd: Working directory to run the script in. Defaults to the
			script directory.
		capture_output: If True, captures stdout/stderr and returns them
			on the CompletedProcess as text.

	Returns:
		subprocess.CompletedProcess returned by `subprocess.run`.

	Raises:
		FileNotFoundError: if the script file does not exist.
		subprocess.CalledProcessError: if the script exits with non-zero.
	"""
	if script_path is None:
		script_path = Path(__file__).parent / "meld.sh"
	script_path = Path(script_path)

	if not script_path.exists():
		raise FileNotFoundError(f"meld script not found at: {script_path}")

	cmd = ["bash", str(script_path)]
	proc = subprocess.run(
		cmd,
		check=False,
		cwd=cwd or script_path.parent,
		capture_output=capture_output,
		text=True,
	)
	return proc


if __name__ == "__main__":
	try:
		result = run_meld_script()
		if result.stdout:
			print(result.stdout)
		if result.stderr:
			print(result.stderr, file=sys.stderr)
	except Exception as exc:
		print("Error running meld script:", exc, file=sys.stderr)
		raise

