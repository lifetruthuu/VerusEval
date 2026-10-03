import subprocess

from utils.file_util import load_config


config = load_config()
verus_path = config["verus"]["verus_path"]
timeout_duration = config["verus"]["timeout_duration"]


def run_code(file_path, with_main=True):
    command = [
        verus_path,
        file_path,
        "--multiple-errors",
        "100",
        "--triggers-mode",
        "silent",
    ]
    if not with_main:
        command.extend(["--crate-type", "lib"])

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_duration,
        )
        return result.stdout.decode("utf-8"), result.stderr.decode("utf-8")
    except subprocess.TimeoutExpired:
        return "", f"Process timed out after {timeout_duration} seconds"
