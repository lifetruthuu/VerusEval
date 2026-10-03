import json
import os
import os.path as osp
from pathlib import Path

import yaml


def load_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def save_content(content, save_path):
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(content)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_path_by_file_name(file_name, source_dir):
    bench_list = (
        "DAFNY2VERUS-COLLECTION",
        "HumanEval-Verus",
        "MBPP-verified",
        "VeriCoding",
        "VerusBench",
    )
    for bench_name in bench_list:
        if bench_name in file_name:
            return osp.join(source_dir, bench_name, file_name)
    return None


def check_rs_files(directory):
    directory = Path(directory)
    return [
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix == ".rs"
    ]


def get_file_name_by_path(file_path):
    return osp.splitext(osp.basename(file_path))[0]


def load_config(path="config.yaml"):
    env_path = os.environ.get("STARVERUS_CONFIG")
    if path == "config.yaml" and env_path:
        path = env_path
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
