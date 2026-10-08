from datetime import datetime
import os
import shutil


def save_code(save_dir):
    """
    Snapshot the source files (.py/.sh/.conf/.txt) into `<save_dir>/@<timestamp>/code-backup`
    for reproducibility.

    :return: the timestamp string `@YYYYmmdd-HHMMSS` identifying this run
    """
    record_time = datetime.now().strftime('@%Y%m%d-%H%M%S')
    dst_dir = os.path.join(save_dir, record_time, "code-backup")
    os.makedirs(dst_dir, exist_ok=True)

    exclude_dirs = {"assets", "log", "data", ".git", ".idea", "output", "__pycache__", "submodules", "tmp_code"}
    src_dir = os.path.abspath(".")
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        target_dir = os.path.join(dst_dir, os.path.relpath(root, src_dir))
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.split(".")[-1] in ['py', 'sh', 'conf', "txt"]:
                shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))
    return record_time
