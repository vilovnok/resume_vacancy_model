import subprocess
import sys

from config import RUN_ID


def extract_resumes():
    cmd = [
        sys.executable,
        "-m",
        "tasks.pipeline.extract_module.extract_core",
        "--input-dir",
        f"/opt/airflow/data/{RUN_ID}/resumes",
        "--output-csv",
        f"/opt/airflow/data/{RUN_ID}/resumes_dataset.csv",
        "--doc-type",
        "resume",
        "--meta-path",
        f"/opt/airflow/data/{RUN_ID}/resumes_meta.csv",
        "--max-tokens",
        "4096",
        "--num-processes",
        "1",
    ]
    subprocess.run(cmd, check=True)