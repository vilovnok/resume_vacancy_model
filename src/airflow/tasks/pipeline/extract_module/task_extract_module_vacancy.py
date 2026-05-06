import subprocess
import sys

from config import RUN_ID


def extract_vacancies():
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tasks.pipeline.extract_module.extract_core",
        "--input-dir",
        f"/opt/airflow/data/{RUN_ID}/vacancies",
        "--output-csv",
        f"/opt/airflow/data/{RUN_ID}/vacancies_dataset.csv",
        "--doc-type",
        "vacancy",
        "--meta-path",
        f"/opt/airflow/data/{RUN_ID}/vacancies_meta.csv",
        "--max-tokens",
        "4096",
        "--num-processes",
        "1",
    ]
    subprocess.run(cmd, check=True)