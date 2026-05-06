import argparse
import csv
import json
import logging
import os
import sys
import zipfile
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import fitz
from docx import Document
from openai import OpenAI
from tqdm import tqdm

from config import AITUNNEL_API_KEY, AITUNNEL_URL

logger = logging.getLogger("hr_matcher")

_client: Optional[OpenAI] = None


def _init_client() -> None:
    global _client
    _client = OpenAI(
        api_key=AITUNNEL_API_KEY,
        base_url=AITUNNEL_URL,
    )


def normalize_filename(name: str) -> str:
    return name.strip().lstrip("._")


def is_valid_doc(file_path: Path) -> bool:
    return not file_path.name.startswith("._")


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported format: {ext}")


def ensure_unzipped(zip_path: str, target_dir: str):
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"📂 Directory already exists and not empty: {target_dir}")
        return

    if not zip_path.exists():
        logger.warning(f"⚠️ Zip file not found: {zip_path}")
        return

    logger.info(f"📦 Extracting {zip_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        logger.info("✅ Extraction completed")
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")


def load_meta_mapping(meta_path: str, id_col: str):
    mapping = {}

    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = normalize_filename(row["filename"])
            mapping[filename] = row[id_col]

    return mapping


def extract_structured_data(text: str, doc_type: str, max_tokens: int) -> dict:
    if _client is None:
        raise RuntimeError("OpenAI client is not initialized in this process")

    if doc_type == "resume":
        prompt = f"""
Ты извлекаешь структурированную информацию из резюме кандидата.

Твоя задача — строго следовать формату и не добавлять лишний текст.

---

ПРАВИЛА:
- Не выдумывай информацию
- Используй только данные из текста
- Игнорируй: компании, даты, адреса, образование, ссылки
- Если данных нет — оставь пустую строку "" или пустой список []
- Роли нормализуй (без компаний и времени)

---

ИЗВЛЕЧЬ:

1. primary_title — самая актуальная/последняя должность кандидата
2. all_titles — все уникальные должности кандидата (без повторов, без компаний)
3. description — краткое связанное, плотное summary обязанностей и достижений кандидата
4. skills — ключевые навыки через запятую (без объяснений, только названия навыков)

---

ФОРМАТ ОТВЕТА (строго JSON, без markdown и пояснений):

{{
  "primary_title": "",
  "all_titles": [],
  "description": "",
  "skills": ""
}}

---

ТЕКСТ РЕЗЮМЕ:
{text[:12000]}
""".strip()

    elif doc_type == "vacancy":
        prompt = f"""
Ты извлекаешь структурированную информацию из текста вакансии.

Твоя задача — строго следовать формату JSON и не добавлять лишний текст.

---

ПРАВИЛА:
- Используй только информацию из текста
- Не выдумывай данные
- Игнорируй: название компании, даты, адреса, зарплатные офферы (если не указано явно), юридическую информацию
- Если поле отсутствует — возвращай "" (пустую строку)
- Не добавляй комментарии, пояснения или markdown

---

ИЗВЛЕЧЬ:

1. title — название позиции (например: "Data Scientist", "Backend Engineer")
2. description — краткое связанное, структурированное описание вакансии:
   - обязанностей
   - требований
   - ожиданий от кандидата
3. skills — ключевые навыки через запятую (только названия технологий и умений, без описаний)

---

ФОРМАТ ОТВЕТА (строго JSON, без markdown и без текста вокруг):

{{
  "title": "",
  "description": "",
  "skills": ""
}}

---

ТЕКСТ ВАКАНСИИ:
{text[:12000]}
""".strip()
    else:
        raise ValueError("doc_type must be 'resume' or 'vacancy'")

    response = _client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="qwen3-30b-a3b",
        max_tokens=max_tokens,
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        return json.loads(content)
    except Exception:
        logger.warning("⚠️ Failed to parse JSON")
        return {}


def _process_one_document(task: tuple) -> Optional[dict]:
    file_path, obj_id, filename, doc_type, max_tokens = task

    try:
        raw_text = extract_text(file_path)
        result = extract_structured_data(raw_text, doc_type, max_tokens)

        base_row = {
            "id": obj_id,
            "filename": filename,
            "path": file_path,
            "description": result.get("description", ""),
            "skills": result.get("skills", ""),
        }

        if doc_type == "resume":
            base_row["primary_title"] = result.get("primary_title", "")
            base_row["all_titles"] = json.dumps(
                result.get("all_titles", []),
                ensure_ascii=False,
            )
        else:
            base_row["title"] = result.get("title", "")

        return base_row

    except Exception as e:
        logger.warning(f"⚠️ Failed {file_path}: {e}")
        return None


def process_documents(
    input_dir: str,
    output_csv: str,
    doc_type: str,
    meta_path: str,
    max_tokens: int,
    num_processes: int = 2,
):
    input_path = Path(input_dir)
    output_path = Path(output_csv)

    zip_path = input_path.parent / f"{input_path.name}.zip"
    ensure_unzipped(zip_path, input_path)

    meta_mapping = load_meta_mapping(
        meta_path,
        "resume_id" if doc_type == "resume" else "vacancy_id",
    )

    files = [
        f for f in input_path.rglob("*")
        if f.suffix.lower() in [".pdf", ".docx"]
        and is_valid_doc(f)
    ]

    logger.info(
        "📄 %s: found %d raw files in %s",
        doc_type,
        len(files),
        input_dir,
    )
    logger.info(
        "🧾 %s: meta mapping size = %d",
        doc_type,
        len(meta_mapping),
    )

    tasks = []
    skipped_not_in_meta = 0

    for file_path in files:
        filename = normalize_filename(file_path.name)

        if filename not in meta_mapping:
            skipped_not_in_meta += 1
            logger.warning("⚠️ Skipping %s: not found in meta", filename)
            continue

        obj_id = meta_mapping[filename]
        tasks.append((str(file_path), str(obj_id), filename, doc_type, max_tokens))

    total_tasks = len(tasks)

    logger.info(
        "🚀 %s: tasks created = %d, skipped_not_in_meta = %d, processes = %d",
        doc_type,
        total_tasks,
        skipped_not_in_meta,
        num_processes,
    )

    if total_tasks == 0:
        raise RuntimeError(
            f"No tasks created for doc_type={doc_type}. "
            f"Check filenames and meta_path={meta_path}"
        )

    rows = []
    processed = 0
    failed = 0

    ctx = get_context("spawn")

    with ctx.Pool(processes=num_processes, initializer=_init_client) as pool:
        with tqdm(
            total=total_tasks,
            desc=f"Processing {doc_type}",
            unit="file",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for row in pool.imap_unordered(_process_one_document, tasks, chunksize=1):
                processed += 1

                if row is not None:
                    rows.append(row)
                else:
                    failed += 1

                pbar.update(1)
                pbar.set_postfix(
                    processed=processed,
                    success=len(rows),
                    failed=failed,
                )

                if processed % 10 == 0 or processed == total_tasks:
                    logger.info(
                        "📈 %s progress: processed=%d/%d, success=%d, failed=%d",
                        doc_type,
                        processed,
                        total_tasks,
                        len(rows),
                        failed,
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["id", "filename", "path", "primary_title", "all_titles", "description", "skills"]
        if doc_type == "resume"
        else ["id", "filename", "path", "title", "description", "skills"]
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "✅ %s finished: total_files=%d, tasks=%d, success=%d, failed=%d, output=%s",
        doc_type,
        len(files),
        total_tasks,
        len(rows),
        failed,
        output_path,
    )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--doc-type", required=True, choices=["resume", "vacancy"])
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-processes", type=int, default=2)

    args = parser.parse_args()

    process_documents(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        doc_type=args.doc_type,
        meta_path=args.meta_path,
        max_tokens=args.max_tokens,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()