import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import UniqueConstraint, delete, select
from sqlalchemy.orm import Session
from datetime import datetime
import logging
import requests

from config import (
    RUN_ID, QDRANT_URL,
    QDRANT_COLLECTION_RESUMES, QDRANT_COLLECTION_VACANCIES)


logger = logging.getLogger("hr_matcher")




def load_csvs():
    resumes = pd.read_csv(f"/opt/airflow/data/{RUN_ID}/resumes_dataset.csv")
    vacancies = pd.read_csv(f"/opt/airflow/data/{RUN_ID}/vacancies_dataset.csv")
    applications = pd.read_csv(f"/opt/airflow/data/{RUN_ID}/applications.csv")
    irrelevant = pd.read_csv(f"/opt/airflow/data/{RUN_ID}/irrelevant_jobs.csv")

    return resumes, vacancies, applications, irrelevant


def create_tables_if_not_exist(engine):
    metadata = MetaData()

    resumes = Table(
        "resumes", metadata,
        Column("id", String, primary_key=True),
        Column("filename", String),
        Column("title", String),
        Column("description", Text),
        Column("skills", Text),
        Column("date", DateTime),
    )

    vacancies = Table(
        "vacancies", metadata,
        Column("id", String, primary_key=True),
        Column("filename", String),
        Column("title", String),
        Column("description", Text),
        Column("skills", Text),
        Column("date", DateTime),
    )

    applications = Table(
        "applications", metadata,
        Column("resume_id", String, ForeignKey("resumes.id")),
        Column("vacancy_id", String, ForeignKey("vacancies.id")),
        UniqueConstraint("resume_id", "vacancy_id", name="uq_resume_vacancy")
    )

    metadata.create_all(engine)
    logger.info("✅ Tables ensured")


def delete_from_qdrant(collection: str, ids: list[str]):
    if not ids:
        return

    try:
        res = requests.post(
            f"{QDRANT_URL}/collections/{collection}/points/delete",
            json={"points": ids},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Qdrant connection error: {e}")
        raise

    if res.ok:
        return

    text = res.text.lower()

    if "doesn't exist" in text or "not found" in text:
        logger.warning(
            f"⚠️ Collection `{collection}` not found in Qdrant. Skip delete."
        )
        return

    logger.error(f"❌ Qdrant delete failed: {res.text}")
    raise RuntimeError(f"Qdrant delete failed: {res.text}")


def remove_irrelevant(session, irrelevant_df, resumes_table, vacancies_table, applications_table):
    to_delete = irrelevant_df[irrelevant_df["status"] == "irrelevant"]["vacancy_id"].astype(str).tolist()

    if not to_delete:
        return

    affected_resumes = session.execute(
        select(applications_table.c.resume_id)
        .where(applications_table.c.vacancy_id.in_(to_delete))
    ).scalars().all()

    affected_resumes = list(set(affected_resumes))


    try:
        delete_from_qdrant(QDRANT_COLLECTION_VACANCIES, to_delete)
    except Exception as e:
        logger.error(f"❌ Critical Qdrant error (vacancies): {e}")
        raise 


    session.execute(
        delete(applications_table).where(
            applications_table.c.vacancy_id.in_(to_delete)
        )
    )


    session.execute(
        delete(vacancies_table).where(
            vacancies_table.c.id.in_(to_delete)
        )
    )


    orphan_resumes = session.execute(
        select(resumes_table.c.id)
        .where(
            resumes_table.c.id.in_(affected_resumes),
            ~resumes_table.c.id.in_(
                select(applications_table.c.resume_id)
            )
        )
    ).scalars().all()


    if orphan_resumes:
        try:
            delete_from_qdrant(QDRANT_COLLECTION_RESUMES, orphan_resumes)
        except Exception as e:
            logger.error(f"❌ Critical Qdrant error (resumes): {e}")
            raise

        session.execute(
            delete(resumes_table).where(
                resumes_table.c.id.in_(orphan_resumes)
            )
        )


def upsert_table(session, table, rows, pk="id"):
    for row in rows:
        stmt = insert(table).values(**row)

        update_dict = {k: row[k] for k in row if k != pk}

        stmt = stmt.on_conflict_do_update(
            index_elements=[pk],
            set_=update_dict
        )

        session.execute(stmt)


def load_metadata_into_postgres():

    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    resumes_df, vacancies_df, applications_df, irrelevant_df = load_csvs()

    create_tables_if_not_exist(engine)

    metadata = MetaData()
    metadata.reflect(bind=engine)

    resumes_table = metadata.tables["resumes"]
    vacancies_table = metadata.tables["vacancies"]
    applications_table = metadata.tables["applications"]

    now = datetime.utcnow()

    resumes_rows = []
    for _, row in resumes_df.iterrows():
        resumes_rows.append({
            "id": str(row["id"]),
            "filename": row["filename"],
            "title": row.get("primary_title", ""),
            "description": row.get("description", ""),
            "skills": row.get("skills", ""),
            "date": now
        })

    vacancies_rows = []
    for _, row in vacancies_df.iterrows():
        vacancies_rows.append({
            "id": str(row["id"]),
            "filename": row["filename"],
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "skills": row.get("skills", ""),
            "date": now
        })


    applications_df["resume_id"] = applications_df["resume_id"].astype(str)
    applications_df["vacancy_id"] = applications_df["vacancy_id"].astype(str)

    applications_rows = applications_df.to_dict(orient="records")


    with Session(engine) as session:
        upsert_table(session, resumes_table, resumes_rows)
        upsert_table(session, vacancies_table, vacancies_rows)

        for row in applications_rows:
            stmt = insert(applications_table).values(**row).on_conflict_do_nothing(
                index_elements=["resume_id", "vacancy_id"]
            )
            session.execute(stmt)

        remove_irrelevant(
            session,
            irrelevant_df,
            resumes_table,
            vacancies_table,
            applications_table
        )
        session.commit()

    logger.info("✅ Aggregation completed successfully")