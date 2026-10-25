import uuid

from app.ingestion.pipeline import process_video
from app.workers.celery_app import celery_app


@celery_app.task(name="video_rag.process_video")
def process_video_task(video_id: str) -> None:
    process_video(uuid.UUID(video_id))
