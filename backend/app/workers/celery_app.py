"""Celery app. On Windows run the worker with the solo pool:

    celery -A app.workers.celery_app worker --pool=solo --concurrency=1 -l info
"""

from celery import Celery

from app.core.config import get_settings

_s = get_settings()

celery_app = Celery("video_rag", broker=_s.redis_url, backend=_s.redis_url,
                    include=["app.workers.tasks"])
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # bounded queue: one heavy video job at a time
    task_time_limit=3600,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)
