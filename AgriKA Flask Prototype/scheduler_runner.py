from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import logging
import os
from Controller import sentinel_get  

logging.basicConfig(filename='sentinel_log.txt', level=logging.INFO)

scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', id='sentinel_get', minutes=10)
def scheduled_task():
    logging.info(f"{datetime.now()} - Running sentinel_get...")
    sentinel_get()

if __name__ == "__main__":
    scheduler.start()