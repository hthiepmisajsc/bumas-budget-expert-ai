import os

# Gunicorn config variables
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "debug")
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", 120))
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 5))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gevent")
workers = int(os.getenv("GUNICORN_WORKERS", 4))
threads = int(os.getenv("GUNICORN_THREADS", 10))
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
