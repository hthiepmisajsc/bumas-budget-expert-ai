# Gunicorn config variables
loglevel = "debug"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 30
keepalive = 5
worker_class = "gthread"
workers = 4
threads = 8
bind = "0.0.0.0:5000"