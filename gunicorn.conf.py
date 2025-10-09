# Gunicorn configuration file
import os
from dotenv import load_dotenv
load_dotenv()
# Set the number of workers based on environment or a default calculation
workers = os.getenv("GUNICORN_WORKERS")
if workers.isdigit():
    workers = int(workers)
else:
    import multiprocessing
    workers = (multiprocessing.cpu_count() * 2) + 1

# Worker settings
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Bind to all interfaces on the PORT environment
bind = os.getenv('PORT', '10000')
bind = f"0.0.0.0:{bind}"

# Customize worker processes (optional)
preload_app = True