"""The packaged Windows desktop app: a pywebview shell around the same API + pipeline
every other entrypoint (`python -m mirage`, `python -m mirage.api`) already runs.

Not imported by any non-desktop code path -- mirage/app.py, mirage/api/app.py etc. know
nothing about this package. See launcher.py for the entrypoint PyInstaller freezes.
"""
