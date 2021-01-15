from flask import Flask
from datetime import datetime
from pathlib import Path
from typing import Dict


def usual_extra_render() -> Dict[str, str]:
    now = datetime.now()
    return dict(
        now=now,
        today=now.strftime("%y-%m-%d")
    )


def root(app: Flask) -> Path:
    """
    app's root path
    A property function for Flask
    """
    return Path(app.instance_path).parent


class HyperFlask:
    def __init__(self, app: Flask):
        app_class = app.__class__
        app_class.root = property(root)
