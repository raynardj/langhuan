from pathlib import Path
from .utility import now_str
import os
import json
import logging
from typing import Dict


class History:
    """
    History log handler
    """

    def __init__(
        self,
        task_name: str,
        load_history: bool = False,
        save_frequency: int = 42,
    ):
        """
        task_name: str, name of the task
            if $HOME/.cache/langhuan/{task_name} exist
        """
        self.task_name = task_name
        self.load_history = load_history
        self.save_frequency = save_frequency
        self.home = Path(os.environ["HOME"])
        self.history = []
        self.history_save_mark = 0
        self.batcher = 0

        self.cache = self.home / ".cache"
        self.cache.mkdir(exist_ok=True)

        self.langhuan = self.cache / "langhuan"
        self.langhuan.mkdir(exist_ok=True)

        self.task = self.langhuan / task_name

        if self.load_history:
            if self.task.exists():
                self.history = self.read_all_history()
            else:
                logging.error(
                    f"no history under {self.task}")
                logging.error(
                    "you can try changing the task_name"
                )
                tasks = os.listdir(self.langhuan)
                logging.error(
                    f"valid task names:\n{tasks}"
                )
        self.task.mkdir(exist_ok=True)

    def __repr__(self):
        return f"CacheFolder:\t{self.task}"

    def save_new_history(self):
        to_save = self.history[self.history_save_mark:]
        self.history_save_mark += len(to_save)
        with open(self.task/f"history_{now_str()}.json", "w") as f:
            f.write(json.dumps(to_save))

    def read_all_history(self):
        """
        read the saved history from self.task
        """
        history_data = []
        files = os.listdir(self.task)

        if len(files) > 200:
            logging.info(f"Loading {len(files)} cache files")
            logging.info("Please wait patiently")

        for js in files:
            if "history_" in js:
                with open(self.task / js, "r") as f:
                    history_data += json.loads(f.read())

        return history_data

    def __add__(self, x: Dict[str, str]):
        """
        Entrying new data
        """
        self.history.append(x)
        self.batcher += 1
        if self.batcher % self.save_frequency == 0:
            self.save_new_history()
        return self.history
