from pathlib import Path
from .utility import now_str
import os
import json
import logging
from typing import Dict
import pandas as pd
from datetime import datetime
from uuid import uuid4


def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def openj(file):
    with open(file, "r") as f:
        return json.loads(f.read())


def history_from_json(file):
    """
    read history from a single JSON file
    """
    df = pd.DataFrame(openj(file))
    df["fname"] = Path(file).name
    return df


def to_one_line(df):
    """
    Combine the duplicate tag into 1 line
    """
    if len(df) == 1:
        return df
    skipped = df["skipped"].sum()

    # filter out the skipped as options
    if (skipped > 0) and (skipped < len(df)):
        df = df[df.skipped.isna()]
    return df.sort_values(by="now", ascending=False).head(1)


def clean_history(history_df):
    history_df["index"] = history_df["index"].apply(int)
    newdf = history_df.groupby(["pandas", "user_id"])\
        .apply(to_one_line)
    newdf = newdf\
        .reset_index(drop=True)

    if "level_0" in newdf:
        newdf = newdf.drop("level_0", axis=1)
    if "level_2" in newdf:
        newdf = newdf.drop("level_2", axis=1)
    return newdf


def combine_history(HISTORY: Path) -> Path:
    """
    combine multiple history json files into 1 file
        remove the duplicate by user_id and pandas
    """
    HISTORY_FILES = list(
        i for i in HISTORY.iterdir() if i.suffix == ".json")

    # concatenate all the history files into 1 single dataframe
    history_df = pd.concat(list(map(history_from_json, HISTORY_FILES)))\
        .reset_index(drop=True)
    if "level_2" in history_df:
        history_df = history_df.drop(
            "level_2", axis=1)
    history_df = clean_history(history_df)
    new_history = HISTORY/f"history_combined_{now()}.json"
    history_df.to_json(new_history, orient="records")
    logging.info(f"save to {new_history}")
    for i in HISTORY_FILES:
        logging.info(f"rm -f {i}")
        os.system(f"rm -f {i}")
    return new_history


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
        logging.info(f"current history length: {len(self.history)}")
        logging.info(f"current save mark: {self.history_save_mark}")
        to_save = self.history[self.history_save_mark:]
        self.history_save_mark += len(to_save)

        hexstr = str(uuid4())[-6:]
        with open(self.task/f"history_{now_str()}_{hexstr}.json", "w") as f:
            f.write(json.dumps(to_save))

    def read_all_history(self):
        """
        read the saved history from self.task
        """
        history_data = []
        history_path = combine_history(self.task)
        with open(history_path, "r") as f:
            history_data = json.loads(f.read())

        self.history_save_mark = len(history_data)

        return history_data

    def __add__(self, x: Dict[str, str]):
        """
        Entrying new data
        """
        self.history.append(x)

        # batcher is used to save the history in batches
        self.batcher += 1
        if self.batcher % self.save_frequency == 0:
            logging.info(f"auto save triggered: {self.batcher}")
            self.save_new_history()
        return self.history
