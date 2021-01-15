from flask import Flask
from .hyperflask import HyperFlask
from flask.templating import render_template
import logging
from typing import List, Optional
import json
import pandas as pd
from datetime import datetime
from itertools import chain
from pathlib import Path


def now_str(): return datetime.now().strftime("%y%m%d_%H%M%S")


class Options(object):
    def __init__(
            self,
            df: pd.DataFrame, options,):
        if options is None:
            options = []
        # when options => a list of options
        if type(options) in [list, set]:
            self.option_col = [list(options), ] * len(df)
            self.known_options = list(set(options))

        # when options => a name of df column
        elif type(options) == str:
            assert options in df,\
                f"when options set to string, it has to be one of {df.columns}"

            self.option_col = df[options]
            self.known_options = self.calc_known_options(self.option_col)
        else:
            raise TypeError(
                f"""
        options type has to be one of 'list, str',
        yours: {type(options)}
        """)

        self.df_idx = df.index

        self.df = pd.DataFrame(
            dict(options=self.option_col, idx=self.df_idx))
        self.df.set_index("idx")

    def __len__(self): return len(self.df)

    def __repr__(self):
        return f"options:{self.known_options}"

    @staticmethod
    def calc_known_options(iterable):
        return list(set(list(chain(*list(iterable)))))

    def __getitem__(self, idx):
        options = dict(self.df.loc[idx])["options"]
        return options


def get_root():
    return Path(__file__).parent.absolute()


class LangHuanBaseTask(Flask):
    task_type = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        text_col: str,
        task_name: str = None,
        options=None,
        app_name: str = "LangHuan",
    ):
        app_name = cls.create_task_name(task_name, cls.task_type)
        app = cls(
            app_name,
            static_folder=str(get_root()/"static"),
            template_folder=str(get_root()/"templates")
            )

        app.config['TEMPLATES_AUTO_RELOAD'] = True
        HyperFlask(app)

        app.register(df, text_col, Options(df, options))
        return app

    def __repr__(self):
        return f"""{self.__class__.__name__},
        self.run('0.0.0.0', debug=True)"""

    @staticmethod
    def create_task_name(task_name, task_type):
        return task_name if task_name else f"task_{task_type}_{now_str}"

    def register(self):
        return NotImplementedError("LangHuanBaseTask should be inherited")


class NERTask(LangHuanBaseTask):
    task_type = "NER"

    def register(self, df, text_col, options):
        self.df = df
        self.text_col = text_col
        self.options = options

        @self.route("/")
        def idx_page():
            idx = 0
            text = self.df.loc[idx, self.text_col]
            options = self.options[idx]
            return render_template(
                "ner.html",
                idx=idx, text=text, options=list(options)
            )
