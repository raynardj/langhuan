from flask import Flask, request, jsonify
from .hyperflask import HyperFlask
from flask.templating import render_template
import logging
from typing import List, Union, Callable
import json
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import chain
from pathlib import Path


def now_str(): return datetime.now().strftime("%y%m%d_%H%M%S")


def now_specific(): return datetime.now().strftime("%y-%m-%d_%H:%M:%S")


def cleanup_tags(x):
    return x.replace("<", "◀️").replace(">", "▶️")


def arg_by_key(key: str) -> Union[str, int, float]:
    """
    get a value either by GET or POST method
    with api function
    """
    if request.method == "POST":
        data = json.loads(request.data)
        rt = data[key]
    elif request.method == "GET":
        rt = request.args.get(key)
    else:
        raise RuntimeError(
            "method has to be GET or POST")
    return rt


tagged = dict()
history = list()


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


class Progress:
    """
    A project progress handler,
        allowing multiple but limited number of users
        working a the same progress, with limited tags
        per entry of raw data
    """

    def __init__(
        self,
        progress_list: List[int],
        cross_verify_num: int = 1,
        history_length: int = 20,
    ):
        self.progress_list = progress_list
        self.history_length = history_length
        self.v_num = cross_verify_num
        self.ct = 0
        self.depth = dict((i, dict()) for i in range(len(progress_list)))
        self.personal_history = dict()
        self.idx_to_index = dict((v, k) for k, v in enumerate(progress_list))

        self.by_user_wip = dict()

    def __getitem__(self, index: int) -> Union[int, str]:
        return self.progress_list[index]

    def next_id(self, user_id: str) -> Union[int, str]:
        """
        user_id, random generated hex string
        return the next id for dataframe index
        """
        if self.ct >= len(self.progress_list):
            raise StopIteration("Task done")

        if len(self.depth[self.ct]) >= self.v_num:
            self.ct += 1
            return self.next_id(user_id)

        elif len(self.depth[self.ct]) +\
            len(list(filter(
                lambda x: x == self.ct,
                self.by_user_wip.values()))) >= self.v_num:
            self.ct += 1
            return self.next_id(user_id)

        else:
            self.by_user_wip[user_id] = self.ct
            return self.ct

    def tagging(self, data):
        index = arg_by_key("index")
        user_id = data["user_id"]
        self.depth[index][user_id] = data
        self.update_personal(data)
        if user_id in self.by_user_wip:
            del self.by_user_wip[user_id]

    def update_personal(self, data):
        """
        update data to personal history
        """
        user_id = data["user_id"]
        personal_history = self.personal_history.get(user_id)
        if type(personal_history) == list:
            personal_history.append(data)
            if len(personal_history) > self.history_length:
                personal_history = personal_history[1:]
        else:
            self.personal_history[user_id] = []
            self.update_personal(data)


class LangHuanBaseTask(Flask):
    task_type = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        text_col: Union[List[str], str],
        task_name: str = None,
        options: List[str] = None,
        app_name: str = "LangHuan",
        order_strategy: Union[str, Callable] = "forward_march",
        order_by_column: str = None,
        cross_verify_num: int = 1,
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
        app.create_progress(order_strategy, order_by_column, cross_verify_num)

        return app

    def forward_march(self, **kwargs) -> List[int]:
        return list(self.df.index)

    def mix_streams(self, *streams):
        if len(streams) == 0:
            return []
        elif len(streams) == 1:
            return list(streams[0])

        min_length = min(list(map(len, streams)))
        combined = []
        for i in range(min_length):
            for stream in streams:
                combined.append(stream[i])

        combined += self.mix_streams(*list(
            stream[min_length:] for stream in streams
            if stream[min_length:] > 0))

        return combined

    def pincer(self, **kwargs) -> List[int]:
        order_by_column = kwargs.get("order_by_column")
        if order_by_column is None:
            raise KeyError(
                "you have to set 'order_by_column' " +
                "when using pincer strategy, " +
                "preferably a score between 0 ~ 1"
            )

        ordered_idx = list(
            self.df.sort_values(by=[order_by_column, ]).index)

        mid_point = len(ordered_idx)//2
        return self.mix_streams(
            ordered_idx[:mid_point], ordered_idx[mid_point:][::-1])

    def trident(self, **kwargs) -> List[int]:
        order_by_column = kwargs.get("order_by_column")
        if order_by_column is None:
            raise KeyError(
                "you have to set 'order_by_column' " +
                "when using trident strategy, " +
                "preferably a score between 0 ~ 1"
            )

        mid_score = self.df[order_by_column].min() +\
            (self.df[order_by_column].max() -
             self.df[order_by_column].min())/2

        sorted_df = self.df.sort_values(by=[order_by_column, ])

        bigger = sorted_df.query(f"{order_by_column}>={mid_score}")
        smaller = sorted_df.query(f"{order_by_column}<{mid_score}")

        bigger_mid_point = len(bigger)//2
        smaller_mid_point = len(smaller)//2

        return self.mix_streams(
            list(bigger.index)[:bigger_mid_point],
            list(bigger.index)[bigger_mid_point:][::-1],
            list(smaller.index)[:smaller_mid_point],
            list(smaller.index)[smaller_mid_point:][::-1],
        )

    def create_progress(
        self,
        order_strategy: str,
        order_by_column: str,
        cross_verify_num: int,
    ) -> List[int]:
        strategy_options = {
            "forward_march": self.forward_march,
            "pincer": self.pincer,
            "trident": self.trident,
        }

        if type(order_strategy) == str:
            assert order_strategy in strategy_options,\
                f"""order_strategy has to be one of
            {list(strategy_options.keys())}"""
            self.progress_list = strategy_options[order_strategy](
                order_by_column=order_by_column)
        else:
            self.progress_list = order_strategy()
        self.progress = Progress(
            self.progress_list, cross_verify_num=cross_verify_num)
        return self.progress_list

    def __repr__(self):
        return f"""{self.__class__.__name__},
        self.run('0.0.0.0', debug=True)"""

    @staticmethod
    def create_task_name(task_name, task_type):
        return task_name if task_name else f"task_{task_type}_{now_str()}"

    def log_skip(self, user_id):
        if user_id in self.progress.by_user_wip:
            history.append(
                dict(
                    index=self.progress.by_user_wip[user_id],
                    user_id=user_id))
            del self.progress.by_user_wip[user_id]

    def register(
        self,
        df: pd.DataFrame,
        text_col: Union[List[str], str],
        options: List[str],
    ) -> None:
        return NotImplementedError(
            "LangHuanBaseTask.register() should be over writen")

    def register_functions(self):
        """
        register the custom decorated route functions
        """
        @self.route("/data", methods=["POST", "GET"])
        def raw_data():
            index = arg_by_key("index")
            user_id = arg_by_key("user_id")
            # move on the progress
            if index == -1:
                try:
                    self.log_skip(user_id)
                    index = self.progress.next_id(user_id)
                except StopIteration:
                    return jsonify(dict(done=True))

            # transform range index to dataframe index
            idx = self.progress[index]
            text = cleanup_tags(self.df.loc[idx, self.text_col])
            options = self.options[idx]

            rt = dict(idx=idx, index=index, text=text, options=list(options))

            if user_id in self.progress.depth[index].keys():
                rt.update({"record": self.progress.depth[index][user_id]})

            return jsonify(rt)

        @self.route("/tagging", methods=["POST"])
        def tagging():
            remote_addr = request.remote_addr
            data = json.loads(request.data)
            data.update({"remote_addr": remote_addr, "now": now_specific()})
            self.progress.tagging(data)
            history.append(data)
            return jsonify({"index": data["index"]}), 200

        @self.route("/latest", methods=["POST", "GET"])
        def lastest():
            if request.method == "POST":
                data = json.loads(request.data)
                n = data["n"] if "n" in data else 20
            else:
                n = 20
            return jsonify(history[-n:][::-1])

        @self.route("/monitor", methods=["POST", "GET"])
        def monitor():
            stats = dict(
                total=len(self.progress.progress_list),
                by_user=self.progress.by_user_wip,
                current_id=self.progress.ct,
            )
            return jsonify(stats)

        @self.route("/tagged", methods=["POST", "GET"])
        def tagged_data():
            return jsonify(
                dict((k, v) for k, v in self.progress.depth.items()
                     if len(v) > 0))


class NERTask(LangHuanBaseTask):
    task_type = "NER"

    def register(
        self,
        df: pd.DataFrame,
        text_col: Union[List[str], str],
        options: List[str],
    ) -> None:
        self.df = df
        self.text_col = text_col
        self.options = options

        @self.route("/")
        def idx_page():
            # with specific index
            index_qs = arg_by_key("index")
            if index_qs is not None:
                index = index_qs[0]
            else:
                index = -1
            return render_template(
                "ner.html",
                index=index,
            )

        @self.route("/result")
        def download_result():
            """
            return the result as a big json string
            """
            return jsonify(self.progress.depth)

        @self.route("/personal_history")
        def personal_history():
            user_id = arg_by_key("user_id")
            result = []
            personal_history = self.progress.personal_history[user_id]

            if personal_history is None:
                return jsonify([])

            for history in personal_history:
                result.append(
                    {"index": history["index"],
                     "time": history["now"],
                     "tags": len(history["tags"])})

            return jsonify(result)

        self.register_functions()
