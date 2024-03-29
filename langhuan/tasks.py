from .hyperflask import HyperFlask
from .history import History
from .options import Options
from .utility import now_str, now_specific, cleanup_tags,\
    arg_by_key, get_root
from .order_strategies import OrderStrategies
from .progress import Progress

from flask import Flask, request, jsonify
from flask.templating import render_template
import logging
import json
import pandas as pd
from typing import List, Union, Callable
from pathlib import Path
from uuid import uuid4


class LangHuanBaseTask(Flask, OrderStrategies):
    task_type = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        text_col: str,
        task_name: str = None,
        options: List[str] = None,
        load_history: bool = False,
        preset_tag_col: str = None,
        save_frequency: int = 5,
        order_strategy: Union[str, Callable] = "forward_march",
        order_by_column: str = None,
        cross_verify_num: int = 1,
        admin_control: bool = False,
    ):
        """
        Load an flask app from pandas dataframe
        Input columns:

        - text_col: str, column name that contains raw data
        - options: List[str] = None, a list of string options
            you don't even have to decide this now, you can input
            None and configure it on /admin page later
        - load_history: bool = False, load the saved history if True
        - preset_tag_col: str = None, a column name
            that contains preset tags
        - task_name: str, name of your task, if not provided
        - order_strategy: Union[str, Callable] = "forward_march",
            a function defining how progress move one by one. As a
            the function will produce a list of pandas index as
            correct order.
            Some known strategies are available(as you can just use
            the name of the strategy), start and end are ordered
            according to the value of order_by_column:
            - forward_march:
                honest to the earth start to end
            - pincer: move 1 by 1 both from start and from end.
            - trident: move 1 by 1 from start, from end and from
                the middle.
        - order_by_column: str = None, set order_by_column, a
            dataframe column name, for order_strategy. "forward_match"
            strategy does not require this info.
        - cross_verify_num: int = 1, this number decides how many
            people should see to one entry at least
        - admin_control: bool = False, if True you have to
            set adminkey as query string or post data when visit
            admin related sites
        """
        app_name = cls.create_task_name(task_name, cls.task_type)

        app = cls(
            app_name,
            static_folder=str(get_root()/"static"),
            template_folder=str(get_root()/"templates")
        )

        app.task_history = History(
            app_name,
            load_history=load_history,
            save_frequency=save_frequency,
        )

        app.admin_control = admin_control
        if app.admin_control:
            app.admin_key = str(uuid4())

        if app.admin_control:
            logging.info(
                f"please visit admin page:/admin?adminkey={app.admin_key}")

        app.config['TEMPLATES_AUTO_RELOAD'] = True
        HyperFlask(app)

        app.register(df, text_col, preset_tag_col, Options(df, options))
        app.create_progress(
            order_strategy,
            order_by_column,
            cross_verify_num=cross_verify_num)

        if load_history:
            logging.info(f"start loading history")
            # loading the history to progress
            if len(app.task_history.history) > 0:
                for data in app.task_history.history:
                    app.progress.recover_history(data)
            logging.info(f"history loaded")
        return app

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

    def admin_access(self, f: Callable) -> Callable:
        """
        simple access control
        you can access the GET url by passing query string
            by key: adminkey
        you can access the POST url by passing data
            by key: adminkey

        adminkey can be obtained in console
            when first initiated,
            or print out app.admin_key

        this function is intended to be used as decorator
        """
        if self.admin_control:
            def wrapper():
                admin_key = arg_by_key("adminkey")
                if admin_key is None:
                    return "<h3>please provide adminkey</h3>", 403
                if admin_key == self.admin_key:
                    return f()
                else:
                    return "<h3>'adminkey' not correct</h3>", 403
            wrapper.__name__ = f.__name__
            return wrapper
        else:
            return f

    def show_history_log(self, history):
        if "skipped" in history:
            return {
                "index": history["index"],
                "time": history["now"],
                "label": "[skipped]",
                "skipped": True,
            }
        else:
            return {"index": history["index"],
                    "time": history["now"],
                    "label": history["label"]}

    def append_text_to_data(self, text_dict, data):
        text_dict[data["pandas"]] = self.df.loc[data["pandas"], self.text_col]
        return data

    def register_functions(self):
        """
        register the custom decorated route functions
        """
        @self.route("/")
        def idx_page():
            index_qs = arg_by_key("index")
            # with specific index
            if index_qs is not None:
                index = index_qs
            # no index
            # let Progress handler handle the progress
            else:
                index = -1
            return render_template(
                self.tagging_template,
                index=index,
            )

        @self.route("/data", methods=["POST", "GET"])
        def raw_data():
            index = arg_by_key("index")
            user_id = arg_by_key("user_id")
            # move on the progress
            if index == -1:
                try:
                    index = self.progress.next_id(user_id)
                except StopIteration:
                    return jsonify(dict(done=True))

            # transform range index to dataframe index
            idx = self.progress[index]
            text = cleanup_tags(self.df.loc[idx, self.text_col])
            options = self.options[idx]

            rt = dict(idx=idx, index=index, text=text, options=list(options))

            # Scenario 1, when we have tagged data in progress
            if user_id in self.progress.depth[index].keys():
                rt.update({"record": self.progress.depth[index][user_id]})

            # Scenario 2, when we have preset tag defined in dataframe
            elif self.preset_tag_col is not None:
                preset_tag = self.df.loc[idx, self.preset_tag_col]
                if preset_tag is not None:
                    rt.update({"record": preset_tag})

            return jsonify(rt)

        @self.route("/tagging", methods=["POST"])
        def tagging():
            remote_addr = request.remote_addr
            data = json.loads(request.data)
            logging.debug(f"""[REQUEST DATA: tagging]{json.dumps(
                data, indent=2)}""")
            data.update({
                "remote_addr": remote_addr,
                "now": now_specific(),
                "pandas": self.progress[data["index"]]
            })
            self.progress.tagging(data)
            self.task_history + data
            return jsonify({"index": data["index"]}), 200

        @self.route("/latest", methods=["POST", "GET"])
        def lastest():
            if request.method == "POST":
                data = json.loads(request.data)
                n = data["n"] if "n" in data else 20
            else:
                n = 20
            return jsonify(self.task_history.history[-n:][::-1])

        @self.route("/save_progress", methods=["GET", "POST"])
        def save_progress():
            self.task_history.save_new_history()
            return jsonify(
                {"so_far": self.task_history.history_save_mark}
            )

        @self.route("/tagged", methods=["POST", "GET"])
        def tagged_data():
            return jsonify(
                dict((k, v) for k, v in self.progress.depth.items()
                     if len(v) > 0))

        @self.route("/get_options", methods=["POST", "GET"])
        @self.admin_access
        def get_options():
            return jsonify(self.options.known_options)

        @self.route("/stats", methods=["POST", "GET"])
        @self.admin_access
        def get_stats():
            """
            get by user statistics
            """
            user_ids = list(self.progress.personal_history.keys())
            by_user = dict((u, dict(
                entries=[],
                skipped=[],
            )) for u in user_ids)
            for index, user_entry in self.progress.depth.items():
                for user_id, data in user_entry.items():
                    index = data["index"]
                    by_user[user_id]["entries"].append(index)
                    if "skipped" in data:
                        if data["skipped"] is True:
                            by_user[user_id]["skipped"].append(index)
            for user_id, v in by_user.items():
                v["entry_ct"] = len(set(v["entries"]))
                v["skip_ct"] = len(set(v["skipped"]))
                del v["entries"]
                del v["skipped"]
            return jsonify(by_user)

        @self.route("/add_option", methods=["POST", "GET"])
        @self.admin_access
        def add_option():
            """
            add a new tagging option
            Input:
            - option: str
            """
            option = arg_by_key("option")
            self.options.add_option(option)
            return jsonify({"option": option})

        @self.route("/delete_option", methods=["POST", "GET"])
        @self.admin_access
        def delete_option():
            """
            delete an existing tagging option
            Input:
            - option: str
            """
            option = arg_by_key("option")
            self.options.delete_option(option)
            return jsonify({"option": option})

        @self.route("/admin")
        @self.admin_access
        def admin():
            """
            The admin page
            """
            return render_template("admin.html")

        @self.route("/result")
        def download_result():
            """
            return the result as a big json string
            """
            texts = dict()
            labels = dict((k, list(
                self.append_text_to_data(texts, d) for u, d in v.items()))
                for k, v in self.progress.depth.items()
                if len(v) > 0)

            result = dict(
                texts=texts,
                labels=labels,
                text_col=self.text_col,
                options=self.options.known_options,
                admin_control=self.admin_control,
            )
            return jsonify(result)

        @self.route("/personal_history")
        def personal_history():
            user_id = arg_by_key("user_id")
            result = []
            personal_history = self.progress.personal_history.get(user_id)

            if personal_history is None:
                return jsonify([])

            for history in personal_history:
                result.append(self.show_history_log(history))
            return jsonify(result[::-1])

    def register(
        self,
        df: pd.DataFrame,
        text_col: Union[List[str], str],
        preset_tag_col: str,
        options: List[str],
    ) -> None:
        """
        Register properties to the class
        """
        self.df = df
        self.text_col = text_col
        self.preset_tag_col = preset_tag_col

        self.options = options
        self.register_functions()


class NERTask(LangHuanBaseTask):
    """
    NER Task
    """
    task_type = "NER"
    tagging_template = "ner.html"

    def show_history_log(self, history):
        if "skipped" in history:
            return {
                "index": history["index"],
                "time": history["now"],
                "tags": 0,
                "skipped": True,
            }
        else:
            return {"index": history["index"],
                    "time": history["now"],
                    "tags": len(history["tags"])}


class ClassifyTask(LangHuanBaseTask):
    """
    Classify Task
    """
    task_type = "Classify"
    tagging_template = "classify.html"
