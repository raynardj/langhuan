from typing import List, Callable, Union
import logging


class Dispatcher:
    def __init__(self, n, v):
        self.n = n
        self.v = v
        self.cache_data = dict()
        self.new = list(range(n))
        self.processing = dict()

    def __repr__(self):
        return f"Job Dispatcher: n:{self.n},v:{self.v}"

    def __getitem__(self, user_id):
        return self.fetch_index(user_id)

    def fetch_index(
        self, user_id,
        known_index: int = None
    ):
        """
        known_index is usually none, it only as value
            when used in recover history
        """
        # read from cache
        if user_id in self.cache_data:
            return self.cache_data[user_id]

        if known_index is not None:
            if known_index in self.new:
                # move ahead, remove the new queue
                self.new.remove(known_index)
            return known_index

        # read from existing process, meaning other user has already
        # started to process this job
        else:
            for index, user_list in self.processing.items():
                if len(self.processing[index]) >= self.v:
                    continue  # read new
                if user_id in user_list:
                    continue  # read new
                else:
                    user_list.append(user_id)
                    self.cache_data[user_id] = index
                    return index

        # read_new
        if len(self.new) > 0:
            index = self.new[0]
            self.processing[index] = []
            self.new.remove(index)
            return self[user_id]
        else:
            return -1

    def finish_update(self, user_id, index):
        if user_id in self.cache_data:
            del self.cache_data[user_id]
        if index in self.processing:
            if len(self.processing[index]) >= self.v:
                del self.processing[index]


class Progress:
    """
    A project progress handler,
        allowing multiple but limited number of users
        working a the same progress, with limited tags
        per entry of raw data
    index is a generated incremental integer series
    idx is the pandas index
    """

    def __init__(
        self,
        progress_list: List[Union[int, str]],
        cross_verify_num: int = 1,
        history_length: int = 20,
    ):
        """
        progress_list: List[int], a list of pandas index (idx)
            reordered by order strategy
        """
        self.progress_list = progress_list
        self.history_length = history_length
        self.v_num = cross_verify_num
        self.dispatcher = Dispatcher(n=len(progress_list), v=cross_verify_num)
        self.depth = dict((i, dict()) for i in range(len(progress_list)))
        self.personal_history = dict()
        self.idx_to_index = dict((v, k) for k, v in enumerate(progress_list))

    def __getitem__(self, index: int) -> Union[int, str]:
        return self.progress_list[index]

    def next_id(self, user_id: str) -> Union[int, str]:
        """
        user_id, random generated hex string
        return the next id for dataframe index
        """
        return self.dispatcher.fetch_index(user_id)

    def recover_history(self, data):
        """
        Recover single entry of history data
            to the dispatcher
        """
        user_id = data["user_id"]
        pandas = data["pandas"]

        # using pandas to map to new index
        index = self.idx_to_index[pandas]
        index = self.dispatcher.fetch_index(user_id, known_index=index)
        self.tagging(data)
        self.dispatcher.finish_update(user_id=user_id, index=index)

    def tagging(self, data):
        index = data["index"]
        user_id = data["user_id"]
        # recover the pandas index
        self.depth[index][user_id] = data
        self.dispatcher.finish_update(user_id=user_id, index=index)
        self.update_personal(data)

    def update_personal(self, data):
        """
        update data to personal history
        This history is for showing history lines on
            the left side of web browswer
        """
        user_id = data["user_id"]
        personal_history = self.personal_history.get(user_id)
        if type(personal_history) == list:
            for d in personal_history:
                if data["index"] == d["index"]:
                    personal_history.remove(d)
            personal_history.append(data)
            if len(personal_history) > self.history_length:
                personal_history = personal_history[
                    len(personal_history) - self.history_length:]
        else:
            self.personal_history[user_id] = []
            self.update_personal(data)
