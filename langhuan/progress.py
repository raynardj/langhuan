from typing import List, Callable, Union


class Dispatcher:
    def __init__(self, n, v):
        self.n = n
        self.v = v
        self.sent = -1
        self.started = dict()
        self.by_user = dict()
        self.busy_by_user = dict()

    def new_user_todo(self, user_id):
        return {user_id: list(range(self.n))[max(0, self.sent):]}

    def user_progress(self, user_id):
        try:
            user_progress = self.by_user[user_id]
        except KeyError:
            self.by_user.update(self.new_user_todo(user_id))
            user_progress = self.by_user[user_id]
        return user_progress

    def __repr__(self):
        return str(self.by_user)+"\n"+str(self.sent)

    def __getitem__(self, user_id):
        """
        get index by user_id
        """
        if user_id in self.busy_by_user:
            # read cache
            return self.busy_by_user[user_id]

        user = self.user_progress(user_id)
        self.user_clear_progress(user_id)
        try:
            index = user[0]
            self.after_get_update(user_id, index)

        except IndexError:
            return -1
        return index

    def after_get_update(self, user_id, index):
        # save cache
        self.busy_by_user[user_id] = index
        user = self.user_progress(user_id)
        if index in user:
            user.remove(index)
        if self.started.get(index) is None:
            self.started[index] = [user_id, ]
        else:
            self.started[index].append(user_id)
        if len(self.started[index]) >= self.v:
            self.tick_sent(index)

    def finish_update(
        self,
        user_id: str,
        index: int,
        callbacks: List[Callable] = []
    ):
        """
        callbacks allow for furthur manuvers
        each callback function can process:
        callback(user_id, index)
        """
        # delete cache
        if self.busy_by_user.get(user_id) == index:
            del self.busy_by_user[user_id]
        for callback in callbacks:
            callback(user_id, index)

    def user_clear_progress(self, user_id):
        user_progress = self.user_progress(user_id)
        for i in user_progress:
            if i <= self.sent:
                user_progress.remove(i)
            else:
                break

    def tick_sent(self, index):
        self.sent = index
        del self.started[index]


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
        return self.dispatcher[user_id]

    def recover_history(self, data):
        """
        Recover single entry of history data
            to the dispatcher
        """
        self.tagging(data)
        user_id = data["user_id"]
        pandas = data["pandas"]
        index = self.idx_to_index[pandas]
        self.dispatcher.after_get_update(user_id=user_id, index=index)
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