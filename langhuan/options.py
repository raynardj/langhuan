import pandas as pd
from itertools import chain


class Options(object):
    def __init__(
            self,
            df: pd.DataFrame, options,):
        if options is None:
            options = []
        # when options => a list of options
        if type(options) in [list, set]:
            self.option_vals = [list(options), ] * len(df)
            self.known_options = list(set(options))

        # when options => a name of df column
        elif type(options) == str:
            assert options in df,\
                f"when options set to string, it has to be one of {df.columns}"

            self.option_vals = df[options]
            self.known_options = self.calc_known_options(self.option_vals)
        else:
            raise TypeError(
                f"""
        options type has to be one of 'list, str',
        yours: {type(options)}
        """)

        self.df_idx = df.index

        self.df = pd.DataFrame(
            dict(options=self.option_vals, idx=self.df_idx))

        self.option_col = self.df["options"]
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

    def new_for_option_col(self, option):
        def new_for_option_col_(x):
            if option in x:
                return x
            else:
                x.append(option)
                return x
        return new_for_option_col_

    def delete_for_option_col(self, option):
        def delete_for_option_col_(x):
            if option in x:
                x.remove(option)
                return x
            else:
                return x
        return delete_for_option_col_

    def add_option(self, option: str):
        if option not in self.known_options:
            self.known_options.append(option)

        self.option_col = self.option_col.apply(
            self.new_for_option_col(option)
        )

        return self.known_options

    def delete_option(self, option: str):
        if option in self.known_options:
            self.known_options.remove(option)
        self.option_col = self.option_col.apply(
            self.delete_for_option_col(option)
        )

        return self.known_options
