from abc import ABC
from typing import List


def check_column_set(column: str) -> None:
    if column is None:
        raise KeyError(
            "you have to set 'order_by_column' " +
            "when using this strategy, " +
            "preferably a score between 0 ~ 1"
        )


class OrderStrategies(ABC):
    """
    Some known strategies are available(as you can just use
    the name of the strategy), start and end are ordered
    according to the value of order_by_column:
    - forward_march:
        honest to the earth start to end
        the simplest ordering strategy
        The default is in ascending order
        If you want descending order, you can design
        your order by column as -1 x your order by column
    - pincer: move 1 by 1 both from start and from end.
    - trident: move 1 by 1 from start, from end and from
        the middle. It's like 2 pincer
    """

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
            if len(stream[min_length:]) > 0))

        return combined

    def forward_march(self, **kwargs) -> List[int]:
        """
        Simplist order strategy
        move 1 by 1 both from start and from end.
        """
        return list(self.df.index)

    def pincer(self, **kwargs) -> List[int]:
        order_by_column = kwargs.get("order_by_column")
        check_column_set(order_by_column)

        ordered_idx = list(
            self.df.sort_values(by=[order_by_column, ]).index)

        mid_point = len(ordered_idx)//2
        return self.mix_streams(
            ordered_idx[:mid_point], ordered_idx[mid_point:][::-1])

    def trident(
        self,
        **kwargs
    ) -> List[int]:
        order_by_column = kwargs.get("order_by_column")
        check_column_set(order_by_column)

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
