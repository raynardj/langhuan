from langhuan import NERTask
import pandas as pd
from sklearn.datasets import fetch_20newsgroups


news20 = fetch_20newsgroups()
df = pd.DataFrame(dict(text=news20['data']))
app = NERTask.from_df(
    df,
    text_col="text",
    options=["school", "company"],
    cross_verify_num=1,
)


if __name__ == "__main__":
    app.run("0.0.0.0", port="5000")
