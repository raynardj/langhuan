# LangHuAn
> **Lang**uage **Hu**man **An**notations, a frontend for tagging AI project labels

> From Chinese word **琅嬛[langhuan]** (Legendary realm where god curates books)

## Minimun configuration walk through

```python
from langhuan import NERTask
app = NERTask.from_df(df, text_col="text", options=["institution", "company", "name"])
app.run("0.0.0.0")
```
