# Ways to load NER dataset

## For huggingface tokenizer
> If you're using huggingface tokenizer, most of the preprocessing can be automated into the following way

First we are loading a tokenizer


```python
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
```

Load the downloaded data with pre-designed pipeline


```python
from langhuan.loaders import load_ner_data_pytorch_huggingface
```

This step will return a dataset


```python
data_ds = load_ner_data_pytorch_huggingface(
    "ner_result_sample.json",
    tk,
)
```

Get a data_loader, this function will save you the effort to specify ```collate_fn```


```python
data_loader = data_ds.get_data_loader(batch_size=3, num_workers=2)
```

Split 1 dataset into train/ valid


```python
train_ds, val_ds = data_ds.split_train_valid(valid_ratio=.2)
len(train_ds), len(val_ds)
```




    (7, 1)



## Test a sample of x, y


```python
batch = data_ds.one_batch(5)
x = batch['input_ids']
y = batch['targets']
```


```python
x, y
```




    (tensor([[ 101, 2013, 1024,  ...,    0,    0,    0],
             [ 101, 2013, 1024,  ...,    0,    0,    0],
             [ 101, 2013, 1024,  ..., 1007, 1012,  102],
             [ 101, 2013, 1024,  ...,    0,    0,    0],
             [ 101, 2013, 1024,  ...,    0,    0,    0]]),
     tensor([[0, 0, 0,  ..., 0, 0, 0],
             [0, 0, 0,  ..., 0, 0, 0],
             [0, 0, 0,  ..., 0, 0, 0],
             [0, 0, 0,  ..., 0, 0, 0],
             [0, 0, 0,  ..., 0, 0, 0]]))



Here we left the slicing configuration to the hands of users


```python
x.shape, y.shape
```




    (torch.Size([5, 838]), torch.Size([5, 838]))



## Convert x, y back to NER tags
This also works for predicted y

Make sure both x and y tensors are:
* torch.LongTenser
* in cpu, not cuda  


```python
data_ds.decode(x, y)
```




    [{'row_id': 1,
      'token_id': 30,
      'text': 'smithsonian astrophysical observatory',
      'label': 'school'},
     {'row_id': 2,
      'token_id': 34,
      'text': 'new mexico state university',
      'label': 'school'},
     {'row_id': 2, 'token_id': 565, 'text': 'ibm', 'label': 'company'},
     {'row_id': 2, 'token_id': 633, 'text': 'ibm', 'label': 'company'},
     {'row_id': 2, 'token_id': 655, 'text': 'quadra', 'label': 'company'},
     {'row_id': 2, 'token_id': 664, 'text': 'apple', 'label': 'company'},
     {'row_id': 2, 'token_id': 809, 'text': 'quadra', 'label': 'company'},
     {'row_id': 2, 'token_id': 821, 'text': 'digital review', 'label': 'company'},
     {'row_id': 3, 'token_id': 32, 'text': 'purdue university', 'label': 'school'},
     {'row_id': 3,
      'token_id': 35,
      'text': 'engineering computer network',
      'label': 'company'},
     {'row_id': 3,
      'token_id': 441,
      'text': 'purdue electrical engineering',
      'label': 'company'},
     {'row_id': 4,
      'token_id': 68,
      'text': 'university of washington',
      'label': 'school'},
     {'row_id': 4, 'token_id': 97, 'text': 'si', 'label': 'company'}]



## Tensorflow:
> Development pending, [check here](https://github.com/raynardj/langhuan) to help
