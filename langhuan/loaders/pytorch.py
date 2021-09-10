from pathlib import Path
import json
from functools import partial
from typing import Dict, Union, List, Tuple, Any
import random
import logging
import torch


def build_word_map(
    word_ids: List[int]
    ) -> List[bool]:
    """
    build a is_word_map
        it will set True for the 1st token of a word
    """
    word_map = []
    last = None
    for i in word_ids:
        if i is None:
            word_map.append(False)
        elif i != last:
            word_map.append(True)
        else:
            word_map.append(False)
        last = i
    return word_map


def word_mapping(tokens):
    """
    process a tokenizer result, returning the result with
        ```tokens.is_word_map```
    """
    result = []
    for row_id in range(tokens.input_ids.shape[0]):
        word_map = build_word_map(tokens.word_ids(row_id))
        result.append(word_map)
    tokens["is_word_map"] = torch.BoolTensor(
        result).to(tokens.input_ids.device)
    return tokens


def get_ner_data_class():
    try:
        from torch.utils.data.dataset import Dataset
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install pytorch first https://pytorch.org/")

    class NERDataset(Dataset):
        """
        NER Dataset
        return text, a list of tags
        """

        def __init__(self, data):
            self.data = data
            self.labels = []
            logging.info(
                f"collecting labels:{len(data['labels'])}")
            for index, labels in data['labels'].items():
                # filter out the texts that user choose to skip
                self.labels += list(filter(
                    lambda x: x.get("skipped") is not True, labels))
            self.texts = data['texts']
            self.options = data['options']
            self.i2c = dict(enumerate(["O", ]+self.options))
            self.c2i = dict((v, k) for k, v in enumerate(["O", ]+self.options))
            self.passon_kwargs = dict()

        def __repr__(self):
            options = self.data["options"]
            return f"NERDataset, {options}\n{len(self)}rows"

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.texts[str(label["pandas"])]
            return text, label['tags']

        def split_train_valid(self, valid_ratio: float = .2):
            """
            split dataset to train and valid.
            since one text can have multiple tags
                 the valid_ratio will be applied to text
                 instead of labels

            """
            all_train_index = []
            train_texts, val_texts = dict(), dict()
            train_labels, val_labels = dict(), dict()

            # when spliting data we put all the textinto
            #   both train and valid
            for index, text in self.texts.items():
                index = str(index)
                if random.random() > valid_ratio:
                    all_train_index.append(str(index))
                train_texts.update({index: text})
                val_texts.update({index: text})

            # we only separate the train valid for labels
            for idx, label_list in self.data["labels"].items():
                index = str(label_list[0]["pandas"])
                if index in all_train_index:
                    train_labels.update({index: label_list})
                else:
                    val_labels.update({index: label_list})

            train_data, val_data = dict(), dict()

            for k, v in self.data.items():
                if k == "texts":
                    train_data.update({"texts": train_texts})
                    val_data.update({"texts": val_texts})
                elif k == "labels":
                    train_data.update({"labels": train_labels})
                    val_data.update({"labels": val_labels})
                else:
                    train_data.update({k: v})
                    val_data.update({k: v})
            # instantiate by the same class
            return self.__class__(train_data, **self.passon_kwargs),\
                self.__class__(val_data, **self.passon_kwargs)
    return NERDataset


def get_ner_hf_class():
    try:
        from torch.utils.data.dataloader import DataLoader
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install pytorch first https://pytorch.org/")
    NERDataset = get_ner_data_class()

    class NERDatasetHF(NERDataset):
        """
        NER Dataset of HuggingFace tokenized sentences
        output Tuple:
            x, y
        """

        def __init__(
            self,
            data: Dict[str, Any],
            tokenizer,
            tokenization_options: Dict[str, Any] = dict({
                'return_offsets_mapping': True,
                'padding': 'max_length',
                'return_tensors': 'pt',
                'truncation': True,
                'max_length': 512}),
            return_word_mapping: bool = True,
        ):
            """
            data: contains at least the following keys
                texts: Dict[str, Any], a dictionary of texts, could be empty
                labels: Dict[str, Any], a dictionary of labels, could be empty
                options: Dict[str, Any], a dictionary of options,
                    could be empty

            tokenizer: a huggingface tokenizer class object

            tokenization_options: tokenization options for tokenizer
                eg.{'return_offsets_mapping': True,
                    'padding': 'max_length',
                    'return_tensors': 'pt',
                    'truncation': True,
                    'max_length': 512}
            """
            super().__init__(data)
            logging.info("initializing NERDatasetHF")
            self.options = self.data["options"]

            self.tokenizer = tokenizer
            self.return_word_mapping = return_word_mapping

            # default tokenization options
            # this will be overriden by the tokenization_options
            tkkw = dict(
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )

            tkkw.update(tokenization_options)
            self.tokenization_options = tkkw
            self.tokenizing = partial(
                self.tokenizer,
                **tkkw
            )

            # keyword arguments passed on to inherited class
            self.passon_kwargs = dict(
                {
                    "tokenizer": tokenizer,
                    "tokenization_options": tokenization_options}
            )

        def __repr__(self):
            return f"NER dataset (HuggingFace Tokenizer):\n\t{len(self)} rows"

        @staticmethod
        def tag_to_offset(tag: Dict[str, Union[str, int]]):
            offset = tag['offset']
            return [offset, len(tag['text'])+offset], tag['label']

        def collate(self, batch):
            texts, labels = zip(*batch)
            tked = self.tokenizing(list(texts))
            x = tked['input_ids']
            offset_mapping = tked['offset_mapping']

            # mark label from an all zero tensor
            y = torch.zeros_like(x)

            for row in range(x.size(0)):
                tags_pos = list(map(self.tag_to_offset, labels[row]))

                for tag_pos, tag_label in tags_pos:

                    tag_pos = torch.LongTensor(tag_pos)
                    tag_mask = (tag_pos[0] <= offset_mapping[row, :, 0]) * \
                        (tag_pos[1] >= offset_mapping[row, :, 1]) * \
                        self.c2i[tag_label]

                    y[row, :len(tag_mask)] += tag_mask

            tked["targets"] = y

            # if chosen so, we will return a mask of the tokens that
            # is the first token of a word
            if self.return_word_mapping:
                tked = word_mapping(tked)
            return tked

        def get_data_loader(self, **kwargs):
            """
            input:
                - batch_size
                - shuffle
                - num_workers
                - ... other key word arguments for dataloader
                    except collate_fn, which will be default
                    to self.collate
            """
            return DataLoader(self, collate_fn=self.collate, **kwargs)

        def one_batch(
            self, batch_size: int = 2
        ) -> Tuple[torch.LongTensor]:
            """
            sample 1 batch of data
            """
            return next(iter(
                self.get_data_loader(
                    batch_size=batch_size, shuffle=True)))

        @staticmethod
        def tensor_position_map(
            x: torch.LongTensor
        ) -> torch.LongTensor:
            idx_by_row = torch.arange(
                x.size(0))[:, None].repeat([1, x.size(1)])
            idx_by_tk = torch.arange(x.size(1))[None, :].repeat([x.size(0), 1])

            row_tk = torch.cat(
                [idx_by_row[:, :, None], idx_by_tk[:, :, None]], dim=-1)
            return row_tk

        @staticmethod
        def equal_to_next_map(
            y: torch.LongTensor
        ) -> torch.BoolTensor:
            equals_next = y[:, :-1] == y[:, 1:]
            equals_next = torch.cat([
                equals_next,
                torch.zeros((equals_next.size(0), 1)).bool()],
                dim=1)
            return equals_next

        def decode(
            self, batch: Dict[str, torch.Tensor], y: torch.Tensor,
        ) -> List[Dict[str, Union[int, str]]]:
            """
            convert batch input, y_index tensor
                back to entity metadata
            """
            x = batch["input_ids"]
            offsets = batch["offset_mapping"]

            equals_next = self.equal_to_next_map(y)
            mapping = self.tensor_position_map(x)
            # has positive prediction
            has_pos = (y != 0)

            ids = []
            tokens = []
            start_pos = None

            for e, offset, x_1, y_1, (row_id, token_id) in zip(
                    equals_next[has_pos],
                    offsets[has_pos],
                    x[has_pos], y[has_pos], mapping[has_pos]):

                ids.append(x_1.item())
                if start_pos is None:
                    start_pos = offset[0]
                if e.item() is False:
                    tokens.append({
                        "row_id": row_id.item(),
                        "token_id": token_id.item(),
                        "text": self.tokenizer.decode(ids),
                        "label": self.i2c[y_1.item()],
                        "offset": int(start_pos.item()),
                    })
                    ids = []
                    start_pos = None

            return tokens
    return NERDatasetHF


def load_ner_data_pytorch(file_path: Path):
    """
    Automate the process from langhuan export
        to a pytorch dataloader
    """
    with open(file_path, "r") as f:
        json_data = json.loads(f.read())

    NERDataset = get_ner_data_class()

    return NERDataset(json_data)


def load_ner_data_pytorch_huggingface(
    file_path: Path,
    tokenizer,
    tokenization_options: Dict[str, Any] = dict({
        'return_offsets_mapping': True,
        'padding': 'max_length',
        'return_tensors': 'pt',
        'truncation': True,
        'max_length': 512}),
):
    """
    Automate the process from langhuan export
        to a pytorch dataloader,
    Outputing long tensors tokenized by a
        huggingface tokenizer
    """
    with open(file_path, "r") as f:
        json_data = json.loads(f.read())

    logging.info(f"loading data from {file_path}")
    logging.info(f"loaded data has keys: {json_data.keys()}")

    NERDatasetHF = get_ner_hf_class()

    return NERDatasetHF(json_data, tokenizer=tokenizer,
                        tokenization_options=tokenization_options, )
