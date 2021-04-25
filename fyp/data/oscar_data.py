import pytorch_lightning as pl

# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import base64
import numpy as np
import os
import os.path as op
import random, time, json
import torch
import torch.distributed as dist
import logging
import yaml
import errno
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_from_yaml_file(yaml_file):
    with open(yaml_file, "r") as fp:
        return yaml.load(fp)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + ".tmp"
    with open(filein, "r") as tsvin, open(idxout_tmp, "w") as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + ".lineidx"
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx_file(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info("{}-{}".format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split("\t")]

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.info("loading lineidx: {}".format(self.lineidx))
            with open(self.lineidx, "r") as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, "r")
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info(
                "re-open {} because the process id changed".format(self.tsv_file)
            )
            self._fp = open(self.tsv_file, "r")
            self.pid = os.getpid()


class CaptionTSVDataset(Dataset):
    def __init__(
        self,
        yaml_file,
        teacher_tokenizer=None,
        student_tokenizer=None,
        add_od_labels=True,
        max_img_seq_length=50,
        max_seq_length=70,
        max_seq_a_length=40,
        is_train=True,
        mask_prob=0.15,
        max_masked_tokens=3,
        **kwargs
    ):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.label_file = find_file_path_in_yaml(self.cfg["label"], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg["feature"], self.root)
        self.caption_file = find_file_path_in_yaml(self.cfg.get("caption"), self.root)

        assert op.isfile(self.feat_file)
        if add_od_labels:
            assert op.isfile(self.label_file)
        if is_train:
            assert (
                op.isfile(self.caption_file)
                and teacher_tokenizer is not None
                and student_tokenizer is not None
            )

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)
        self.captions = []
        if self.caption_file and op.isfile(self.caption_file):
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_tensorizer = CaptionTensorizer(
            self.teacher_tokenizer,
            max_img_seq_length,
            max_seq_length,
            max_seq_a_length,
            mask_prob,
            max_masked_tokens,
            is_train=is_train,
        )
        self.student_tokenizer = student_tokenizer
        self.student_tensorizer = CaptionTensorizer(
            self.student_tokenizer,
            max_img_seq_length,
            max_seq_length,
            max_seq_a_length,
            0,
            0,
            is_train=is_train,
        )
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.key2captions = self.prepare_image_key_to_captions()

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0]: i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self):
        if self.captions:
            key2captions = {key: [] for key in self.image_keys}
            for cap in self.captions:
                key2captions[cap["image_id"]].append(cap["caption"])
            return key2captions

    def get_image_index(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            img_key = img_cap_pair["image_id"]
            return self.key2index[img_key]
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        feat_info = json.loads(self.feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info["num_boxes"]
        features = np.frombuffer(
            base64.b64decode(feat_info["features"]), np.float32
        ).reshape((num_boxes, -1))
        return torch.Tensor(features)

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair["caption"]
        return ""

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels = " ".join([l["class"] for l in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = op.splitext(self.caption_file)[0] + "_coco_format.json"
        return cap_file

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(img_idx)
        caption = self.get_caption(idx)
        od_labels = self.get_od_labels(img_idx)
        teacher_example = self.teacher_tensorizer.tensorize_example(
            caption, features, text_b=od_labels
        )
        student_example = self.student_tensorizer.tensorize_example(
            caption, features, text_b=od_labels
        )
        s_input_ids = student_example[0]

        masked_ids = teacher_example[5]

        expanded = []
        for id in masked_ids:
            expanded.append(
                self.student_tokenizer.encode(
                    self.teacher_tokenizer.decode([id.item()])
                )
            )
        expanded = [x for y in expanded for x in y]
        expanded = torch.tensor(expanded).squeeze()
        no_pad_ex = expanded[expanded != 0]
        idxs = (s_input_ids[..., None] == no_pad_ex).any(-1).nonzero().squeeze()
        s_input_ids[idxs] = 103

        student_example[4][idxs] = 1
        student_example = (*student_example[:5], expanded)
        return img_key, teacher_example, student_example

    def __len__(self):
        if self.is_train:
            return len(self.captions)
        return self.get_valid_tsv().num_rows()


class CaptionTensorizer(object):
    def __init__(
        self,
        tokenizer,
        max_img_seq_length=50,
        max_seq_length=70,
        max_seq_a_length=40,
        mask_prob=0.15,
        max_masked_tokens=3,
        is_train=True,
    ):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long)
        )

    def tensorize_example(
        self,
        text_a,
        img_feat,
        text_b=None,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
    ):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[: (self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (
            len(tokens) - 1
        )
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += [pad_token_segment_id] * padding_a_len

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(
                max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens
            )
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint

                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + (
                    [self.tokenizer.pad_token] * (self.max_masked_tokens - num_masked)
                )
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += [pad_token_segment_id] * padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[
                0 : self.max_img_seq_len,
            ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros(
                (self.max_img_seq_len - img_len, img_feat.shape[1])
            )
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start:c_end, c_start:c_end].copy_(
            self._triangle_mask[0:seq_a_len, 0:seq_a_len]
        )
        # full attention for L-L, R-R
        attention_mask[l_start:l_end, l_start:l_end] = 1
        attention_mask[r_start:r_end, r_start:r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start:c_end, l_start:l_end] = 1
        attention_mask[c_start:c_end, r_start:r_end] = 1
        # full attention for L-R:
        attention_mask[l_start:l_end, r_start:r_end] = 1
        attention_mask[r_start:r_end, l_start:l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (
                input_ids,
                attention_mask,
                segment_ids,
                img_feat,
                masked_pos,
                masked_ids,
            )
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos)


class OscarModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.external_dir = "data/external/coco"
        self.interim_dir = "data/interim/coco"
        self.processed_dir = "data/processed/coco"

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset = CocoCaptionsDataset(
            features_directory=self.processed_dir + "/features",
            captions_directory=self.processed_dir + "/captions",
        )

        train_length = int(len(self.dataset) * 0.92)
        val_length = len(self.dataset) - train_length
        self.train, self.val = random_split(
            self.dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8, num_workers=8, shuffle=False)


if __name__ == "__main__":
    data = CocoCaptions()
    data.setup()

    loader = data.train_dataloader()

    print("iter")
    it = iter(loader)
    print("iter done")
    features, mask, captions = next(it)

    print(features.shape)
    print(mask.shape)
    features, mask, captions = next(it)

    print(features.shape)
    print(mask.shape)
