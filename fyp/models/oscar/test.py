from fyp.models.oscar.other import CaptionTensorizer
from fyp.models.oscar.pytorch_transformers.tokenization_bert import BertTokenizer
from fyp.models.oscar.pytorch_transformers.modeling_bert import BertConfig
from fyp.models.oscar.modeling_bert import BertForImageCaptioning
import torch
import logging
import numpy as np
import os
import os.path as op
import argparse
import json
import base64
import pandas as pd

logger = logging.getLogger()


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, "training_args.bin"))
    if hasattr(train_args, "max_seq_a_length"):
        if hasattr(train_args, "scst") and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning(
            "Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}".format(
                max_seq_length, args.max_gen_length, max_od_labels_len
            )
        )

    override_params = [
        "max_seq_a_length",
        "do_lower_case",
        "add_od_labels",
        "max_img_seq_length",
    ]
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning(
                    "Override {} with train args: {} -> {}".format(
                        param, test_v, train_v
                    )
                )
                setattr(args, param, train_v)
    return args


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default="datasets/coco_caption",
    type=str,
    required=False,
    help="The input data dir with all required files.",
)
parser.add_argument(
    "--train_yaml",
    default="train.yaml",
    type=str,
    required=False,
    help="yaml file for training.",
)
parser.add_argument(
    "--test_yaml",
    default="test.yaml",
    type=str,
    required=False,
    help="yaml file for testing.",
)
parser.add_argument(
    "--val_yaml",
    default="val.yaml",
    type=str,
    required=False,
    help="yaml file used for validation during training.",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=False,
    help="Path to pre-trained model or model type.",
)
parser.add_argument(
    "--output_dir",
    default="output/",
    type=str,
    required=False,
    help="The output directory to save checkpoint and test results.",
)
parser.add_argument(
    "--loss_type",
    default="sfmx",
    type=str,
    help="Loss function types: support kl, x2, sfmx",
)
parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name.",
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name.",
)
parser.add_argument(
    "--max_seq_length",
    default=70,
    type=int,
    help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, "
    "sequences shorter will be padded.",
)
parser.add_argument(
    "--max_seq_a_length",
    default=40,
    type=int,
    help="The maximum sequence length for caption.",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_test", action="store_true", help="Whether to run inference.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--mask_prob",
    default=0.15,
    type=float,
    help="Probability to mask input sentence during training.",
)
parser.add_argument(
    "--max_masked_tokens",
    type=int,
    default=3,
    help="The max number of masked tokens per sentence.",
)
parser.add_argument(
    "--add_od_labels",
    default=False,
    action="store_true",
    help="Whether to add object detection labels or not",
)
parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
parser.add_argument(
    "--max_img_seq_length",
    default=50,
    type=int,
    help="The maximum total input image sequence length.",
)
parser.add_argument(
    "--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension."
)
parser.add_argument(
    "--img_feature_type", default="frcnn", type=str, help="Image feature type."
)
parser.add_argument(
    "--tie_weights",
    default=False,
    action="store_true",
    help="Whether to tie decoding weights to that of encoding",
)
parser.add_argument(
    "--freeze_embedding",
    default=False,
    action="store_true",
    help="Whether to freeze word embeddings in Bert",
)
parser.add_argument("--label_smoothing", default=0, type=float, help=".")
parser.add_argument("--drop_worst_ratio", default=0, type=float, help=".")
parser.add_argument("--drop_worst_after", default=0, type=int, help=".")
parser.add_argument(
    "--per_gpu_train_batch_size",
    default=64,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=64,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)
parser.add_argument(
    "--output_mode",
    default="classification",
    type=str,
    help="output mode, support classification or regression.",
)
parser.add_argument(
    "--num_labels",
    default=2,
    type=int,
    help="num_labels is 2 for classification and 1 for regression.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before backward.",
)
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam."
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
parser.add_argument(
    "--scheduler", default="linear", type=str, help="constant or linear or"
)
parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
parser.add_argument(
    "--num_train_epochs",
    default=40,
    type=int,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="Total number of training steps. Override num_train_epochs.",
)
parser.add_argument("--logging_steps", type=int, default=20, help="Log every X steps.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=-1,
    help="Save checkpoint every X steps. Will also perform evaluatin.",
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each save_steps.",
)
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA.")
parser.add_argument(
    "--local_rank", type=int, default=0, help="For distributed training."
)
parser.add_argument(
    "--seed", type=int, default=88, help="random seed for initialization."
)
# for self-critical sequence training
parser.add_argument(
    "--scst", action="store_true", help="Self-critical sequence training"
)
parser.add_argument(
    "--sc_train_sample_n",
    type=int,
    default=5,
    help="number of sampled captions for sc training",
)
parser.add_argument(
    "--sc_baseline_type",
    type=str,
    default="greedy",
    help="baseline tyep of REINFORCE algorithm",
)
parser.add_argument(
    "--sc_beam_size", type=int, default=1, help="beam size for scst training"
)
parser.add_argument(
    "--cider_cached_tokens",
    type=str,
    default="coco-train-words.p",
    help="path to cached cPickle file used to calculate CIDEr scores",
)
# for generation
parser.add_argument(
    "--eval_model_dir", type=str, default="", help="Model directory for evaluation."
)
parser.add_argument(
    "--max_gen_length", type=int, default=20, help="max length of generated sentences"
)
parser.add_argument(
    "--output_hidden_states", action="store_true", help="Turn on for fast decoding"
)
parser.add_argument(
    "--num_return_sequences", type=int, default=1, help="repeating times per image"
)
parser.add_argument("--num_beams", type=int, default=1, help="beam search width")
parser.add_argument(
    "--num_keep_best",
    type=int,
    default=1,
    help="number of hypotheses to keep in beam search",
)
parser.add_argument(
    "--temperature", type=float, default=1, help="temperature in softmax for sampling"
)
parser.add_argument(
    "--top_k", type=int, default=0, help="filter distribution for sampling"
)
parser.add_argument(
    "--top_p", type=float, default=1, help="filter distribution for sampling"
)
parser.add_argument(
    "--repetition_penalty",
    type=int,
    default=1,
    help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)",
)
parser.add_argument(
    "--length_penalty", type=int, default=1, help="beam search length penalty"
)
# for Constrained Beam Search
parser.add_argument(
    "--use_cbs", action="store_true", help="Use constrained beam search for decoding"
)
parser.add_argument(
    "--min_constraints_to_satisfy",
    type=int,
    default=2,
    help="minimum number of constraints to satisfy",
)
args = parser.parse_args()

args.num_gpus = 1
args.distributed = False
args.device = torch.device("cuda")

args = restore_training_settings(args)

config_class, model_class, tokenizer_class = (
    BertConfig,
    BertForImageCaptioning,
    BertTokenizer,
)

checkpoint = args.eval_model_dir
assert op.isdir(checkpoint)
config = config_class.from_pretrained(checkpoint)
config.output_hidden_states = args.output_hidden_states
tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=True)
logger.info("Evaluate the following checkpoint: %s", checkpoint)
model = model_class.from_pretrained(checkpoint, config=config)

model.to(args.device)


(
    cls_token_id,
    sep_token_id,
    pad_token_id,
    mask_token_id,
    period_token_id,
) = tokenizer.convert_tokens_to_ids(
    [
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token,
        tokenizer.mask_token,
        ".",
    ]
)
world_size = 1

model.eval()
inputs_param = {
    "is_decode": True,
    "do_sample": False,
    "bos_token_id": cls_token_id,
    "pad_token_id": pad_token_id,
    "eos_token_ids": [sep_token_id],
    "mask_token_id": mask_token_id,
    # for adding od labels
    "add_od_labels": args.add_od_labels,
    "od_labels_start_posid": args.max_seq_a_length,
    # hyperparameters of beam search
    "max_length": args.max_gen_length,
    "num_beams": args.num_beams,
    "temperature": args.temperature,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "repetition_penalty": args.repetition_penalty,
    "length_penalty": args.length_penalty,
    "num_return_sequences": args.num_return_sequences,
    "num_keep_best": args.num_keep_best,
}

if args.use_cbs:
    inputs_param.update(
        {
            "use_cbs": True,
            "min_constraints_to_satisfy": args.min_constraints_to_satisfy,
        }
    )

tensorizer = CaptionTensorizer(
    tokenizer,
    max_img_seq_length=args.max_img_seq_length,
    max_seq_length=args.max_seq_length,
    max_seq_a_length=args.max_seq_a_length,
    is_train=False,
    mask_prob=args.mask_prob,
    max_masked_tokens=args.max_masked_tokens,
)

# data = pd.read_csv('data/processed/coco-oscar/test.feature.tsv', sep='\t', header=None).rename(columns={0: "index", 1: "data"})
# print(data)

data = ""
with open('data/processed/coco-oscar/test.feature.tsv') as f:
    line = f.readline()
    data = line.split('\t')[1]

import io
import hashlib
buffer = io.BytesIO()

print(model.state_dict()['bert.encoder.layer.0.attention.self.query.weight'])
# torch.save(model.state_dict(), buffer)
torch.save(model.state_dict()['bert.encoder.layer.0.attention.self.query.weight'].cpu().tolist(), buffer)

m = hashlib.md5()

m.update(buffer.getvalue())

print(m.hexdigest())

with torch.no_grad():
    img_keys = [0]
    feat_info = json.loads(data)
    num_boxes = feat_info['num_boxes']
    features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
            ).reshape((num_boxes, -1))
    features =  torch.Tensor(features)

    caption = ""
    example = tensorizer.tensorize_example(
        caption, features, text_b="man helmet scooter sky ground ground mountain wheel man shirt mountain road mountain shoe trees rocks bike wheel post bridge grass fence jeans mirror bush grass mountains road bridge people trees tree man post bag motorcycle bridge"
    )
    batch = tuple(t.unsqueeze(0).to(args.device) for t in example)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "img_feats": batch[3],
        "masked_pos": batch[4],
    }
    print(inputs['img_feats'].shape)
    if args.use_cbs:
        inputs.update(
            {
                "fsm": batch[5],
                "num_constraints": batch[6],
            }
        )
    inputs.update(inputs_param)
    # captions, logprobs
    outputs = model(**inputs)
    all_caps = outputs[0]  # batch_size * num_keep_best * max_len
    all_confs = torch.exp(outputs[1])

    for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
        res = []
        for cap, conf in zip(caps, confs):
            print(cap.tolist())
            cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
            res.append({"caption": cap, "conf": conf.item()})
        if isinstance(img_key, torch.Tensor):
            img_key = img_key.item()
        print(json.dumps(res))