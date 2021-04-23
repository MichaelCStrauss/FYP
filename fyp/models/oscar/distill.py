import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import pytorch_lightning.metrics.functional as metrics

from fyp.models.oscar.pytorch_transformers.tokenization_bert import BertTokenizer
from fyp.models.oscar.modeling_bert import BertConfig, BertForImageCaptioning
from fyp.models.oscar.pytorch_transformers import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from fyp.data.oscar_data import CaptionTSVDataset
from torchinfo import summary

import os
import os.path as op

from fyp.models.oscar.args import args
from tqdm import tqdm


class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels, temperature, alpha=0.5):
        return (
            self.kld_loss(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
            )
            * alpha
            * temperature ** 2
            + self.ce_loss(student_logits, labels) * (1 - alpha)
        )


class HiddenLayerMSELoss(nn.Module):
    def __init__(self, teacher_hidden_dim, student_hidden_dim):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.transform = nn.Linear(student_hidden_dim, teacher_hidden_dim, bias=False)

    def forward(self, teacher_hiddens, student_hiddens, inputs):
        total_loss = torch.tensor(0, dtype=torch.float32).cuda()
        for i in range(1, len(student_hiddens)):
            masked_pos = inputs["masked_pos"]
            num_features = inputs["img_feats"].shape[1]

            teacher_hidden = teacher_hiddens[i * 3 - 1]
            student_hidden = student_hiddens[i]

            # teacher_hidden = teacher_hidden[:, num_features:, :]
            # student_hidden = student_hidden[:, num_features:, :]

            # teacher_hidden = teacher_hidden[masked_pos == 1, :]
            # student_hidden = student_hidden[masked_pos == 1, :]

            transformed_student = self.transform(student_hidden)

            total_loss += self.mse_loss(transformed_student, teacher_hidden)
        return total_loss


# class AttentionLayerMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()

#     def forward(self, teacher_attns, student_attns):
#         total_loss = torch.tensor(0, dtype=torch.float32).cuda()
#         for i in range(len(student_attns)):
#             teacher_attn = teacher_hiddens[(i+1) * 3 - 1]
#             student_attn = student_hiddens[i]

#             teacher_hidden = teacher_hidden[:, num_features:, :]
#             student_hidden = student_hidden[:, num_features:, :]

#             teacher_hidden = teacher_hidden[masked_pos == 1, :]
#             student_hidden = student_hidden[masked_pos == 1, :]

#             transformed_student = self.transform(student_hidden)

#             total_loss += self.mse_loss(transformed_student, teacher_hidden)
#         return total_loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def load_student_model(args):
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = (
        BertConfig,
        BertForImageCaptioning,
        BertTokenizer,
    )
    assert args.student_model_name_or_path is not None
    config = config_class.from_pretrained(
        args.student_model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task="image_captioning",
    )
    config.output_hidden_states = True
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    config.output_attentions = False
    model = model_class.from_pretrained(
        args.student_model_name_or_path,
        from_tf=False,
        config=config,
    )

    return model, config


def load_teacher_model(args):
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = (
        BertConfig,
        BertForImageCaptioning,
        BertTokenizer,
    )
    assert args.teacher_model_name_or_path is not None
    config = config_class.from_pretrained(
        args.teacher_model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task="image_captioning",
    )
    config.output_hidden_states = True
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    config.output_attentions = False
    model = model_class.from_pretrained(
        args.teacher_model_name_or_path,
        from_tf=False,
        config=config,
    )

    return model, config


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(
        args.output_dir, "checkpoint-{}-{}".format(epoch, iteration)
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, "training_args.bin"))
            tokenizer.save_pretrained(checkpoint_dir)
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def main():
    if args.wandb:
        wandb.init(config=args)

    tokenizer = BertTokenizer.from_pretrained(
        args.teacher_model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    teacher_model, teacher_config = load_teacher_model(args)
    student_model, student_config = load_student_model(args)
    summary(student_model)
    if args.wandb:
        wandb.watch(student_model)

    teacher_model.to(args.device)
    student_model.to(args.device)

    yaml_file = args.train_yaml
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    dataset = CaptionTSVDataset(
        yaml_file,
        tokenizer=tokenizer,
        add_od_labels=args.add_od_labels,
        max_img_seq_length=args.max_img_seq_length,
        max_seq_length=args.max_seq_length,
        max_seq_a_length=args.max_seq_a_length,
        is_train=True,
        mask_prob=args.mask_prob,
        max_masked_tokens=args.max_masked_tokens,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        sampler=torch.utils.data.sampler.RandomSampler(dataset),
        batch_size=8,
        pin_memory=True,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = (
            len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
        )

    hidden_loss = HiddenLayerMSELoss(
        teacher_config.hidden_size, student_config.hidden_size
    )
    hidden_loss.to(args.device)
    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in student_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in student_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": hidden_loss.parameters(),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    hidden_loss.train()

    global_step = 0

    for epoch in range(int(args.num_train_epochs)):
        epoch_loss, epoch_acc = 0.0, 0.0
        num_steps = len(data_loader)
        print(num_steps)
        pbar = tqdm(data_loader)
        for step, (img_keys, batch) in enumerate(pbar):
            global_step += 1
            batch = tuple(t.to(args.device) for t in batch)

            teacher_model.eval()
            student_model.train()
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "img_feats": batch[3],
                "masked_pos": batch[4],
                "masked_ids": batch[5],
            }
            masked_ids = inputs["masked_ids"]
            masked_ids = masked_ids[masked_ids != 0]
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            student_outputs = student_model(**inputs)

            loss = hidden_loss(
                teacher_outputs[2],
                student_outputs[2],
                inputs,
            )
            epoch_loss += loss.item() / num_steps

            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), args.max_grad_norm
            )

            if step % 1000 == 0:
                print(f"Teacher loss: {teacher_outputs[0]}")
                print(f"Student loss: {student_outputs[0]}")
                print(f"KD Loss: {loss}")

            loss.backward()

            optimizer.step()
            scheduler.step()
            student_model.zero_grad()

            pbar.set_postfix(
                {
                    "student_loss": round(student_outputs[0].item(), 2),
                    "teacher_loss": round(teacher_outputs[0].item(), 2),
                    "loss": round(loss.item(), 2),
                }
            )

            if args.wandb:
                wandb.log(
                    {
                        "student_loss": student_outputs[0],
                        "teacher_loss": teacher_outputs[0],
                        "loss": loss,
                    }
                )

        if args.wandb:
            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                },
                step=global_step,
            )

        save_checkpoint(student_model, tokenizer, args, epoch, num_steps)


if __name__ == "__main__":
    main()