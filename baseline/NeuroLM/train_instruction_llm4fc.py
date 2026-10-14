"""
Instruction fine-tuning / zero-shot evaluation of NeuroLM on LLM4FC datasets.

Adapted from the official train_instruction.py:
  - single GPU, no text-corpus loss (OpenWebText not needed for a classification baseline)
  - datasets prepared by prepare_llm4fc.py (19 ch, 200 Hz, 10 s segments, subject-level splits)
  - evaluation reports AUROC from answer-token probabilities, plus accuracy/F1 from
    greedily generated answer strings, aggregated at window (60 s) and subject level

Run from the LLM4FC repo root:
    python baseline/NeuroLM/train_instruction_llm4fc.py --train_dataset CAUEEG \
        --eval_datasets CAUEEG TUAB TUEP --NeuroLM_path baseline/NeuroLM/checkpoints/NeuroLM-B.pt
"""

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from model.model import GPTConfig
from model.model_neurolm import NeuroLM
from metrics.binary import binary_metrics_fn

import tiktoken

# ── dataset-specific instruction templates (binary tasks) ───────────────────
DATASET_TEXTS = {
    'CAUEEG': ('Question: Does this EEG segment belong to a patient with Alzheimer\'s disease? Answer:',
               {1: ' Yes <|endoftext|>', 0: ' No <|endoftext|>'}),
    'TUAB': ('Question: Is this EEG segment abnormal? Answer:',
             {1: ' Yes <|endoftext|>', 0: ' No <|endoftext|>'}),
    'TUEP': ('Question: Does this EEG segment show epileptic activity? Answer:',
             {1: ' Yes <|endoftext|>', 0: ' No <|endoftext|>'}),
}

# 19-channel standard 10-20 order used by LLM4FC
CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
               'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz']

# standard_1020 channel list copied from dataset.py (NeuroLM upstream)
from dataset import standard_1020  # noqa: E402


def get_chans(ch_names):
    lookup = {name.upper(): i for i, name in enumerate(standard_1020)}
    return [lookup[name.upper()] for name in ch_names]


class LLM4FCLoader(Dataset):
    """Binary-instruction loader for a single LLM4FC dataset split (train/val/test dir)."""

    def __init__(self, root, files, dataset_name, is_instruct=True, is_val=False,
                 eeg_max_len=-1, text_max_len=-1):
        self.root = root
        self.files = files
        self.dataset_name = dataset_name
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        self.question, self.answers = DATASET_TEXTS[dataset_name]

        if is_instruct:
            enc = tiktoken.get_encoding('gpt2')
            encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
            # 50257 for [SEP]
            self.text = {lab: torch.IntTensor([50257] + encode(self.question + ans))
                         for lab, ans in self.answers.items()}
            self.prompt = torch.IntTensor([50257] + encode(self.question))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), 'rb') as f:
            sample = pickle.load(f)
        X = sample['X']       # (19, 2000) uV
        Y = int(sample['y'])
        subject_id = sample.get('subject_id')
        window_id = sample.get('window_id')

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = CHANNELS_19
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)
            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans, i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()

        if self.is_val:
            text = self.prompt
        else:
            text = self.text[Y]
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0
            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans, i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0

        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), \
                subject_id, window_id

        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), \
            subject_id, window_id


def prepare_llm4fc_dataset(root, dataset_name, is_instruct=True, eeg_max_len=-1,
                           text_max_len=-1, limit=-1):
    out = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root, dataset_name, split)
        if not os.path.isdir(split_dir):
            out[split] = None
            continue
        files = sorted(os.listdir(split_dir))
        if limit > 0:
            files = files[:limit]
        out[split] = LLM4FCLoader(split_dir, files, dataset_name, is_instruct=is_instruct,
                                  is_val=(split != 'train'), eeg_max_len=eeg_max_len,
                                  text_max_len=text_max_len)
    return out


def build_model(neuro_lm_path, device):
    """Load the official NeuroLM checkpoint (NeuroLM-B/L/XL.pt)."""
    checkpoint = torch.load(neuro_lm_path, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    # force these config attributes to be equal to the checkpoint
    gpt_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=False,
                    vocab_size=50257, dropout=0.0)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        gpt_args[k] = model_args[k]
    print(f'checkpoint model_args: {gpt_args}')
    gptconf = GPTConfig(**gpt_args)
    model = NeuroLM(gptconf, init_from='scratch')
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected')
    if missing:
        print('missing keys:', missing[:10])
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_train / 1e6:.2f}M; block_size={gpt_args["block_size"]}')
    return model, gpt_args


@torch.no_grad()
def predict_score(model, X_eeg, text, input_chans, input_time, input_mask, gpt_mask,
                  yes_id, no_id, temperature=1.0):
    """P(yes) from the logits of the first generated answer token."""
    B = X_eeg.size(0)
    input_mask4d = input_mask.unsqueeze(1).repeat(1, X_eeg.size(1), 1).unsqueeze(1)
    x_eeg = model.tokenizer(X_eeg, input_chans, input_time, input_mask4d, return_all_tokens=True)
    x_eeg = model.encode_transform_layer(x_eeg)
    x_eeg = x_eeg + model.pos_embed(input_chans)
    logits, _, _ = model.GPT2(x_eeg=x_eeg, x_text=text, eeg_time_idx=input_time,
                              eeg_mask=input_mask, eeg_text_mask=gpt_mask)
    logits = logits[:, -1, :50257] / temperature
    probs = F.softmax(logits, dim=-1)
    p_yes = probs[:, yes_id] / (probs[:, yes_id] + probs[:, no_id] + 1e-12)
    return p_yes


@torch.no_grad()
def generate_answers(model, X_eeg, text, input_chans, input_time, input_mask, gpt_mask,
                     max_new_tokens=5):
    return model.generate(X_eeg, text, input_chans, input_time, input_mask,
                          eeg_text_mask=gpt_mask, max_new_tokens=max_new_tokens)


def aggregate_and_report(seg_scores, seg_labels, seg_windows, tag, out_log):
    """Aggregate 10 s segment scores to 60 s window level, report metrics."""
    seg_scores = np.asarray(seg_scores)
    seg_labels = np.asarray(seg_labels, dtype=int)
    seg_windows = np.asarray(seg_windows, dtype=int)

    order = np.argsort(seg_windows)
    sorted_keys = seg_windows[order]
    boundaries = np.nonzero(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
    groups = np.split(order, boundaries)
    scores = np.array([seg_scores[g].mean() for g in groups])
    labels = np.array([seg_labels[g][0] for g in groups])

    metrics = ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'f1']
    if len(np.unique(labels)) < 2:
        res = {m: float('nan') for m in metrics}
    else:
        print(f'    [debug window] labels {labels.shape} {labels.dtype} '
              f'scores {scores.shape} {scores.dtype}')
        res = binary_metrics_fn(labels, scores, metrics=metrics)
    lines = [f'  [{tag}] segments={len(seg_scores)}',
             f'    window  n={len(scores):6d} ' +
             ' '.join(f'{k}={v:.4f}' for k, v in res.items())]
    print('\n'.join(lines))
    out_log.write('\n'.join(lines) + '\n')
    return res


@torch.no_grad()
def evaluate(model, loaders, dataset_name, device, dtype, yes_id, no_id, out_log,
             batch_size=64):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    results = {}
    for split, loader in loaders.items():
        if loader is None:
            continue
        seg_scores, seg_labels, seg_windows = [], [], []
        gen_preds, gen_labels = [], []
        data_loader = torch.utils.data.DataLoader(
            loader, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
        for batch in tqdm(data_loader, desc=f'eval {dataset_name}/{split}'):
            X_eeg, text, label, input_chans, input_time, input_mask, gpt_mask, subj, win = batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            gpt_mask = gpt_mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                p_yes = predict_score(model, X_eeg, text, input_chans, input_time,
                                      input_mask, gpt_mask, yes_id, no_id)
                gen = generate_answers(model, X_eeg, text, input_chans, input_time,
                                       input_mask, gpt_mask)
            gen = gen[:, -5:]  # only the generated tokens
            for i, t in enumerate(gen):
                toks = t.tolist()
                if yes_id in toks and no_id not in toks:
                    gen_preds.append(1)
                elif no_id in toks and yes_id not in toks:
                    gen_preds.append(0)
                else:
                    gen_preds.append(-1)
            seg_scores.extend(p_yes.cpu().numpy().tolist())
            seg_labels.extend(label.numpy().tolist())
            gen_labels.extend(label.numpy().tolist())
            seg_windows.extend(win)

        window_metrics = aggregate_and_report(
            seg_scores, seg_labels, np.asarray(seg_windows, dtype=int), split, out_log)
        valid = [i for i, p in enumerate(gen_preds) if p >= 0]
        acc = (np.mean([gen_preds[i] == seg_labels[i] for i in valid])
               if valid else float('nan'))
        out_log.write(f'  [{dataset_name}/{split}] generated-answer accuracy '
                      f'({len(valid)}/{len(gen_preds)} parsed): {acc:.4f}\n')
        print(f'  [{dataset_name}/{split}] generated-answer accuracy '
              f'({len(valid)}/{len(gen_preds)} parsed): {acc:.4f}')
        results[split] = {'window': window_metrics, 'gen_accuracy': acc}
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description='NeuroLM instruction tuning on LLM4FC')
    parser.add_argument('--dataset_dir', default='baseline/NeuroLM/llm4fc_data',
                        help='root of prepared datasets (prepare_llm4fc.py --out_dir)')
    parser.add_argument('--out_dir', default='baseline/NeuroLM/out_llm4fc',
                        help='where to save checkpoints/results')
    parser.add_argument('--NeuroLM_path', default='baseline/NeuroLM/checkpoints/NeuroLM-B.pt')
    parser.add_argument('--train_dataset', default=None,
                        help='dataset used for instruction tuning (None = eval only)')
    parser.add_argument('--eval_datasets', nargs='+', default=['CAUEEG', 'TUAB', 'TUEP'])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--patience', default=3, type=int,
                        help='early stopping patience on val window roc_auc')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--eeg_max_len', default=-1, type=int,
                        help='pad EEG tokens to this length (-1 = no padding)')
    parser.add_argument('--text_max_len', default=80, type=int)
    parser.add_argument('--limit', default=-1, type=int,
                        help='max files per split (smoke test; -1 = all)')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)
    out_log = open(os.path.join(args.out_dir, 'run.log'), 'a')

    enc = tiktoken.get_encoding('gpt2')
    yes_id = enc.encode(' Yes')[0]
    no_id = enc.encode(' No')[0]
    print(f'answer token ids: Yes={yes_id}, No={no_id}')
    out_log.write(f'answer token ids: Yes={yes_id}, No={no_id}\n')

    # ── model ──
    model, gpt_args = build_model(args.NeuroLM_path, device)
    print(f'GPT block_size={gpt_args["block_size"]}')

    # ── datasets ──
    datasets = {}
    for name in args.eval_datasets:
        datasets[name] = prepare_llm4fc_dataset(args.dataset_dir, name,
                                                eeg_max_len=args.eeg_max_len,
                                                text_max_len=args.text_max_len,
                                                limit=args.limit)
    if args.train_dataset is not None:
        assert args.train_dataset in datasets, \
            f'train dataset {args.train_dataset} not in eval_datasets'
        train_loader = torch.utils.data.DataLoader(
            datasets[args.train_dataset]['train'], batch_size=args.batch_size,
            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # ── optimizer / scheduler ──
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (0.9, 0.95), device)
    num_steps_per_epoch = (len(datasets[args.train_dataset]['train']) // args.batch_size
                           if args.train_dataset else 0)

    def lr_at(step, epoch):
        total_steps = num_steps_per_epoch * args.epochs
        warmup = num_steps_per_epoch * args.warmup_epochs
        if step < warmup:
            return args.learning_rate * (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return args.min_lr + 0.5 * (args.learning_rate - args.min_lr) * (1 + math.cos(math.pi * p))

    scaler = torch.amp.GradScaler(device, enabled=False)

    if args.eval_only:
        print('== eval-only mode ==')
        all_res = {}
        for name in args.eval_datasets:
            all_res[name] = evaluate(model, {k: v for k, v in datasets[name].items() if k != 'train'},
                                     name, device, dtype,
                                     yes_id, no_id, out_log, batch_size=args.batch_size)
        with open(os.path.join(args.out_dir, 'eval_results.json'), 'w') as f:
            json.dump(all_res, f, indent=2, default=float)
        out_log.close()
        return

    # ── training loop ──
    best_auc = -1.0
    best_epoch = -1
    no_improve = 0
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            X_eeg, X_text, Y_text, input_chans, input_time, input_mask, gpt_mask, subj, win = batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            X_text = X_text.to(device, non_blocking=True)
            Y_text = Y_text.to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            gpt_mask = gpt_mask.to(device, non_blocking=True)

            Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)),
                               fill_value=-1 - model.GPT2.config.vocab_size,
                               device=device, dtype=torch.long)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_at(global_step, epoch)

            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                loss1, log1, _ = model(X_eeg, Y_eeg, X_text, Y_text, input_chans,
                                       input_time, input_mask, eeg_text_mask=gpt_mask)
            scaler.scale(loss1).backward()
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss1.item())
            global_step += 1

        train_loss = np.mean(losses)
        dt = time.time() - t0
        msg = (f'epoch {epoch} step {global_step} train_loss={train_loss:.4f} '
               f'lr={optimizer.param_groups[0]["lr"]:.2e} dt={dt:.0f}s')
        print(msg)
        out_log.write(msg + '\n')

        # validation (train dataset only; early stopping on window roc_auc)
        val_res = evaluate(model, {k: datasets[args.train_dataset][k] for k in ['val', 'test']},
                           args.train_dataset, device, dtype, yes_id, no_id, out_log,
                           batch_size=args.batch_size)
        val_auc = val_res['val']['window']['roc_auc']
        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, no_improve = val_auc, epoch, 0
            torch.save({'model': model.state_dict(), 'model_args': gpt_args, 'epoch': epoch},
                       os.path.join(args.out_dir, 'checkpoints', 'ckpt-best.pt'))
            print(f'  saved best ckpt (val window roc_auc={val_auc:.4f})')
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f'early stop at epoch {epoch} (best {best_auc:.4f} @ epoch {best_epoch})')
            break

    # ── final test eval on all datasets with best ckpt ──
    best_ckpt = os.path.join(args.out_dir, 'checkpoints', 'ckpt-best.pt')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f'loaded best ckpt (epoch {ckpt["epoch"]}, val auc {best_auc:.4f})')
    all_res = {'best_val_auc': best_auc, 'best_epoch': best_epoch}
    for name in args.eval_datasets:
        all_res[name] = evaluate(model, {k: v for k, v in datasets[name].items() if k != 'train'},
                                 name, device, dtype,
                                 yes_id, no_id, out_log, batch_size=args.batch_size)
    with open(os.path.join(args.out_dir, 'test_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2, default=float)
    out_log.close()


if __name__ == '__main__':
    main()
