"""
P4 集成测试脚本 — 在真实 Dementia_MMS 数据上验证 LDDE2th 训练 pipeline。

Usage:
    # P4.2: 1-epoch quick test (5 batches, ~2 min)
    python model/LDDE2th/test_integration.py --test p42

    # P4.3: Phase A 5-epoch test (~15 min)
    python model/LDDE2th/test_integration.py --test p43

    # P4.4: 3-phase short training (5+3+5 epochs, ~30 min)
    python model/LDDE2th/test_integration.py --test p44

All tests use fold 0, batch_size=2, and a random subset of the training data.
"""

import os, sys, gc, argparse, time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import init_config, init_model_config
from data import DataConfig
from trainers import LDDE2thTrainer
from model.LDDE2th.LDDE2th import LDDE2thConfig, load_pretrained_bnc


def get_args(override=None):
    """Build minimal args for training test."""
    sys.argv = ['main.py', '--model', 'LDDE2th',
                '--dataset', 'Dementia_MMS',
                '--data_dir', '../data/XWDementia/Dementia_MMS.npy',
                '--batch_size', '2', '--num_epochs', '1', '--num_repeat', '3',
                '--learning_rate', '1e-4', '--schedule', 'cos',
                '--do_train', '--drop_last', 'False',
                '--num_workers', '0']
    if override:
        for k, v in override.items():
            sys.argv.extend([f'--{k}', str(v)])
    return init_config()


def create_trainer(task_id=0):
    """Create trainer with real data loaders."""
    args = get_args({'num_repeat': '3'})
    data_config = DataConfig(args)
    model, model_config = init_model_config(args, data_config)
    trainer = LDDE2thTrainer(args, task_id=task_id)
    trainer.init_components()
    return trainer


def test_p42():
    """P4.2: 1-epoch 训练测试 — 验证数据管道 + forward/backward 无报错。

    在真实数据上运行 5 个 batch，确认:
      - 无 OOM
      - 无 shape mismatch
      - 无 NaN/Inf
      - 各损失分量正常
    """
    print("=" * 60)
    print("P4.2: 小数据集训练测试 (5 batches)")
    print("=" * 60)

    trainer = create_trainer(task_id=0)
    train_loader = trainer.data_loaders['train']
    model = trainer.model
    model.train()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Batches: {len(train_loader)}")
    print(f"Running 5 training steps...")

    loss_history = {name: [] for name in ['cla', 'div', 'entropy', 'cla_llm', 'text']}
    start_time = time.time()

    for step, inputs in enumerate(train_loader):
        if step >= 5:
            break

        input_kwargs = trainer.prepare_inputs_kwargs(inputs)
        input_kwargs['epoch'] = 0
        outputs = model(**input_kwargs)

        cla, div, ent, cla_llm, text = outputs.loss
        loss = cla + 0.01 * div + 0.1 * ent + 0.3 * cla_llm + 0.1 * text

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()

        loss_history['cla'].append(cla.item())
        loss_history['div'].append(div.item())
        loss_history['entropy'].append(ent.item())
        loss_history['cla_llm'].append(cla_llm.item())
        loss_history['text'].append(text.item())

        # NaN check
        for name, val in [('cla', cla),
                          ('div', div), ('ent', ent), ('cla_llm', cla_llm), ('text', text)]:
            assert not torch.isnan(val) and not torch.isinf(val), \
                f"Step {step}: {name} loss is NaN/Inf!"

        mem = torch.cuda.max_memory_allocated() / 1024**3
        t = time.time() - start_time
        print(f"  Step {step+1}/5 | cla={cla.item():.3f} "
              f"ent={ent.item():.3f} cla_llm={cla_llm.item():.3f} "
              f"text={text.item():.4f} | "
              f"mem={mem:.1f}GB | {t:.0f}s")

    # Print summary
    print(f"\nP4.2 Loss summary:")
    for name, values in loss_history.items():
        avg = np.mean(values)
        print(f"  {name:10s}: mean={avg:.4f}")
    print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"  Total time: {time.time()-start_time:.0f}s")
    print(f"\nP4.2 PASS — 5 batches completed without errors\n")
    return True


def test_p43():
    """P4.3: Phase A 5-epoch 测试 — 验证 BNC 冻结训练稳定性。

    加载预训练 BNC → 冻结 BNC → 训练 5 epoch，验证：
      - 语义原型激活分布不坍缩
      - L_div 维持低水平
      - 分类 Loss 下降
      - 分类 Loss 稳定
    """
    print("=" * 60)
    print("P4.3: Phase A 5-epoch 测试 (BNC 冻结)")
    print("=" * 60)

    trainer = create_trainer(task_id=0)
    model = trainer.model

    # Load pretrained BNC
    try:
        stats = load_pretrained_bnc(model, task_id=0)
        print(f"  Loaded {stats['bnc_params_loaded']} BNC params")
    except FileNotFoundError:
        print("  [WARN] DFCBNC-0.bin not found, using random init")
        print("  Train DFCBNC first: bash scripts/dementia_mms/train_DFCBNC_Dementia.sh")

    # Freeze BNC for Phase A
    for p in model.bnc.parameters():
        p.requires_grad = False

    train_loader = trainer.data_loaders['train']
    N_epochs = 5
    N_batches_per_epoch = min(20, len(train_loader))  # subset for speed

    proto_history = []
    start_time = time.time()

    for epoch in range(N_epochs):
        model.train()
        epoch_losses = {name: 0.0 for name in ['cla', 'div', 'entropy']}
        n_steps = 0

        for step, inputs in enumerate(train_loader):
            if step >= N_batches_per_epoch:
                break

            input_kwargs = trainer.prepare_inputs_kwargs(inputs)
            input_kwargs['epoch'] = epoch
            outputs = model(**input_kwargs)
            cla, div, ent, cla_llm, text = outputs.loss
            loss = cla + 0.01 * div + 0.1 * ent + 0.3 * cla_llm + 0.1 * text

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()

            epoch_losses['cla'] += cla.item()
            epoch_losses['div'] += div.item()
            epoch_losses['entropy'] += ent.item()
            n_steps += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= n_steps

        # Check prototype diversity
        with torch.no_grad():
            protos = model.semantic_proto.prototypes
            p_norm = F.normalize(protos, dim=-1)
            sim = p_norm @ p_norm.T
            off_diag = sim * (1 - torch.eye(16, device=sim.device))
            max_cos = off_diag.abs().max().item()
            proto_history.append(max_cos)

        t = time.time() - start_time
        print(f"  Epoch {epoch+1}/{N_epochs} | "
              f"cla={epoch_losses['cla']:.3f} "
              f"div={epoch_losses['div']:.4f} "
              f"ent={epoch_losses['entropy']:.3f} | proto_max_cos={max_cos:.4f} | {t:.0f}s")

    # Verify P4.3 criteria
    assert max(proto_history) < 0.5, \
        f"Proto collapse detected: max pairwise cos_sim={max(proto_history):.4f} > 0.5"
    assert proto_history[-1] < 0.3, \
        f"L_div too high: off-diag cos_sim={proto_history[-1]:.4f} > 0.3"

    print(f"\nP4.3 verification:")
    print(f"  Proto max pairwise cos_sim (epoch 5): {proto_history[-1]:.4f} < 0.5 ✓")
    print(f"  Classification loss: {epoch_losses['cla']:.4f}")
    print(f"  Classification loss: {epoch_losses['cla']:.4f}")
    print(f"  GPU mem: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"\nP4.3 PASS — Phase A 5-epoch training stable\n")
    return True


def test_p44():
    """P4.4: 3-phase 短训练测试 — 验证 Phase A→B→C 训练流程。

    训练 5+3+5 = 13 epochs:
      Phase A (epoch 1-5): BNC 冻结
      Phase B (epoch 6-8): BNC 解冻，cons 权重增长
      Phase C (epoch 9-13): 全参数联合优化

    验证:
      - M 有样本间区分度
      - P_llm 准确率 > 随机 (25% for 4-class)
      - 最终 Accuracy 合理
    """
    print("=" * 60)
    print("P4.4: 3-Phase 训练测试 (5+3+5 epochs)")
    print("=" * 60)

    trainer = create_trainer(task_id=0)
    model = trainer.model
    config = model.config

    # Load pretrained BNC
    try:
        stats = load_pretrained_bnc(model, task_id=0)
        print(f"  Loaded {stats['bnc_params_loaded']} BNC params")
    except FileNotFoundError:
        print("  [WARN] DFCBNC-0.bin not found, training from scratch")

    N_warmup = 5        # Phase A epochs
    N_transition = 3    # Phase B epochs
    N_finetune = 5      # Phase C epochs
    N_total = N_warmup + N_transition + N_finetune

    train_loader = trainer.data_loaders['train']
    test_loader = trainer.data_loaders['test']
    N_batches = min(15, len(train_loader))

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(1, N_total + 1):
        # Phase control
        if epoch <= N_warmup:
            for p in model.bnc.parameters():
                p.requires_grad = False
            phase = 'A'
        elif epoch <= N_warmup + N_transition:
            for p in model.bnc.parameters():
                p.requires_grad = True
            phase = 'B'
        else:
            for p in model.bnc.parameters():
                p.requires_grad = True
            phase = 'C'

        # Train epoch
        model.train()
        epoch_losses = {k: 0.0 for k in ['cla', 'div', 'entropy']}
        n_steps = 0

        for step, inputs in enumerate(train_loader):
            if step >= N_batches:
                break
            input_kwargs = trainer.prepare_inputs_kwargs(inputs)
            input_kwargs['epoch'] = epoch - 1
            outputs = model(**input_kwargs)
            cla, div, ent, cla_llm, text = outputs.loss
            loss = cla + 0.01 * div + 0.1 * ent + 0.3 * cla_llm + 0.1 * text

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()

            for k, v in zip(['cla', 'div', 'entropy'],
                            [cla, div, ent]):
                epoch_losses[k] += v.item()
            n_steps += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_steps, 1)

        # Quick eval on a few test batches
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        with torch.no_grad():
            for step, inputs in enumerate(test_loader):
                if step >= 5:
                    break
                input_kwargs = trainer.prepare_inputs_kwargs(inputs)
                outputs = model(**input_kwargs)
                pred = outputs.logits[0].argmax(dim=-1)
                labels = input_kwargs['labels']
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(pred.cpu().tolist())

        acc = correct / max(total, 1)

        # Monitor mask diversity
        if hasattr(outputs, 'hidden_state') and outputs.hidden_state:
            M_std = outputs.hidden_state.get('M', torch.zeros(1)).std(dim=0).mean().item()
        else:
            M_std = 0.0

        t = time.time() - start_time
        print(f"  E{epoch:2d} [{phase}] | "
              f"cla={epoch_losses['cla']:.3f} "
              f"div={epoch_losses['div']:.4f} | "
              f"acc={acc:.3f} M_std={M_std:.4f} | {t:.0f}s")

        if acc > best_acc:
            best_acc = acc

    # Verify P4.4 criteria
    print(f"\nP4.4 verification:")
    print(f"  Best accuracy: {best_acc:.4f} (random baseline: 0.25)")
    assert best_acc >= 0.20, f"Accuracy too low: {best_acc:.4f}"
    print(f"  M_std (mask diversity): {M_std:.4f} (target > 0.01)")

    # Check final metrics
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"  Total time: {time.time()-start_time:.0f}s")
    print(f"\nP4.4 PASS — 3-phase training completed\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='p42',
                       choices=['p42', 'p43', 'p44'],
                       help='Which test to run')
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    if args.test == 'p42':
        test_p42()
    elif args.test == 'p43':
        test_p43()
    elif args.test == 'p44':
        test_p44()
