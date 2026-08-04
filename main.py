import os
import json as _json

# import wandb

from config import init_config
from trainers import *
from utils import *
from torch import distributed
import random
import gc


import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
logger = logging.getLogger(__name__)
# os.environ['WANDB_MODE'] = "offline"

def cleanup_memory():
    """清理显存和缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def set_seed(seed=42):
    """设置所有随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CuDNN以确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置SDPA和cuBLAS确定性（仅cudnn.deterministic不覆盖FlashAttention和bfloat16 matmul）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # 设置Python哈希种子（用于字典等）
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    # ── 判断运行模式 ──
    transfer_mode = bool(args.pretrain_path)
    is_zero_shot = transfer_mode and args.few_shot == 0

    if args.do_train:
        local_rank = 0
        if args.deepspeed:
            # DeepSpeed launcher sets --local_rank (arg) and LOCAL_RANK (env)
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            # 自动推导 DeepSpeed 配置文件路径
            ds_path = args.deepspeed_config
            if not os.path.exists(ds_path):
                auto_path = "deepspeed/train.json"
                if os.path.exists(auto_path):
                    args.deepspeed_config = auto_path
                else:
                    raise FileNotFoundError(
                        f"DeepSpeed config not found: {ds_path} or {auto_path}")
        elif args.do_parallel:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            distributed.init_process_group('nccl', world_size=world_size, rank=rank)
            torch.cuda.set_device(local_rank)

        # ── 迁移学习：只跑一次，用户手动多次运行收集结果 ──
        if transfer_mode:
            episode_seed = args.few_shot_seed
            pretrain_dir = os.path.basename(args.pretrain_path.rstrip('/'))
            src_name = (args.pretrain_path.rstrip('/').split('/')[-1]
                        if '/' in args.pretrain_path else pretrain_dir)

            if args.abla_channel >= 0:
                log_file = (f'{args.log_dir}/fewshot_{args.model}{args.append}'
                            f'_wo_C{args.abla_channel}_{src_name}pretrain'
                            f'_{args.few_shot}shot_{args.dataset}.log')
            elif args.abla_vae != "n":
                log_file = (f'{args.log_dir}/fewshot_{args.model}{args.append}'
                            f'_wo_{args.abla_vae}_{src_name}pretrain'
                            f'_{args.few_shot}shot_{args.dataset}.log')
            else:
                log_file = (f'{args.log_dir}/fewshot_{args.model}{args.append}'
                            f'_{src_name}pretrain_{args.few_shot}shot'
                            f'_{args.dataset}.log')
            init_logger(log_file)

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info(f"{'#'*10} few-shot seed={episode_seed} {'#'*10}")

            if is_zero_shot:
                # Zero-shot: 不加 few-shot 采样，直接加载预训练模型评估
                trainer = eval(args.model + 'Trainer')(
                    args, local_rank=local_rank, task_id=0)
                trainer.load_model(path=args.pretrain_path)
                trainer.model.eval()
                result = trainer.evaluate(dataloader_key='test')
            else:
                # K-shot: 按 few_shot_seed 采样被试，fine-tune 后评估
                if args.deepspeed:
                    ds_path = args.deepspeed_config
                    finetune_ds_path = "deepspeed/finetune.json"
                        if os.path.exists(finetune_ds_path):
                            args.deepspeed_config = finetune_ds_path
                            with open(finetune_ds_path) as _f:
                                _ds_cfg = _json.load(_f)
                            args.learning_rate = _ds_cfg['optimizer']['params']['lr']
                        else:
                            logger.warning(
                                f"Finetune DeepSpeed config not found: "
                                f"{finetune_ds_path}, using original: {ds_path}")
                else:
                    args.learning_rate = 1e-5
                trainer = eval(args.model + 'Trainer')(
                    args, local_rank=local_rank, task_id=0,
                    episode_seed=episode_seed)
                trainer.load_model(path=args.pretrain_path)
                result = trainer.finetune()

            # ── 单次结果输出 ──
            is_rank_0 = (not torch.distributed.is_initialized()
                         or torch.distributed.get_rank() == 0)
            if is_rank_0 and result is not None:
                mode_label = (f"Zero-shot ({src_name} → {args.dataset})"
                              if is_zero_shot else
                              f"{args.few_shot}-shot ({src_name} → {args.dataset})")
                header = f"\n{'='*60}\n  {mode_label}  seed={episode_seed}\n{'='*60}"
                print(header)
                logger.info(header)
                for k, v in result.items():
                    if v is not None:
                        if isinstance(v, (int, float, np.floating, np.integer)):
                            line = f"  {k:15s}: {v:.4f}"
                        else:
                            line = f"  {k:15s}: {v}"
                        print(line)
                        logger.info(line)
                footer = f"{'='*60}\n"
                print(footer)

            del trainer
            cleanup_memory()
        else:
            # ── 原有正常训练逻辑，完全不变 ──
            all_episode_results = []
            for i in range(args.num_repeat):
                group_name = f"{args.model}" \
                             f"_{args.dataset}" \
                             f"_{args.batch_size}" \
                             f"{f'sparsity-{args.sparsity}' if 'DFaST' in args.model else ''}" \
                             f'F{args.frequency}D{args.D}F{args.num_kernels}P{args.p1}={args.p2}_dp{args.dropout}' \

                             f"-cross"

                # run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True, group=f"{group_name}", tags=[args.dataset])

                trainer = eval(args.model + 'Trainer')(args, local_rank=local_rank, task_id=i)
                if args.abla_channel >= 0:
                    init_logger(f'{args.log_dir}/train_{args.model}{args.append}_wo_C{args.abla_channel}_{args.dataset}.log')
                elif args.abla_vae != "n":
                    init_logger(f'{args.log_dir}/train_{args.model}{args.append}_wo_{args.abla_vae}_{args.dataset}.log')
                else:
                    init_logger(f'{args.log_dir}/train_{args.model}{args.append}_{args.dataset}.log')
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logger.info(f"{'#'*10} Repeat:{i} {'#'*10}")
                trainer.train()
                all_episode_results.append(trainer.best_result)

                # run.finish()

                del trainer
                cleanup_memory()

            # ── 正常训练结果汇总 ──
            is_rank_0 = (not torch.distributed.is_initialized()
                         or torch.distributed.get_rank() == 0)
            if is_rank_0 and all_episode_results:
                valid_results = [r for r in all_episode_results if r is not None]
                if valid_results:
                    results = Recorder()
                    for r in valid_results:
                        results.add_record(r)
                    results.save(os.path.join(args.model_dir, args.model,
                                              'results.json'))
    elif args.do_test:
        if transfer_mode and is_zero_shot:
            # ── Zero-shot 纯评估（无训练）──
            trainer = eval(args.model + 'Trainer')(args)
            init_logger(f'{args.log_dir}/test_{args.model}{args.append}'
                        f'_zero-shot_{args.dataset}.log')
            trainer.load_model(path=args.pretrain_path)
            trainer.model.eval()
            trainer.evaluate()
        else:
            trainer = eval(args.model + 'Trainer')(args)
            init_logger(f'{args.log_dir}/test_{args.model}{args.append}'
                        f'_{args.dataset}.log')
            trainer.load_model()
            trainer.evaluate()


def parameters(args):
    trainer = eval(args.model + 'Trainer')(args)
    total = sum([param.nelement() for param in trainer.model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))


if __name__ == '__main__':
    set_seed(42)
    Args = init_config()
    main(Args)