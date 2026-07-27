import os

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
    
    # 设置Python哈希种子（用于字典等）
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    if args.do_train:
        results = Recorder()
        local_rank = 0
        if args.deepspeed:
            # DeepSpeed launcher sets --local_rank (arg) and LOCAL_RANK (env)
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
        elif args.do_parallel:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            distributed.init_process_group('nccl', world_size=world_size, rank=rank)
            torch.cuda.set_device(local_rank)
        for i in range(args.num_repeat):
            group_name = f"{args.model}" \
                         f"_{args.dataset}" \
                         f"_{args.batch_size}" \
                         f"{f'sparsity-{args.sparsity}' if 'DFaST' in args.model else ''}" \
                         f'F{args.frequency}D{args.D}F{args.num_kernels}P{args.p1}={args.p2}_dp{args.dropout}' \
                         f"_w{args.window_size}" \
                         f"{'_mp' if args.mix_up else ''}" \
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
            results.add_record(trainer.best_result)

            # run.finish()

            del trainer
            cleanup_memory()
        # results.save(os.path.join(args.model_dir, args.model, 'results.json'))
    elif args.do_test:
        trainer = eval(args.model + 'Trainer')(args)
        init_logger(f'{args.log_dir}/test_{args.model}{args.append}_{args.dataset}.log')
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