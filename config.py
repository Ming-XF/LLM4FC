from model import *
import torch.nn as nn
from data import DataConfig
import argparse

import pdb


def init_model_config(args, data_config: DataConfig):
    if args.model == "BNT":
        model_config = BNTConfig(node_size=data_config.node_size,
                                 sizes=(data_config.node_size, data_config.node_size // 2),
                                 pooling=(False, True),
                                 pos_encoding=None,  # identity, none
                                 orthogonal=True,
                                 # freeze_center=True,
                                 freeze_center=False,
                                 project_assignment=True,
                                 num_heads=args.num_heads,
                                 pos_embed_dim=data_config.node_size,
                                 dim_feedforward=1024,
                                 task_type=data_config.task_type,
                                 output_dim=data_config.output_dim,
                                 )
        model = BNT(model_config)
    elif args.model == 'BrainNetCNN':
        model_config = BrainNetCNNConfig(node_size=data_config.node_size,
                                         task_type=data_config.task_type,
                                         output_dim=data_config.output_dim)
        model = BrainNetCNN(model_config)
    elif args.model == "ALTER":
        model_config = ALTERConfig(node_size=data_config.node_size,
                                 # sizes=(data_config.node_size, data_config.node_size // 2),
                                 sizes=(360, 100),
                                 pooling=(False, True),
                                 pos_encoding="rrwp",  # identity, none
                                 orthogonal=True,
                                 freeze_center=False,
                                 project_assignment=True,
                                 num_heads=args.num_heads,
                                 # pos_embed_dim=data_config.node_size,
                                 pos_embed_dim=32,
                                 dim_feedforward=1024,
                                 task_type=data_config.task_type,
                                 output_dim=data_config.output_dim,
                                 )
        model = ALTER(model_config)
    elif args.model == 'GCDGCN':
        model_config = GCDGCNConfig(node_size=data_config.node_size,
                                         task_type=data_config.task_type,
                                         output_dim=data_config.output_dim)
        model = GCDGCN(model_config)
    elif args.model == 'TimeLLM':
        model_config = TimeLLMConfig(
            node_size=data_config.node_size,
            d_model=args.d_model,
            n_heads=args.num_heads,
            d_ff=args.d_ff,
            num_prototypes=args.num_prototypes,
            gcn_hidden=args.gcn_hidden,
            num_gcn_layers=args.num_gcn_layers,
            dropout=args.dropout,
            num_windows=args.num_windows,
            task_type=data_config.task_type,
            output_dim=data_config.output_dim,
            dataset_name=args.dataset,
            llm_type=args.llm_type,
            llm_path=args.llm_path,
        )
        model = Model(model_config)
    else:
        model = None
        model_config = None
    if model is not None:
        init_parameters(model, model_config)
    return model, model_config


def init_parameters(model, model_config):
    if model_config.initializer is not None:
        for p in model.parameters():
            if p.dim() > 1:
                if model.config.initializer == 'xavier':
                    nn.init.xavier_uniform_(p)
                elif model.config.initializer == 'orthogonal':
                    nn.init.orthogonal_(p)


def init_config():
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--project", default="", type=str, help="")
    global_group.add_argument("--wandb_entity", default='cwg', type=str, help="")
    global_group.add_argument("--log_dir", default="./log_dir", type=str, help="")
    global_group.add_argument("--model", default="TimeLLM", type=str, help="")
    global_group.add_argument("--num_repeat", default=1, type=int, help="")
    global_group.add_argument("--visualize", action="store_true", help="")
    global_group.add_argument("--append", default="", type=str, help="")

    data_group = parser.add_argument_group(title="data", description="")
    data_group.add_argument("--dataset", default='DiseaseBeirut', type=str, help="")
    data_group.add_argument("--data_dir", default="", type=str, help="")
    data_group.add_argument("--data2_dir", default="", type=str, help="")
    data_group.add_argument("--data_processors", default=0, type=int, help="")
    data_group.add_argument("--train_set", default=0.6, type=float, help="")
    data_group.add_argument("--val_set", default=0.2, type=float, help="")
    data_group.add_argument("--few_shot", default=0, type=int, help="few-shot: number of subjects per class (0 = use all)")
    data_group.add_argument("--few_shot_seed", default=42, type=int,
                             help="Random seed for few-shot subject sampling "
                                  "(different seeds select different subjects)")
    data_group.add_argument("--batch_size", default=64, type=int, help="")
    data_group.add_argument("--num_workers", default=5, type=int, help="")
    data_group.add_argument("--num_epochs", default=200, type=int, help="")
    data_group.add_argument("--drop_last", default=True, type=bool, help="")
    data_group.add_argument("--frequency", default=500, type=int, help="")
    data_group.add_argument("--D", default=2, type=int, help="")
    data_group.add_argument("--F1", default=8, type=int, help="")
    data_group.add_argument("--p1", default=4, type=int, help="")
    data_group.add_argument("--p2", default=8, type=int, help="")

    preprocess_group = parser.add_argument_group(title="preprocess", description="")

    preprocess_group.add_argument("--fc_threshold", default=0.0, type=float,
                                  help="FC sparsification: edges with |value| < threshold are set to 0")
    preprocess_group.add_argument("--fc_keep_ratio", default=1.0, type=float,
                                  help="FC sparsification: top-K ratio of edges to keep by absolute value (1.0 = keep all)")

    model_group = parser.add_argument_group(title="model", description="")
    model_group.add_argument("--d_model", default=4, type=int, help="")
    model_group.add_argument("--k", default=5, type=int, help="")
    model_group.add_argument("--num_kernels", default=5, type=int, help="")
    model_group.add_argument("--sparsity", default=0.7, type=float, help="")
    model_group.add_argument("--dynamic_stride", default=1, type=int, help="")
    model_group.add_argument("--hidden_dim", default=1024, type=int, help="")
    model_group.add_argument("--num_heads", default=1, type=int, help="")
    model_group.add_argument("--abla_channel", default=-1, type=int, help="Channel Ablation EXP")
    model_group.add_argument("--abla_vae", default="n", type=str, help="Vae Ablation EXP")
    model_group.add_argument("--num_layers", default=2, type=int, help="")
    model_group.add_argument("--num_node_temporal_layers", default=1, type=int, help="")
    model_group.add_argument("--num_graph_temporal_layers", default=2, type=int, help="")
    model_group.add_argument("--attention_depth", default=2, type=int, help="")
    model_group.add_argument("--activation", default="gelu", type=str, help="")
    model_group.add_argument("--model_dir", default="output_dir", type=str, help="")
    model_group.add_argument("--dropout", default=0.5, type=float, help="")
    model_group.add_argument("--distill", action="store_true", help="")
    model_group.add_argument("--initializer", default=None, type=str, help="")
    model_group.add_argument("--integration", default=None, type=str, help="")
    model_group.add_argument("--llm_type", default="chatglm", type=str,
                             choices=["chatglm", "llama"],
                             help="LLM backbone: chatglm (default) or llama")
    model_group.add_argument("--llm_path", default="./model/chatglm-6b", type=str,
                             help="Path to pretrained LLM directory")
    model_group.add_argument("--patch_stride", default=5, type=int, help="DFC temporal subsampling stride (TimeLLM v1 only)")
    model_group.add_argument("--num_prototypes", default=500, type=int, help="Number of text prototypes")
    model_group.add_argument("--d_ff", default=128, type=int, help="LLM output truncation dim")
    model_group.add_argument("--gcn_hidden", default=128, type=int, help="GCN hidden dimension (TimeLLM v2)")
    model_group.add_argument("--num_gcn_layers", default=1, type=int, help="Number of GCN layers (TimeLLM)")
    model_group.add_argument("--num_windows", default=10, type=int, help="Number of DFC time windows")

    train_group = parser.add_argument_group(title="train", description="")
    train_group.add_argument("--max_steps", default=-1, type=int, help="Limit training steps per epoch (debug only, -1 = full)")
    train_group.add_argument("--pretrain_path", default="", type=str,
                             help="Pretrained checkpoint dir for transfer-learning "
                                  "few-shot / zero-shot")
    train_group.add_argument("--do_train", action="store_true", help="")
    train_group.add_argument("--do_parallel", action="store_true", help="")
    train_group.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed ZeRO-2/3")
    train_group.add_argument("--deepspeed_config", default="ds_config_zero3.json", type=str, help="DeepSpeed config JSON path")
    train_group.add_argument("--local_rank", default=0, type=int, help="Local rank (set by DeepSpeed launcher)")
    train_group.add_argument("--device", default="cuda", type=str, help="")
    train_group.add_argument("--save_steps", default=200, type=int, help="")
    train_group.add_argument("--epsilon_ls", default=0, type=float, help=" label_smoothing")
    train_group.add_argument("--alpha", default=1., type=float, help="")
    train_group.add_argument("--beta", default=1., type=float, help="")
    train_group.add_argument("--early_stop_patience", default=20, type=int,
                             help="Early stopping patience (epochs, 0 = disabled)")
    train_group.add_argument("--early_stop_min_delta", default=0.001, type=float,
                             help="Minimum improvement to reset patience")
    train_group.add_argument("--early_stop_metric", default="Loss", type=str,
                             help="Metric to monitor for early stopping (Accuracy / AUC / Loss)")

    # ── Train/Val/Test split mode (used when --num_repeat == 1) ──
    train_group.add_argument("--standard_split", action="store_true",
                             help="Force train/val/test split even when num_repeat > 1")

    optimizer_group = parser.add_argument_group(title="optimizer", description="")
    optimizer_group.add_argument("--optimizer", default='Adam', type=str, help="")
    optimizer_group.add_argument("--learning_rate", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--target_learning_rate", default=1e-5, type=float, help="")
    optimizer_group.add_argument("--max_learning_rate", default=0.001, type=float, help="")
    optimizer_group.add_argument("--beta1", default=0.9, type=float, help="")
    optimizer_group.add_argument("--beta2", default=0.98, type=float, help="")
    optimizer_group.add_argument("--epsilon", default=1e-9, type=float, help="")
    optimizer_group.add_argument("--schedule", default='cos', type=str, help="")
    optimizer_group.add_argument("--warmup_steps", default=400, type=int, help="")
    optimizer_group.add_argument("--weight_decay", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--eps", default=1e-8, type=float, help="")
    optimizer_group.add_argument("--no_weight_decay", action="store_true", help="")
    optimizer_group.add_argument("--match_rule", default=None, type=str, help="")
    optimizer_group.add_argument("--except_rule", default=None, type=str, help="")

    evaluate_group = parser.add_argument_group(title="evaluate", description="")
    evaluate_group.add_argument("--do_evaluate", action="store_true", help="")
    evaluate_group.add_argument("--do_test", action="store_true", help="")


    return parser.parse_args()
