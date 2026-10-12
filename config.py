from model import *
import torch.nn as nn
from data import DataConfig
import argparse
import sys

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
    elif args.model == "EEGNet":
        model_config = EEGNetConfig(node_size=data_config.node_size,
                                    time_series_size=data_config.time_series_size,
                                    output_dim=data_config.output_dim,
                                    task_type=data_config.task_type,
                                    frequency=args.frequency,
                                    D=args.D,
                                    num_kernels=args.num_kernels,
                                    p1=args.p1,
                                    p2=args.p2,
                                    dropout=args.dropout)
        model = EEGNet(model_config)
    elif args.model == 'TimeLLM':
        model_config = TimeLLMConfig(
            node_size=data_config.node_size,
            d_model=args.d_model,
            n_heads=args.num_heads,
            d_ff=args.d_ff,
            num_prototypes=args.num_prototypes,
            num_gcn_layers=args.num_gcn_layers,
            dropout=args.dropout,
            num_windows=args.num_windows,
            task_type=data_config.task_type,
            output_dim=data_config.output_dim,
            dataset_name=args.dataset,
            llm_type=args.llm_type,
            llm_path=args.llm_path,
            llm_init=args.llm_init,
            use_task_prompt=args.use_task_prompt,
            use_stats_prompt=args.use_stats_prompt,
        )
        model = Model(model_config)
    elif args.model == 'TimeLLMPatch':
        model_config = TimeLLMPatchConfig(
            node_size=data_config.node_size,
            time_series_size=data_config.time_series_size,
            d_model=args.d_model,
            n_heads=args.num_heads,
            d_ff=args.d_ff,
            num_prototypes=args.num_prototypes,
            dropout=args.dropout,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            task_type=data_config.task_type,
            output_dim=data_config.output_dim,
            dataset_name=args.dataset,
            llm_type=args.llm_type,
            llm_path=args.llm_path,
            llm_init=args.llm_init,
            use_task_prompt=args.use_task_prompt,
        )
        model = TimeLLMPatchModel(model_config)
    elif args.model == 'TimeLLMKnowledge':
        model_config = TimeLLMKnowledgeConfig(
            node_size=data_config.node_size,
            time_series_size=data_config.time_series_size,
            hz=data_config.hz,
            num_local_segments=data_config.num_local_segments,
            d_model=args.d_model,
            n_heads=args.num_heads,
            num_prototypes=args.num_prototypes,
            dropout=args.dropout,
            task_type=data_config.task_type,
            output_dim=data_config.output_dim,
            dataset_name=args.dataset,
            llm_type=args.llm_type,
            llm_path=args.llm_path,
            llm_init=args.llm_init,
            use_task_prompt=args.use_task_prompt,
            readout_layers=args.readout_layers,
            use_local_knowledge=args.use_local_knowledge,
            use_knowledge_bypass=args.use_knowledge_bypass,
            freeze_head_bias=args.freeze_head_bias,
            learn_layer_weights=args.learn_layer_weights,
            use_semantic_prototypes=args.use_semantic_prototypes,
            proto_preselect=args.proto_preselect,
            proto_per_phrase=args.proto_per_phrase,
            use_proto_mixed_bank=args.use_proto_mixed_bank,
            proto_mixed_gate=args.proto_mixed_gate,
            use_patch_embeds=args.use_patch_embeds,
        )
        model = TimeLLMKnowledgeModel(model_config)
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


class TrackingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that records which options were explicitly passed on
    the command line (both ``--opt value`` and ``--opt=value`` forms)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_options = set()

    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        for token in args:
            if token.startswith('--'):
                self.explicit_options.add(token[2:].split('=', 1)[0])
        return super().parse_known_args(args, namespace)


def init_config():
    parser = TrackingArgumentParser()

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
    data_group.add_argument("--data_processors", default=0, type=int, help="")
    data_group.add_argument("--train_set", default=0.6, type=float, help="")
    data_group.add_argument("--val_set", default=0.2, type=float, help="")
    data_group.add_argument("--few_shot", default=0, type=int, help="few-shot: subjects per class (>0), -1=full-data finetune, 0=use all")
    data_group.add_argument("--few_shot_seed", default=42, type=int,
                             help="Random seed for few-shot subject sampling "
                                  "(different seeds select different subjects)")
    data_group.add_argument("--batch_size", default=64, type=int, help="")
    data_group.add_argument("--num_workers", default=5, type=int, help="")
    data_group.add_argument("--num_epochs", default=200, type=int, help="")
    data_group.add_argument("--drop_last", action="store_true",
                            help="Drop the last incomplete batch (off by default)")
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
    model_group.add_argument("--llm_init", default="pretrained", type=str,
                             choices=["pretrained", "random"],
                             help="LLM weight init for ablation: 'pretrained' (default) "
                                  "or 'random' (same arch, randomly-initialized weights, "
                                  "to test whether pretrained knowledge contributes)")
    model_group.add_argument("--patch_stride", default=5, type=int, help="Patch stride (TimeLLMPatch raw-EEG patching; legacy DFC subsampling for TimeLLM v1)")
    model_group.add_argument("--num_prototypes", default=500, type=int, help="Number of text prototypes")
    model_group.add_argument("--d_ff", default=128, type=int, help="LLM output truncation dim")
    model_group.add_argument("--num_gcn_layers", default=1, type=int, help="Number of GCN layers (TimeLLM)")
    model_group.add_argument("--use_task_prompt", action="store_true",
                             help="TimeLLM: include task description prompt")
    model_group.add_argument("--use_stats_prompt", action="store_true",
                             help="TimeLLM: include CAER-v3 statistics as a "
                                  "text prompt before the DFC patches")
    model_group.add_argument("--num_windows", default=10, type=int, help="Number of DFC time windows")
    model_group.add_argument("--patch_len", default=512, type=int, help="Patch length in raw samples (TimeLLMPatch)")
    model_group.add_argument("--readout_layers", default=4, type=int,
                             help="TimeLLMKnowledge: number of final LLM layers "
                                  "aggregated for READOUT position readout (K)")
    model_group.add_argument("--use_local_knowledge",
                             action=argparse.BooleanOptionalAction, default=True,
                             help="TimeLLMKnowledge: append 6x10s local knowledge blocks")
    model_group.add_argument("--use_knowledge_bypass",
                             action=argparse.BooleanOptionalAction, default=True,
                             help="TimeLLMKnowledge: mean-pool(q) residual head with gate lambda")
    model_group.add_argument("--freeze_head_bias",
                             action=argparse.BooleanOptionalAction, default=True,
                             help="TimeLLMKnowledge: freeze output-head bias at normalized label prior")
    model_group.add_argument("--learn_layer_weights",
                             action=argparse.BooleanOptionalAction, default=False,
                             help="TimeLLMKnowledge: learn anchored layer-aggregation weights (default: fixed uniform)")
    model_group.add_argument("--use_semantic_prototypes",
                             action=argparse.BooleanOptionalAction, default=True,
                             help="TimeLLMKnowledge: per-phrase vocab-neighborhood "
                                  "semantic prototype bank (cosine top-K + shared "
                                  "Linear) instead of learned full-vocab mapping_layer")
    model_group.add_argument("--proto_preselect", default=1000, type=int,
                             help="TimeLLMKnowledge: per-phrase top-K vocab "
                                  "candidates for semantic prototypes")
    model_group.add_argument("--proto_per_phrase", default=100, type=int,
                             help="TimeLLMKnowledge: prototypes per phrase after "
                                  "shared Linear compression")
    model_group.add_argument("--use_proto_mixed_bank",
                             action=argparse.BooleanOptionalAction, default=False,
                             help="TimeLLMKnowledge: add small shared learned "
                                  "prototype bank gated onto semantic bank output")
    model_group.add_argument("--proto_mixed_gate", default=0.1, type=float,
                             help="TimeLLMKnowledge: init gate for mixed learned "
                                  "prototype bank")
    model_group.add_argument("--use_patch_embeds",
                             action=argparse.BooleanOptionalAction, default=True,
                             help="TimeLLMKnowledge: per-group patch embeddings "
                                  "(FC graph 2D CNN, relative PSD 2D CNN) replace "
                                  "the shared PatchStem raw branch")

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


    option_dest = {opt.lstrip('-'): act.dest
                   for act in parser._actions for opt in act.option_strings}
    args = parser.parse_args()
    # CLI 值快照：只含命令行上显式出现的参数。日志时用当前值对比，
    # 以反映 DeepSpeed JSON 对 batch_size/learning_rate 的覆盖。
    args.explicit_args = {option_dest[opt]: getattr(args, option_dest[opt])
                          for opt in parser.explicit_options
                          if opt in option_dest}

    return args
