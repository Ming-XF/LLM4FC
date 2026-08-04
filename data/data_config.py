class DataConfig:
    # ── 任务类型常量 ──
    TASK_CLASSIFICATION = "classification"
    TASK_REGRESSION = "regression"
    TASK_MULTI_OUTPUT_REGRESSION = "multi_output_regression"

    def __init__(self, args, time_series_size=0, node_size=0, node_feature_size=0, output_dim=2,
                 task_type=None):
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.data2_dir = args.data2_dir
        self.train_set = args.train_set
        self.few_shot = args.few_shot
        self.few_shot_seed = args.few_shot_seed
        self.pretrain_path = args.pretrain_path
        self.val_set = args.val_set
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epochs = args.num_epochs
        self.drop_last = args.drop_last
        self.alpha = args.alpha
        self.beta = args.beta
        self.n_splits = args.num_repeat
        self.time_series_size = time_series_size
        self.node_size = node_size
        self.node_feature_size = node_feature_size
        self.num_heads = args.num_heads
        self.abla_channel = args.abla_channel
        self.abla_vae = args.abla_vae
        self.fc_threshold = args.fc_threshold
        self.fc_keep_ratio = args.fc_keep_ratio

        # 任务类型：由各数据集 load_data 显式设置；
        # 未设置时根据 output_dim 推断（向后兼容旧数据集）
        if task_type is not None:
            self.task_type = task_type
        elif output_dim == 1:
            self.task_type = self.TASK_REGRESSION
        elif output_dim >= 2:
            self.task_type = self.TASK_CLASSIFICATION
        else:
            self.task_type = self.TASK_CLASSIFICATION

        # 模型输出维度：分类=类别数，回归=1，多值回归=目标展平维度
        # 由各数据集 load_data 设置
        self.output_dim = output_dim

        # 类别权重仅对分类任务有意义
        if self.task_type == self.TASK_CLASSIFICATION:
            self.class_weight = [1] * output_dim
        else:
            self.class_weight = None

    @property
    def is_classification(self):
        return self.task_type == self.TASK_CLASSIFICATION

    @property
    def is_regression(self):
        return self.task_type == self.TASK_REGRESSION

    @property
    def is_multi_output_regression(self):
        return self.task_type == self.TASK_MULTI_OUTPUT_REGRESSION


