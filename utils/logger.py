import logging
import os
import sys


class Logger(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if config.log.log_level == "debug":
            logging_level = logging.DEBUG
        elif config.log.log_level == "info":
            logging_level = logging.INFO
        elif config.log.log_level == "warn":
            logging_level = logging.WARN
        elif config.log.log_level == "error":
            logging_level = logging.ERROR
        else:
            raise TypeError(
                "No logging type named %s, candidate is: info, debug, error")
        logging.basicConfig(filename=config.log.logger_file,
                            level=logging_level,
                            format='%(asctime)s : %(levelname)s  %(message)s',
                            filemode="a", datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def debug(msg):
        """Log debug message
            msg: Message to log
        """
        logging.debug(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def info(msg):
        """"Log info message
            msg: Message to log
        """
        logging.info(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def warn(msg):
        """Log warn message
            msg: Message to log
        """
        logging.warning(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def error(msg):
        """Log error message
            msg: Message to log
        """
        logging.error(msg)
        sys.stderr.write(msg + "\n")


def init_logger(file_name='main.log'):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=file_name)


def log_args(args):
    """实验开始前，将运行命令中的参数与 DeepSpeed 实际生效值写入日志。

    只记录命令行显式出现的参数（即 sh 脚本传入的参数），config.py 的
    默认参数不重复记录；另附一组由 DeepSpeed 配置覆盖后的实际生效值
    （batch_size / learning_rate）。必须在 init_logger 之后调用。
    """
    logger = logging.getLogger(__name__)
    logger.info("[ARGS] command: %s", " ".join(sys.argv))

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith('--'):
            i += 1
            continue
        key = token[2:]
        if '=' in key:                       # --key=value 形式
            key, value = key.split('=', 1)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            value = argv[i + 1]
            i += 2
        else:                                # store_true 类参数
            value = True
            i += 1
        logger.info("[ARGS]   %s = %s", key, value)

    # DeepSpeed 配置覆盖后的实际生效值（仅在 --do_train + --deepspeed 下）
    if getattr(args, 'do_train', False) and getattr(args, 'deepspeed', False):
        ds_cfg = getattr(args, 'deepspeed_config', '')
        logger.info("[ARGS-effective] batch_size = %s (from %s)",
                    getattr(args, 'batch_size', None), ds_cfg)
        logger.info("[ARGS-effective] learning_rate = %s (from %s)",
                    getattr(args, 'learning_rate', None), ds_cfg)
        logger.info("[ARGS-effective] deepspeed_config = %s", ds_cfg)
