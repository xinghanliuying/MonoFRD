import argparse  # 导入命令行参数解析库
import glob  # 导入文件路径模式匹配库
import os  # 导入操作系统交互库
from pathlib import Path  # 导入路径处理库
from test import repeat_eval_ckpt  # 从test模块导入repeat_eval_ckpt函数

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式计算模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from tensorboardX import SummaryWriter  # 导入TensorBoard可视化库

# 导入自定义模块和函数
from monofrd.config import cfg, cfg_from_list, cfg_from_yaml_file, update_cfg_by_args, log_config_to_file
from monofrd.datasets import build_dataloader
from monofrd.models import build_network, model_fn_decorator
from monofrd.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

torch.backends.cudnn.benchmark = True  # 启用cudnn的自动优化
#####  --ckpt ./ckpt/checkpoint.pth




def parse_config():  # 定义解析配置的函数
    parser = argparse.ArgumentParser(description='arg parser')  # 创建解析器对象
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')  # 添加配置文件参数

    # 基础训练选项
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')  # 添加批大小参数
    parser.add_argument('--epochs', type=int, default=60, required=False, help='number of epochs to train for')  # 添加训练轮数参数
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')  # 添加数据加载器的工作线程数参数
    parser.add_argument('--exp_name', type=str, default='default', help='extra tag for this experiment')  # 添加实验名称参数
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')  # 添加固定随机种子选项
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')  # 添加检查点保存间隔参数
    parser.add_argument('--max_ckpt_save_num', type=int, default=20, help='max number of saved checkpoint')  # 添加最大检查点保存数量参数
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')  # 添加合并所有迭代到一个epoch的选项
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')  # 添加最大等待时间参数
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')  # 添加保存到文件选项

    # 加载选项
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')  # 添加从哪个检查点开始的参数
    # parser.add_argument('--ckpt', type=str, default='./ckpt/checkpoint_epoch_20.pth', help='checkpoint to start from')  # 添加从哪个检查点开始的参数
    parser.add_argument('--start_epoch', type=int, default=1, help='')  # 添加开始轮数参数
    parser.add_argument('--continue_train', action='store_true', default=False)  # 添加继续训练选项

    # 分布式选项
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')  # 添加启动器类型参数
    parser.add_argument('--tcp_port', type=int, default=1110, help='tcp port for distrbuted training')  # 添加TCP端口参数 13093317310
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')  # 添加是否使用同步批归一化选项
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, help='whether to find find_unused_parameters')  # 添加是否查找未使用的参数选项
    parser.add_argument('--local_rank', type=int, default=1, help='local rank for distributed training')  # 添加本地排名参数

    # 配置选项
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')  # 添加额外配置键参数
    parser.add_argument('--trainval', action='store_true', default=False, help='')  # 添加训练验证选项
    parser.add_argument('--imitation', type=str, default="2d")  # 添加模仿类型参数

    args = parser.parse_args()  # 解析命令行参数

    print(f'args local rank{args.local_rank}')  # 打印本地排名信息
    cfg_from_yaml_file(args.cfg_file, cfg)  # 从YAML文件加载配置到cfg变量
    update_cfg_by_args(cfg, args)  # 使用命令行参数更新配置
    cfg.TAG = Path(args.cfg_file).stem  # 设置配置标签为配置文件名（不含扩展名）
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])  # 设置实验组路径

    if args.set_cfgs is not None:  # 如果有额外的配置设置
        cfg_from_list(args.set_cfgs, cfg)  # 从列表更新配置

    return args, cfg  # 返回解析的参数和配置


def main():  # 定义主函数
    args, cfg = parse_config()  # 调用解析配置函数，获取命令行参数和配置信息

    # 判断是否使用分布式训练
    if args.launcher == 'none':  # 如果启动器参数为'none'
        dist_train = False  # 不使用分布式训练
        total_gpus = 1  # 总GPU数量设置为1
    else:  # 如果启动器参数不为'none'
        print(f'args local rank{args.local_rank}')  # 打印本地排名（用于分布式训练）
        # 初始化分布式环境，并获取总GPU数量和本地排名
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True  # 设置为使用分布式训练
        print(f'total_gpus{total_gpus}')  # 打印总GPU数量
        print(f'LOCAL_RANK{cfg.LOCAL_RANK}')  # 打印本地排名

    # 设置批处理大小
    if args.batch_size is None:  # 如果批处理大小未设置
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU  # 使用配置文件中的默认值
    else:  # 如果批处理大小已设置
        # 确保批大小能被总GPU数量整除
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus  # 更新批处理大小

    # 设置训练周期数
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    # 设置随机种子
    if args.fix_random_seed:  # 如果需要固定随机种子
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)  # 设置随机种子

    # 设置输出目录和检查点目录
    output_dir = cfg.ROOT_DIR / 'outputs' / cfg.EXP_GROUP_PATH / (cfg.TAG + '.' + args.exp_name)
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # 创建检查点目录

    # 创建一个日志文件
    log_file = output_dir / 'log_train.txt'
    # 使用common_utils库创建一个日志器
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 将信息记录到日志文件
    logger.info('**********************Start logging**********************')
    # 获取环境变量中的CUDA_VISIBLE_DEVICES，如果没有则默认为'ALL'
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    # 记录CUDA_VISIBLE_DEVICES信息
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # 如果是分布式训练
    if dist_train:
        # 记录总批量大小
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # 记录命令行参数
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    # 将配置信息记录到文件
    log_config_to_file(cfg, logger=logger)
    # 如果是主节点
    if cfg.LOCAL_RANK == 0:
        # 记录git差异和日志
        os.system('git diff > %s/%s' % (output_dir, 'git.diff'))
        os.system('git log > %s/%s' % (output_dir, 'git.log'))
        # 复制配置文件到输出目录
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # 创建TensorBoard日志
    # tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------创建数据加载器、网络和优化器---------------------------
    # 使用build_dataloader函数创建训练数据集、数据加载器和数据采样器
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,  # 数据集配置
        class_names=cfg.CLASS_NAMES,  # 类别名称
        batch_size=args.batch_size,  # 批量大小
        dist=dist_train,  # 是否进行分布式训练
        workers=args.workers,  # 工作进程数
        logger=logger,  # 日志器
        training=True,  # 是否为训练模式
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,  # 是否合并所有迭代到一个epoch
        total_epochs=args.epochs  # 总的训练轮数
    )

    # 使用build_network函数创建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=train_set)

    # 如果启用同步批量归一化，则进行转换
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 将模型移动到GPU上
    model.cuda()

    # 使用build_optimizer函数创建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 如果可能，加载检查点
    start_epoch = it = 0  # 初始化起始轮数和迭代次数
    last_epoch = -1  # 初始化最后一轮的轮数
    # 如果提供了检查点文件
    if args.ckpt is not None:
        # 从检查点文件中加载模型和优化器状态
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1  # 设置最后一轮的轮数
    # 如果设置了继续训练
    elif args.continue_train:
        # 查找所有检查点文件
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            # 对检查点文件进行排序
            ckpt_list.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))
            print("using ckpt", ckpt_list[-1])
            # 加载最新的检查点文件
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1  # 设置最后一轮的轮数
        else:
            raise FileNotFoundError("no ckpt files found")

    # 设置模型为训练模式
    model.train()

    # 如果是分布式训练，使用DistributedDataParallel包装模型
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
            find_unused_parameters=args.find_unused_parameters)

    # 记录模型信息
    logger.info(model)

    # 创建学习率调度器和热身调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader) // (args.epochs if args.merge_all_iters_to_one_epoch else 1),
        total_epochs=args.epochs,
        last_epoch=last_epoch,
        optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------开始训练---------------------------
    # 在日志中记录开始训练的信息
    logger.info('*******Start training: {} ********'.format(output_dir))

    # 调用train_model函数进行模型训练
    train_model(
        model,  # 模型
        optimizer,  # 优化器
        train_loader,  # 训练数据加载器
        model_func=model_fn_decorator(),  # 模型函数装饰器
        lr_scheduler=lr_scheduler,  # 学习率调度器
        optim_cfg=cfg.OPTIMIZATION,  # 优化配置
        start_epoch=start_epoch,  # 开始的轮数
        total_epochs=args.epochs,  # 总轮数
        start_iter=it,  # 开始的迭代次数
        rank=cfg.LOCAL_RANK,  # 本地排名（用于分布式训练）
        tb_log=tb_log,  # TensorBoard日志
        ckpt_save_dir=ckpt_dir,  # 检查点保存目录
        train_sampler=train_sampler,  # 训练数据采样器
        lr_warmup_scheduler=lr_warmup_scheduler,  # 学习率热身调度器
        ckpt_save_interval=args.ckpt_save_interval,  # 检查点保存间隔
        max_ckpt_save_num=args.max_ckpt_save_num,  # 最大检查点保存数量
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,  # 是否合并所有迭代到一个epoch
        dist_train=dist_train,  # 是否进行分布式训练
        logger=logger  # 日志器
    )

    # 在日志中记录结束训练的信息
    logger.info('*******End training: {} ********'.format(output_dir))

    # 在日志中记录开始评估的信息
    logger.info('*******Start evaluation: {} ********'.format(output_dir))

    # 使用build_dataloader函数创建测试数据集和数据加载器
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,  # 数据集配置
        class_names=cfg.CLASS_NAMES,  # 类别名称
        batch_size=args.batch_size,  # 批量大小
        dist=dist_train,  # 是否进行分布式训练
        workers=args.workers,  # 工作进程数
        logger=logger,  # 日志器
        training=False  # 非训练模式
    )

    # 创建评估输出目录
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 只评估最后10个轮次
    args.start_epoch = max(cfg.OPTIMIZATION.DECAY_STEP_LIST[-1], args.epochs - 10) if len(
        cfg.OPTIMIZATION.DECAY_STEP_LIST) > 0 else cfg.OPTIMIZATION.NUM_EPOCHS - 10

    # 调用repeat_eval_ckpt函数进行模型评估
    repeat_eval_ckpt(
        model.module if dist_train else model,  # 模型（如果是分布式训练，则使用model.module）
        test_loader,  # 测试数据加载器
        args,  # 命令行参数
        eval_output_dir,  # 评估输出目录
        logger,  # 日志器
        ckpt_dir,  # 检查点目录
        dist_test=dist_train  # 是否进行分布式测试
    )

    # 在日志中记录结束评估的信息
    logger.info('*******End evaluation: {} ********'.format(output_dir))

# 主函数入口
if __name__ == '__main__':
    main()
