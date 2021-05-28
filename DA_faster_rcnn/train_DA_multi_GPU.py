# --------------------------------------------------------
# 一、训练过程：
# 1、运行命令：python -m torch.distributed.launch --nproc_per_node=4 --use_env train_DA_multi_GPU.py
# 2、代码验证环节：开启 --source-val，即在训练过后，同时在t_test和s_val上均验证，并写入result.txt及绘制对应的map曲线图
# 3、pth模型保存位置：‘./save/model/DA/数据集名称_backbone/model_num.pth’
# 4、图片结果位置：‘./save/outputs/DA/’存放：
#     1）运行过程产生的三个文件：loss_and_lr.png：训练过程的损失及学习率变化情况
#                             mAP.png：训练过程中，验证集的map
#                             results20210526-110237.txt：各个loss
# --------------------------------------------------------
# 二、代码修改记录
# 20200527： 1）添加：加入DA_img、DA_instance、consistency功能，修改DA_faster_rcnn_framework.py等其他文件，已测试
#            2）测试：代码在voc2007+voc2012-->clipart 训练通过
# 20200528： 1）添加：results20210526-110237.txt中添加各个类别的map值，待测试
# --------------------------------------------------------

import time
import os
import datetime

import torch
import transforms
from my_dataset import VOC2012DataSet
from backbone import resnet50_fpn_backbone
from network_files.DA_faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
import train_utils.train_DA_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir
from validation import summarize

def create_model(backbone, num_classes, device):
    if backbone == resnet50_fpn_backbone:
        # 如果显存很小，建议使用默认的FrozenBatchNorm2d
        # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                         trainable_layers=3)
        # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
        model = FasterRCNN(backbone=backbone, num_classes=91)
        # 载入预训练模型权重
        # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


def main(args):
    init_distributed_mode(args)
    print(args)

    # 训练文件的根目录(VOCdevkit)
    mode = args.domain_datasets
    if mode == "V2C":
        domain_src = "../datasets/pascal_voc"  # voc2007+voc2012
        domain_tar = "../datasets/clipart"
    elif mode == "S2C":
        domain_src = "../datasets/sim10k"
        domain_tar = "../datasets/cityscape_car"
    elif mode == "C2F":
        domain_src = "../datasets/cityscape"
        domain_tar = "../datasets/foggy_cityscape"

    device = torch.device(args.device)

    # 用来保存coco_info的文件
    if args.source_val:
        s_results_file = os.path.join(args.record_dir, args.backbone, "{}_s_results{}.txt"
                                    .format(mode, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    t_results_file = os.path.join(args.record_dir, args.backbone, "{}_t_results{}.txt"
                                  .format(mode, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Data loading code
    print("Loading data")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # check voc root
    if os.path.exists(os.path.join(domain_src, "VOCdevkit")) is False:
        raise FileNotFoundError("Source VOCdevkit dose not in path:'{}'.".format(domain_src))
    if os.path.exists(os.path.join(domain_tar, "VOCdevkit")) is False:
        raise FileNotFoundError("target VOCdevkit dose not in path:'{}'.".format(domain_tar))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt    mode用来找对应的json文件
    src_train_dataset = VOC2012DataSet(domain_src, mode, data_transform["train"], "trainval.txt")
    src_val_dataset = VOC2012DataSet(domain_src, mode, data_transform["train"], "val.txt")
    tar_train_dataset = VOC2012DataSet(domain_tar, mode, data_transform["val"], "train.txt")
    tar_test_dataset = VOC2012DataSet(domain_tar, mode, data_transform["val"], "test.txt")
    print("[source_doamin] train images: {}, val images: {}".format(len(src_train_dataset),
                                                                    len(src_val_dataset)))
    print("[targer_domain] train images: {}, val images: {}".format(len(tar_train_dataset),
                                                                    len(tar_test_dataset)))
    print("Creating data loaders")
    if args.distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_dataset)
        src_test_sampler = torch.utils.data.distributed.DistributedSampler(src_val_dataset)
        tar_train_sampler = torch.utils.data.distributed.DistributedSampler(tar_train_dataset)
        tar_test_sampler = torch.utils.data.distributed.DistributedSampler(tar_test_dataset)
    else:
        src_train_sampler = torch.utils.data.RandomSampler(src_train_dataset)
        src_test_sampler = torch.utils.data.SequentialSampler(src_val_dataset)
        tar_train_sampler = torch.utils.data.RandomSampler(tar_train_dataset)
        tar_test_sampler = torch.utils.data.SequentialSampler(tar_test_dataset)

    if args.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        src_group_ids = create_aspect_ratio_groups(src_train_dataset, k=args.aspect_ratio_group_factor)
        src_train_batch_sampler = GroupedBatchSampler(src_train_sampler, src_group_ids, args.batch_size)
        tar_group_ids = create_aspect_ratio_groups(tar_train_dataset, k=args.aspect_ratio_group_factor)
        tar_train_batch_sampler = GroupedBatchSampler(tar_train_sampler, tar_group_ids, args.batch_size)
    else:
        src_train_batch_sampler = torch.utils.data.BatchSampler(src_train_sampler, args.batch_size, drop_last=True)
        tar_train_batch_sampler = torch.utils.data.BatchSampler(tar_train_sampler, args.batch_size, drop_last=True)

    src_loader = torch.utils.data.DataLoader(src_train_dataset, batch_sampler=src_train_batch_sampler,
                                             num_workers=args.workers, collate_fn=src_train_dataset.collate_fn)
    src_loader_val = torch.utils.data.DataLoader(src_val_dataset, batch_size=1, sampler=src_test_sampler,
                                                  num_workers=args.workers, collate_fn=src_val_dataset.collate_fn)

    tar_loader = torch.utils.data.DataLoader(tar_train_dataset, batch_sampler=tar_train_batch_sampler,
                                             num_workers=args.workers, collate_fn=tar_train_dataset.collate_fn)
    tar_loader_test = torch.utils.data.DataLoader(tar_test_dataset, batch_size=1, sampler=tar_test_sampler,
                                                  num_workers=args.workers, collate_fn=tar_test_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(args.backbone, num_classes=args.num_classes + 1, device=device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # find_unused_parameters=True 有forward的返回值不在计算loss的计算图里
        # 即返回值不进入backward去算grad，也不需要在不同进程之间进行通信。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        print("source test:")
        utils.evaluate(model, src_loader_val, device=device)
        print("target test:")
        utils.evaluate(model, tar_loader_test, device=device)
        return

    train_loss = []
    learning_rate = []
    t_test_map = []
    s_val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            src_train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, src_loader, tar_loader, device,
                                              epoch, args.print_freq, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        # 目标域效果
        coco_info = utils.evaluate(model, tar_loader_test, device=device)
        t_test_map.append(coco_info[1])  # pascal mAP

        # /add: calculate voc info for every classes(IoU=0.5)
        voc_map_info_list = []
        for i in range(len(args.num_classes)):
            stats, _ = summarize(coco_info, catId=i)
            voc_map_info_list.append(" {:15}: {}".format(args.num_classes[i + 1], stats[1]))
        print_voc = "\n".join(voc_map_info_list)   # add/

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            # write into txt
            with open(t_results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                # round(x,y)函数返回浮点数x的四舍五入值，保留4个小数点
                txt = "epoch:{} {} {}".format(epoch, '  '.join(result_info), '  '.join(print_voc))
                f.write(txt + "\n")
                # evaluate after every epoch

        # ADD：源域上测试看效果
        if args.source_val:
            s_coco_info = utils.evaluate(model, src_loader_val, device=device)
            s_val_map.append(s_coco_info[1])  # pascal mAP

            # /add: calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            for i in range(len(args.num_classes)):
                stats, _ = summarize(s_coco_info, catId=i)
                voc_map_info_list.append(" {:15}: {}".format(args.num_classes[i + 1], stats[1]))
            print_voc = "\n".join(voc_map_info_list)  # add/

            # 写入txt文件，只在主进程上进行写操作
            if args.rank in [-1, 0]:
                # write into txt
                with open(s_results_file, "a") as f:
                    # 写入的数据包括coco指标还有loss和learning rate
                    result_info = [str(round(i, 4)) for i in s_coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info), '  '.join(print_voc))
                    f.write(txt + "\n")

        # 保存模型
        if args.output_dir_weight:
            # 只在主节点上执行保存权重操作
            save_name = os.path.join(args.output_dir_weight, mode + "_"  + args.backbone,
                                     'model_{}.pth'.format(epoch))
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch}, save_name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate, mode,
                             dir=os.path.join(args.record_dir, args.backbone))

        # plot mAP curve
        if len(t_test_map) != 0:
            from plot_curve import plot_map
            plot_map(t_test_map, 't_test_map', mode,
                     dir=os.path.join(args.record_dir, args.backbone))

        # ADD：源域上测试map
        if len(s_val_map) != 0 and args.source_val:
            from plot_curve import plot_map
            plot_map(s_val_map, 's_val_map', mode,
                     dir=os.path.join(args.record_dir, args.backbone))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--domain_datasets', default='V2C', help='mode')
    parser.add_argument('--backbone', default='resnet50_fpn_backbone', help='model backbone')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 学习率，这个需要根据gpu的数量以及batch_size进行设置0.02 / 8 * num_GPU
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[7, 12], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    # pth模型保存地址
    parser.add_argument('--output-dir-weight', default='./save/model/DA', help='path where to save')
    # png、txt保存地址
    parser.add_argument('--record-dir', default='./save/outputs/DA', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # anchor的设置k
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 默认只看t_test效果，开启时也会看s_val的效果
    parser.add_argument( "--source-val",
                         dest="source_val",
                         help="test the t_test and s_val",
                         action="store_true",)
    # 不训练，仅测试
    parser.add_argument( "--test-only",
                         dest="test_only",
                         help="Only test the model",
                         action="store_true",)

    # 开启的进程数(注意不是线程)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if not os.path.exists(os.path.join(args.output_dir_weight, args.domain_datasets + "_" + args.backbone)):
        os.makedirs(os.path.join(args.output_dir_weight, args.domain_datasets + "_" + args.backbone))

    if not os.path.exists(args.record_dir, args.backbone):
        os.makedirs(args.record_dir, args.backbone)

    main(args)
