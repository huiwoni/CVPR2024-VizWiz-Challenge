import math


from torch import optim
from .parallel_psedo import Parallel_psedo
from .parallel_psedo_contrast import Parallel_psedo_contrast

import torch
import timm
import torch.backends.cudnn as cudnn


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == 'AdamW':
        return optim.AdamW(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_adacontrast_optimizer(model, cfg):
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": cfg.OPTIM.LR,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
                {
                    "params": extra_params,
                    "lr": cfg.OPTIM.LR * 10,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIM.METHOD} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


def setup_source(model, cfg=None):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model, None


def setup_parallel_psedo(model, cfg, num_classes):
######################################################################################################################## func start
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Parallel_psedo.configure_model(model)
    params, param_names = Parallel_psedo.collect_params(model)
    optimizer = setup_optimizer(params, cfg)

    if cfg.MODEL.ARCH == "convnextv2_huge_para":
        ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para_384":
        ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para":
        ema_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_clip_para":
        ema_model = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=True)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            ema_model = torch.nn.DataParallel(ema_model)
            ema_model.to(device)

    ema_model = Parallel_psedo.configure_model_ema(ema_model)
    params_ema, param_names_ema = Parallel_psedo.collect_params_ema(ema_model)

    cfg.OPTIM.LR = 5e-5
    optimizer_teacher = setup_optimizer(params_ema, cfg)

    parallel_psedo_model = Parallel_psedo(model, optimizer, ema_model, optimizer_teacher,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP,
                        adaptation_type=cfg.MODEL.ADAPTATION_TYPE,
                        output_dir = cfg.OUTPUT,
                        use_memory = cfg.TEST.USEMEMORY,
                        max_epoch = cfg.TEST.EPOCH)

    parallel_psedo_model.to(device)
    cudnn.benchmark = True

######################################################################################################################## func end
    return parallel_psedo_model, param_names, param_names_ema



def setup_parallel_psedo_contrast(model, cfg, num_classes):
######################################################################################################################## func start
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Parallel_psedo_contrast.configure_model(model)
    params, param_names = Parallel_psedo_contrast.collect_params(model)
    optimizer = setup_optimizer(params, cfg)

    if cfg.MODEL.ARCH == "convnextv2_huge_para":
        ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para_384":
        ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para":
        ema_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_clip_para":
        ema_model = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=True)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            ema_model = torch.nn.DataParallel(ema_model)
            ema_model.to(device)

    ema_model = Parallel_psedo_contrast.configure_model_ema(ema_model)
    params_ema, param_names_ema = Parallel_psedo_contrast.collect_params_ema(ema_model)

    cfg.OPTIM.LR = 5e-5
    optimizer_teacher = setup_optimizer(params_ema, cfg)

    parallel_psedo_model = Parallel_psedo_contrast(model, optimizer, ema_model, optimizer_teacher,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP,
                        adaptation_type=cfg.MODEL.ADAPTATION_TYPE,
                        output_dir=cfg.OUTPUT,
                        use_memory=cfg.TEST.USEMEMORY,
                        max_epoch=cfg.TEST.EPOCH,
                        arch_name=cfg.MODEL.ARCH,
                        contrast=cfg.MODEL.CONTRAST)

    cudnn.benchmark = True
######################################################################################################################## func send
    return parallel_psedo_model, param_names, param_names_ema


