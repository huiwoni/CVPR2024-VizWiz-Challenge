import math

from torch import optim, nn

from src.data import load_dataset
from .T3A import T3A
from .adacontrast import AdaContrast
from .bn import AlphaBatchNorm
from .cotta import CoTTA
from .eata import EATA
from .lame import LAME
from .memo import MEMO
from .norm import Norm
from .sar import SAR
from .parallel_conv_tent import Tent_para
from .parallel_conv_sar import SAR_para
from .parallel_psedo import Parallel_psedo
from .tent import Tent
from .challenge_tent_gem_t import Tent_da
from .parallel_psedo_promix import Parallel_psedo_promix
from .parallel_psedo_contrast import Parallel_psedo_contrast
from ..models import *
from ..utils.utils import split_up_model
from .vida import configure_model, collect_params, ViDA
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
#######################################################################################################################
def setup_optimizer_vida(params, params_vida, cfg):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": cfg.OPTIM.LR},
                                  {"params": params_vida, "lr": cfg.VIDA.ViDALR}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.VIDA.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": cfg.OPTIM.LR},
                                  {"params": params_vida, "lr": cfg.VIDA.ViDALR}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.VIDA.WD)
    else:
        raise NotImplementedError
########################################################################################################################
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


def setup_NRC_optimizer(model, cfg):
    if isinstance(model, (OfficeHome_Shot, DomainNet126_Shot)):
        param_group = []
        param_group_c = []
        for k, v in model.netF.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]

        for k, v in model.netB.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        for k, v in model.netC.named_parameters():
            param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        optimizer = optim.SGD(param_group)
        optimizer_c = optim.SGD(param_group_c)
        return op_copy(optimizer), op_copy(optimizer_c)
    else:
        encoder, classifier = split_up_model(model)
        param_group = []
        param_group_c = []
        for k, v in encoder.named_parameters():
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]
        for k, v in classifier.named_parameters():
            param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        optimizer = optim.SGD(param_group)
        optimizer_c = optim.SGD(param_group_c)
        return op_copy(optimizer), op_copy(optimizer_c)


def setup_shot_optimizer(model, cfg):
    if isinstance(model, (OfficeHome_Shot, DomainNet126_Shot)):
        param_group = []
        for k, v in model.netF.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]

        for k, v in model.netB.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        for k, v in model.netC.named_parameters():
            v.requires_grad = False
        optimizer = optim.SGD(param_group)
        return op_copy(optimizer)
    else:
        encoder, classifier = split_up_model(model)
        param_group = []
        for k, v in encoder.named_parameters():
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]
        for k, v in classifier.named_parameters():
            v.requires_grad = False
        optimizer = optim.SGD(param_group)
        return op_copy(optimizer)


def setup_source(model, cfg=None):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model, None


def setup_t3a(model, cfg=None):
    model.eval()
    T3A_model = T3A(model, filter_k=cfg.T3A.FILTER_K, cached_loader=False)
    return T3A_model, None


def setup_test_norm(model, cfg):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    model.eval()
    for m in model.modules():
        # Re-activate batchnorm layer
        if (isinstance(m, nn.BatchNorm1d) and cfg.TEST.BATCH_SIZE > 1) or isinstance(m,
                                                                                     (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # Wrap test normalization into Norm class to enable sliding window approach
    norm_model = Norm(model)
    return norm_model, None


def setup_tent(model, cfg):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return tent_model, param_names

########################################################################################################################
def setup_tent_da(model, cfg):
    model = Tent_da.configure_model(model)
    params, param_names = Tent_da.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent_da(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return tent_model, param_names

########################################################################################################################
def setup_tent_para(model, cfg):
    model = Tent_para.configure_model(model)
    params, param_names = Tent_para.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent_para(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return tent_model, param_names

########################################################################################################################

def setup_parallel_psedo(model, cfg, num_classes):
######################################################################################################################## func start
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Parallel_psedo.configure_model(model)
    params, param_names = Parallel_psedo.collect_params(model)
    optimizer = setup_optimizer(params, cfg)

    if cfg.MODEL.ARCH == "convnextv2_huge_para":
        # ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
        ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para_384":
        # ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
        ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para":
        ema_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif cfg.MODEL.ARCH == "volo_para":
        ema_model = timm.create_model('volo_d5_224.sail_in1k', pretrained=True)

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


########################################################################################################################
def setup_parallel_psedo_contrast(model, cfg, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Parallel_psedo_contrast.configure_model(model)
    params, param_names = Parallel_psedo_contrast.collect_params(model)
    optimizer = setup_optimizer(params, cfg)

    if cfg.MODEL.ARCH == "convnextv2_huge_para":
        # ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
        ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para_384":
        # ema_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
        ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para":
        ema_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif cfg.MODEL.ARCH == "volo_para":
        ema_model = timm.create_model('volo_d5_224.sail_in1k', pretrained=True)

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

    return parallel_psedo_model, param_names, param_names_ema

########################################################################################################################

def setup_parallel_psedo_promix(model, cfg):

    model = Parallel_psedo_promix.configure_model(model)
    params, param_names = Parallel_psedo_promix.collect_params(model)
    optimizer = setup_optimizer(params, cfg)

    if cfg.MODEL.ARCH == "convnext_base_para_384":
        ema_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
    elif cfg.MODEL.ARCH == "convnext_base_para":
        ema_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)

    ema_model = Parallel_psedo_promix.configure_model_ema(ema_model)
    params_ema, param_names_ema = Parallel_psedo_promix.collect_params_ema(ema_model)
    cfg.OPTIM.LR = 5e-5
    optimizer_teacher = setup_optimizer(params_ema, cfg)

    parallel_psedo_model = Parallel_psedo_promix(model, optimizer, ema_model, optimizer_teacher,
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

    return parallel_psedo_model, param_names, param_names_ema

########################################################################################################################

def setup_cotta(model, cfg):
    model = CoTTA.configure_model(model)
    params, param_names = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    cotta_model = CoTTA(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP,
                        adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return cotta_model, param_names


def setup_eata(model, num_classes, cfg):
    # compute fisher informatrix
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    fisher_dataset, fisher_loader = load_dataset(dataset=cfg.CORRUPTION.SOURCE_DATASET,
                                                 split='val' if cfg.CORRUPTION.SOURCE_DATASET == 'imagenet' else 'train',
                                                 root=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                                 batch_size=batch_size_src,
                                                 domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    # sub dataset
    fisher_dataset = torch.utils.data.Subset(fisher_dataset, range(cfg.EATA.NUM_SAMPLES))
    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=batch_size_src, shuffle=True,
                                                num_workers=cfg.TEST.NUM_WORKERS, pin_memory=True)
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, batch in enumerate(fisher_loader, start=1):
        images = batch[0].cuda(non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    # logger.info("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = setup_optimizer(params, cfg)
    eta_model = EATA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     fishers=fishers,
                     fisher_alpha=cfg.EATA.FISHER_ALPHA,
                     e_margin=math.log(num_classes) * (0.40 if cfg.EATA.E_MARGIN_COE is None else cfg.EATA.E_MARGIN_COE),
                     d_margin=cfg.EATA.D_MARGIN
                     )

    return eta_model, param_names


def setup_lame(model, cfg):
    model = LAME.configure_model(model)
    lame_model = LAME(model,
                      affinity=cfg.LAME.AFFINITY,
                      knn=cfg.LAME.KNN,
                      sigma=cfg.LAME.SIGMA,
                      force_symmetry=cfg.LAME.FORCE_SYMMETRY)
    return lame_model, None


def setup_memo(model, cfg):
    model, param_names = setup_alpha_norm(model, cfg)
    model = model.cuda()
    # model.eval()
    params, param_names = MEMO.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    memo_model = MEMO(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      n_augmentations=cfg.TEST.N_AUGMENTATIONS,
                      dataset_name=cfg.CORRUPTION.DATASET,
                      adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return memo_model, param_names


def setup_alpha_norm(model, cfg):
    """Set up BN--0.1 (test-time normalization adaptation with source prior).
    Normalize features by combining the source moving statistics and the test batch statistics.
    """
    model.eval()
    norm_model = AlphaBatchNorm.adapt_model(model,
                                            alpha=cfg.BN.ALPHA).cuda()  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model, None


def setup_adacontrast(model, cfg):
    model = AdaContrast.configure_model(model)
    params, param_names = AdaContrast.collect_params(model)

    optimizer = setup_optimizer(params, cfg)

    adacontrast_model = AdaContrast(model, optimizer,
                                    steps=cfg.OPTIM.STEPS,
                                    episodic=cfg.MODEL.EPISODIC,
                                    arch_name=cfg.MODEL.ARCH,
                                    queue_size=cfg.ADACONTRAST.QUEUE_SIZE,
                                    momentum=cfg.M_TEACHER.MOMENTUM,
                                    temperature=cfg.CONTRAST.TEMPERATURE,
                                    contrast_type=cfg.ADACONTRAST.CONTRAST_TYPE,
                                    ce_type=cfg.ADACONTRAST.CE_TYPE,
                                    alpha=cfg.ADACONTRAST.ALPHA,
                                    beta=cfg.ADACONTRAST.BETA,
                                    eta=cfg.ADACONTRAST.ETA,
                                    dist_type=cfg.ADACONTRAST.DIST_TYPE,
                                    ce_sup_type=cfg.ADACONTRAST.CE_SUP_TYPE,
                                    refine_method=cfg.ADACONTRAST.REFINE_METHOD,
                                    num_neighbors=cfg.ADACONTRAST.NUM_NEIGHBORS,
                                    adaptation_type = cfg.MODEL.ADAPTATION_TYPE)

    return adacontrast_model, param_names


def setup_sar(model, cfg, num_classes):
    sar_model = SAR(model, lr=cfg.OPTIM.LR, batch_size=cfg.TEST.BATCH_SIZE, steps=cfg.OPTIM.STEPS,
                    num_classes=num_classes, episodic=cfg.MODEL.EPISODIC, reset_constant=cfg.SAR.RESET_CONSTANT,
                    e_margin=math.log(num_classes) * (0.40 if cfg.SAR.E_MARGIN_COE is None else cfg.SAR.E_MARGIN_COE),
                    adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return sar_model

########################################################################################################################
def setup_sar_para(model, cfg, num_classes):
    sar_model = SAR_para(model, lr=cfg.OPTIM.LR, batch_size=cfg.TEST.BATCH_SIZE, steps=cfg.OPTIM.STEPS,
                    num_classes=num_classes, episodic=cfg.MODEL.EPISODIC, reset_constant=cfg.SAR.RESET_CONSTANT,
                    e_margin=math.log(num_classes) * (0.40 if cfg.SAR.E_MARGIN_COE is None else cfg.SAR.E_MARGIN_COE),
                    adaptation_type=cfg.MODEL.ADAPTATION_TYPE)
    return sar_model

def setup_vida(model, cfg):
    model = configure_model(model, cfg)
    model_param, vida_param = collect_params(model)
    optimizer = setup_optimizer_vida(model_param, vida_param, cfg)
    vida_model = ViDA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = 0.2,
                           ema = cfg.VIDA.MT,
                           ema_vida = cfg.VIDA.MT_ViDA,
                           )
    return vida_model

########################################################################################################################