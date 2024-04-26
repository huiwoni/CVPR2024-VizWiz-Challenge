import logging
import os
import time

import numpy as np
from datetime import datetime
from src.methods import *
from src.utils import create_submit_file, get_args, create_submit_file_for_new_idea
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
from src.models.load_model import load_model
from src.data.data import load_dataset
import json
logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def evaluate(cfg):

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN, para_scale=cfg.MODEL.PARA_SCALE)
    base_model = base_model.cuda()

    # logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == 'parallel_psedo':
        model, param_names, param_names_ema = setup_parallel_psedo(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == 'parallel_psedo_lame':
        model, param_names, param_names_ema = setup_parallel_psedo(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == 'parallel_psedo_contrast':
        model, param_names, param_names_ema = setup_parallel_psedo_contrast(base_model, cfg, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = dom_names_all
    severities = cfg.CORRUPTION.SEVERITY

    ######################################################################################################### func start
    annotations = json.load(open(cfg.DATA_DIR + "/challenge/annotations.json"))
    image_list = annotations["images"]
    indices_in_1k = [d['id'] for d in annotations['categories']]
    ######################################################################################################### func end

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):

        for severity in severities:
            testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                                cfg.TEST.BATCH_SIZE,
                                                split='all', domain=domain_name, level=severity,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                num_aug=cfg.TEST.N_AUGMENTATIONS,
                                                model_arch=cfg.MODEL.ARCH)

            for epoch in range(cfg.TEST.EPOCH):
            ############################################################################################################ for start

                if cfg.MODEL.ADAPTATION == 'parallel_psedo_contrast':
                ############################################################### if start
                    results = create_submit_file_for_new_idea(model, data_loader=test_loader, mask = indices_in_1k, epoch = epoch, image_list = image_list)
                ############################################################### if end

                else:
                ######################################################## else start
                    results = create_submit_file(model, data_loader=test_loader, mask = indices_in_1k)
                ######################################################## else end

                if ((epoch + 1) % 5) == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT, f'trained_{epoch+1}.pth'))

                file_path = os.path.join(cfg.OUTPUT, datetime.now().strftime(f'{epoch+1}epoch_prediction-%m-%d-%Y-%H:%M:%S.json'))
                with open(file_path, 'w') as outfile:
                    json.dump(results, outfile)
            ############################################################################################################ for end

    return results


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'online_evaluation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)
    start_time = time.time()
    accs = []
    for domain in cfg.CORRUPTION.SOURCE_DOMAINS:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        results = evaluate(cfg)

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
