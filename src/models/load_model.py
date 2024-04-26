import os
import timm
import torch
from torchvision.models import resnet50, ResNet50_Weights, convnext_base, ConvNeXt_Base_Weights, efficientnet_b0, EfficientNet_B0_Weights
from ..models import *
from copy import deepcopy
from src.models.convnext import ConvNeXt_para
from src.models.ResNet_para import ResNet_para
from src.models.vit_para import VisionTransformer_para
from src.models.volo_para import VOLO_para

def load_model(model_name, checkpoint_dir=None, domain=None, para_scale=0.1):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'vit':
        model=timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'convnext_base':

        model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    elif model_name == 'resnet50-gn':
    ############################################################################################### elif start
        model = timm.create_model('resnet50_gn', pretrained=True)
    ############################################################################################### elif end

    elif model_name == 'resnet50_gn_para':
    #################################################################################################################### elif start
        model_args = dict(layers=[3, 4, 6, 3], norm_layer='groupnorm')
        model = ResNet_para(**dict(model_args))

        model_load = timm.create_model('resnet50_gn', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state ,strict=False)
    #################################################################################################################### elif end

    elif model_name == 'vit_para':
    ######################################################################################################################### elif start
        model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model = VisionTransformer_para(**dict(model_args))

        model_load = timm.create_model('vit_base_patch16_224', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
    ########################################################################################################################### elif end

    elif model_name == 'convnext_base_para':
    ################################################################################################### elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state ,strict=False)

        if torch.cuda.is_available():
             if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    #################################################################################################### elif end

    elif model_name == 'convnext_base_para_384':
    ################################################################################################### elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state ,strict=False)
    #################################################################################################### elif end

    elif model_name == 'convnextv2_huge_para':
    ######################################################################################## elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, ls_init_value=None)
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state ,strict=False)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    ########################################################################################### elif end

    elif model_name == 'volo_para':
    ################################################################################################################## elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(layers=(12, 12, 20, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), mlp_ratio=4, stem_hidden_dim=128, para_scale=para_scale)
        model = VOLO_para(**dict(model_args))
        model_load = timm.create_model('volo_d5_224.sail_in1k', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state ,strict=False)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    ################################################################################################################## elif end

    elif model_name == 'volo':
    ################################################################################################################## elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model= timm.create_model('volo_d5_224.sail_in1k', pretrained=True)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    ################################################################################################################## elif end

    else:
        raise ValueError('Unknown model name')


    return model
