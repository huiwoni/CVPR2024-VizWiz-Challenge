"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from src.data.augmentations import get_tta_transforms
from src.utils.utils import deepcopy_model
from torch.autograd import Variable
from src.utils.loss import SupConLoss
from ..models.base_model import BaseModel
import torch.nn.functional as F


class Parallel_psedo_contrast(nn.Module):
    ######################################################################################################################## class start
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, ema_model, optimizer_teacher, mt_alpha, rst_m, ap, dataset_name, steps=1,
                 episodic=False, num_aug=32, adaptation_type='otta', output_dir=None, use_memory=None, max_epoch=10,
                 arch_name=None, contrast=3):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BaseModel(model, arch_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.to(device)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

        self.model_ema = ema_model
        self.optimizer_teacher = optimizer_teacher

        self.model_state, self.optimizer_state, self.model_ema_state, self.optimizer_teacher_state, self.model_anchor = \
            copy_model_and_optimizer(model, self.optimizer, self.model_ema, self.optimizer_teacher)

        self.transform = get_tta_transforms(dataset_name)
        self.softmax_entropy = softmax_entropy_cifar if "cifar" in dataset_name else softmax_entropy_imagenet
        self.num_aug = num_aug
        self.adaptation_type = adaptation_type
        self.output_dir = output_dir

        self.use_memory = use_memory
        self.psedo_lable_bank = torch.zeros((8900, max_epoch), device=device)
        self.supconloss = SupConLoss()
        self.contrast = contrast
    ######################################################################################################################## class end


    def forward(self, x, epoch, iter, class_mask, class_number):
        ######################################################################################################################################## forward start

        if self.adaptation_type == 'ttba':
        ################################################################################################################ if for ttba start
            self.reset()
            self.model.train()

            if self.steps > 0:
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        ################################################################################################################ if for ttba end

        elif (self.adaptation_type == 'otta') or (self.adaptation_type == 'ttda'):
        ####################################################################################### if for otta, ttda start
            self.model.train()
            self.model_ema.train()

            if self.steps > 0:
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(x, self.optimizer, self.optimizer_teacher, epoch, iter, class_mask,
                                                     class_number)
        ######################################################################################## if for otta end

        ######################################################################################################################################### forward end
        return outputs



    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer, optimizer_teacher, epoch, iter, class_mask, class_number):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """

        outputs_emas = []
        loss_cts_bank = []

        # forward
        _, outputs = self.model(x, return_feats=True)
        outputs = (outputs * class_mask)[:, class_number]

        standard_ema = (self.model_ema(x) * class_mask)[:, class_number]
        anchor_prob = torch.nn.functional.softmax((self.model_anchor(x) * class_mask)[:, class_number], dim=1).max(1)[0]
        to_aug = anchor_prob.mean(0) < self.ap

        if to_aug:
            for i in range(self.num_aug):
                outputs_ = (self.model_ema(self.transform(x)) * class_mask)[:, class_number].detach()
                outputs_emas.append(outputs_)
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        w = loss_weight(outputs_ema.detach())

        ################################################################################################################ loss ce for student start
        loss_ce = self.softmax_entropy(outputs, outputs_ema.detach())
        loss_ce = (w * loss_ce)

        if self.use_memory is not None:
        #################################################################################################### if start
            label_mask = self.save_refine_psedo_lable(outputs_ema.detach(), epoch, iter)
            loss_ce = (label_mask * loss_ce).mean(0)
        #################################################################################################### if end

        else:
        ################################################## else start
            loss_ce = (w * loss_ce).mean(0)
        ################################################## else end

        ################################################################################ loss div for student start
        loss_div = softmax_entropy(outputs).mean(0)
        ################################################################################ loss div for student end

        student_loss = (loss_ce + loss_div) / 2
        student_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ################################################################################################################ loss ce for student end

        ############################################################### loss ce for teacher start
        teacher_loss = softmax_entropy(outputs_ema).mean(0)
        teacher_loss.backward()
        optimizer_teacher.step()
        optimizer_teacher.zero_grad()
        ############################################################### loss ce for teacher end

        ######################################################################################################################################### contrast loss start
        feats_masks = torch.where(label_mask == True)[0]
        target_image = x[feats_masks, :, :]
        outputs_ema_masked = outputs_ema[feats_masks, :].detach()
        w_masked = w[feats_masks]

        batch_sise = target_image.size()[0]

        for i in range(int(batch_sise / self.contrast)):
        ######################################################### for start
            image_bank = []
            num_image = i * self.contrast

            if i == (int(batch_sise / self.contrast) - 1):
            #################################################################### if start
                loss_input_img = target_image[(batch_sise - self.contrast):]
                loss_input_w = w_masked[(batch_sise - self.contrast):].repeat(self.contrast)
                loss_input_label = (outputs_ema_masked[(batch_sise - self.contrast):].argmax(dim=1)).repeat(self.contrast)
            #################################################################### if end
            else:
            #################################################################### if start
                loss_input_img = target_image[num_image:num_image + self.contrast]
                loss_input_w = w_masked[num_image:num_image + self.contrast].repeat(self.contrast)
                loss_input_label = (outputs_ema_masked[num_image:num_image + self.contrast].argmax(dim=1)).repeat(self.contrast)
            #################################################################### if end

            image_bank.append(loss_input_img)
            aug_image = self.augment_image(loss_input_img.detach(), image_bank)

            feats, _ = self.model(aug_image, return_feats=True)

            norm_feature = self.norm_feature(feats)

            loss_cts = (loss_input_w * self.supconloss(norm_feature, loss_input_label)).mean()
            loss_cts_bank.append(loss_cts)
            loss_cts.backward()
            optimizer.step()
            optimizer.zero_grad()
        ######################################################### for end

        if batch_sise < self.contrast:
        ################################################################################################# if start
            loss_input_img = target_image
            loss_input_w = w_masked
            loss_input_label = outputs_ema_masked.argmax(dim=1)

            image_bank.append(loss_input_img)
            aug_image = self.augment_image(loss_input_img.detach(), image_bank)

            feats, _ = self.model(aug_image, return_feats=True)

            norm_feature = self.norm_feature(feats)

            loss_cts = (loss_input_w * self.supconloss(norm_feature, loss_input_label)).mean()
            loss_cts_bank.append(loss_cts)
            loss_cts.backward()
            optimizer.step()
            optimizer.zero_grad()
        ################################################################################################### if end

        ######################################################################################################################################### contrast loss end

        with open(self.output_dir + f'/parallel_psedo_{str(self.model)[:7]}.txt', 'a') as f:
        ######################################################################################################################### func start
            f.write(
                f'mask {int(label_mask.sum())} loss_cts: {sum(loss_cts_bank, 0)/len(loss_cts_bank)} loss_div: {loss_div} loss_ce: {loss_ce} student_loss: {student_loss} teacher_loss: {teacher_loss} \n')
        ######################################################################################################################### func end

        return outputs


    def reset(self):

        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def augment_image(self, aug_image, image_bank):
    ######################################################################################### func start

        for i in range(self.contrast - 1):
        ######################################################### for start
            aug_image = self.transform(aug_image)
            image_bank.append(aug_image)
        ######################################################### for end

    ######################################################################################### func end
        return torch.concat(image_bank)



    def save_refine_psedo_lable(self, psedo, epoch, iter):
        ############################################################################## func start
        predictions = psedo.argmax(1)
        start = iter * len(psedo)
        end = start + len(psedo)
        mask = torch.ones(len(psedo), dtype=bool, device="cuda")

        self.psedo_lable_bank[start:end, epoch] = predictions

        if epoch == 0:
        ###################################### if start
            return mask
        ###################################### if end

        elif epoch < (self.use_memory + 1):
            ############################################################## else start
            select_past = range(epoch)

            for i in range(len(psedo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[start + i, select_past] != predictions[i])[0]) < 1
            ######################################## for end

        else:
        ######################################################### else start
            select_past = range(epoch - self.use_memory, epoch)

            for i in range(len(psedo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[start + i, select_past] != predictions[i])[0]) < 1
            ######################################## for end

        ############################################################## else end

    ############################################################################## func end
        return mask

    @staticmethod
    def norm_feature(feature):
    ############################################# func start
        feat_euc = (feature ** 2).sum(dim=1).unsqueeze(dim=1)
        feat_size = feat_euc ** (0.5)
        nomed_feat = (feature / feat_size)
    ############################################# func end
        return nomed_feat.unsqueeze(1)

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()

        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)

        # configure norm for tent updates: enable grad + force batch statisics
        for nm, m in model.named_modules():
            if "parallel" in nm.split("."):
                m.requires_grad_(True)

        return model

    @staticmethod
    def configure_model_ema(model_ema):
        ######################################################################################################################## func start
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model_ema.train()

        # disable grad, to (re-)enable only what tent updates
        model_ema.requires_grad_(False)

        for nm, m in model_ema.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                m.requires_grad_(True)
        ######################################################################################################################## func end
        return model_ema

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []

        for nm, m in model.named_modules():
            if "parallel" in nm.split("."):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    @staticmethod
    def collect_params_ema(model):
        #################################################################################################################### func start
        params = []
        names = []

        for nm, m in model.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        #################################################################################################################### func end
        return params, names

    @staticmethod
    def check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "tent should not update all params: " \
                                   "check which require grad"
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"


@torch.jit.script
def loss_weight(x: torch.Tensor) -> torch.Tensor:
    ################################################################################################ func start
    x = torch.nn.functional.softmax(x, dim=1)
    max_entropy = torch.log2(torch.tensor(200))
    w = -torch.sum(x * torch.log2(x + 1e-5), dim=1)
    w = w / max_entropy
    w = torch.exp(-w)
    ################################################################################################ func end
    return w


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def mean_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    temprature = 1
    x = x / temprature
    mean_probe_d = torch.mean(x.softmax(1), dim=0)
    entropy = -torch.sum(mean_probe_d * torch.log(mean_probe_d))
    return entropy


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1
    x = -(temprature * torch.logsumexp(x / temprature, dim=1))
    if torch.rand(1) > 0.95:
        print(x.mean(0).item())
    return x


def copy_model_and_optimizer(model, optimizer, model_ema, optimizer_theacher):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    #################################################################################################################### func start
    model_ema_state = deepcopy(model_ema.state_dict())
    optimizer_theacher_state = deepcopy(optimizer_theacher.state_dict())

    model_anchor = deepcopy_model(model_ema)
    model_anchor.requires_grad_(False)
    #################################################################################################################### func end

    return model_state, optimizer_state, model_ema_state, optimizer_theacher_state, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def softmax_entropy_cifar(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema):  # -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)