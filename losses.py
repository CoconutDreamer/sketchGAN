import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def l1_pixel_loss(fake_img, gt_img):
    # if gt_img is 0/255, that pixel won't count loss
    l1_loss = 0
    gt_mask = 1- ((gt_img == 255) | (gt_img == 0)).prod(1, keepdim=True).type_as(fake_img)
    diff = (fake_img - gt_img) * gt_mask
    l1_loss += diff.abs().mean()

    return l1_loss


def weight_l1_loss(fake_img, gt_img, mask):
    l1_loss = 0
    diff = (fake_img - gt_img) * mask
    l1_loss += diff.abs().mean()

    return l1_loss