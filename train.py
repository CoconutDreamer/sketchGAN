import torch
import config
from dataset import *
import numpy as np
from skimage.io import imsave
from torch.autograd import Variable
from utils import *
from losses import *
import matplotlib.pyplot as plt
from torch.backends import cudnn
import torch.nn as nn
from models import *
import functools

cudnn.benchmark = True
load_checkpoint = False
if __name__ == "__main__":
    train_set = getTrainData(config.train['data_root'])
    dataloader = torch.utils.data.DataLoader(dataset=train_set, num_workers=8, batch_size=config.train['batch_size'], shuffle=True)

    '''G1 = ResnetGenerator(input_nc=3, output_nc=3,  norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                          use_dropout=True).cuda()'''
    G1 = sketchGenerator(input_nc=1, output_nc=3).cuda()
    G1.apply(weightsInit)
    D1 = D(in_nc=3, out_nc=3).cuda()
    #D1 = Discriminator(input_nc=3).cuda()
    D1.apply(weightsInit)

    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G1.parameters()), lr=config.train['learning_rate'])
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D1.parameters()), lr=config.train['learning_rate'])
    if load_checkpoint:
        G1.load_state_dict(torch.load('./saved/G_checkpoint.pth'), strict=True)
        D1.load_state_dict(torch.load('./saved/D_checkpoint.pth'), strict=True)
        optimizer_D.load_state_dict(torch.load('./saved/optimizer_D_checkpoint.pth'))
        optimizer_G.load_state_dict(torch.load('./saved/optimizer_G_checkpoint.pth'))

    l1_loss = torch.nn.L1Loss().cuda()
    gan_loss = GANLoss().cuda()

    last_epoch = -1
    for epoch in range(last_epoch + 1, config.train['num_epochs']):
        for step, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = Variable(batch[k].cuda(async=True), requires_grad=False)
            batch_size = batch['photo'].size(0)
            '''import utils
            vis = utils.tensor2im((batch['sketch']))
            vis2 = utils.tensor2maskim(batch['photo'])
            import matplotlib.pyplot as plt
            plt.subplot(321), plt.imshow(vis)
            plt.subplot(322), plt.imshow(vis2)
            plt.show()'''

            #face = batch['face'] > 0
            print(batch['face'])
            # generate fake imgs
            fake_imgs = G1(batch['photo'], batch['face'])
            #fake_imgs = (face_mask * face) + fake_imgs
            if step % config.train['train_D_step'] == 0:
                # train D
                set_requires_grad(D1, True)
                adv_D_loss_real = (gan_loss(D1(batch['sketch']), True) + gan_loss(D1(fake_imgs.detach()), False)) * 0.5
                #adv_D_loss_local = (gan_loss(D1(batch['mask_d'] * batch['sketch']), True) + gan_loss(D1(batch['mask_d'] * fake_imgs.detach()), False)) * 0.5
                adv_D_loss = adv_D_loss_real

                '''alpha = torch.rand(batch['sketch'].size(0), 1, 1, 1).expand_as(
                    batch['sketch']).contiguous().pin_memory().cuda(async=True)
                interpolated_x = Variable(alpha * fake_imgs.detach().data + (1.0 - alpha) * batch['sketch'].data,
                                          requires_grad=True)
                out = D1(interpolated_x)
                dxdD = \
                torch.autograd.grad(outputs=out, inputs=interpolated_x, grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True, create_graph=True, only_inputs=True)[0].view(out.shape[0], -1)
                gp_loss = torch.mean((torch.norm(dxdD, p=2) - 1) ** 2)'''
                # + config.loss['weight_gradient_penalty'] * gp_loss
                L_D = adv_D_loss

                optimizer_D.zero_grad()
                L_D.backward()
                optimizer_D.step()

            # train G
            set_requires_grad(D1, False)
            adv_G_loss_real = gan_loss(D1(fake_imgs), True)
            #adv_G_loss_local = gan_loss(D1(batch['mask_d'] * fake_imgs), True)
            adv_G_loss = adv_G_loss_real
            # adv_G_loss = gan_loss(D(depth), True)
            pixel_loss = l1_loss(fake_imgs, batch['sketch'])
            tv_loss = torch.mean(torch.abs(fake_imgs[:, :, :-1, :] - fake_imgs[:, :, 1:, :])) + torch.mean(
                torch.abs(fake_imgs[:, :, :, :-1] - fake_imgs[:, :, :, 1:]))

            L_G = config.loss['weight_adv_G'] * adv_G_loss + config.loss['weight_pixel_loss'] * pixel_loss + config.loss['weight_tv_loss'] * tv_loss

            optimizer_G.zero_grad()
            L_G.backward(retain_graph=True)
            optimizer_G.step()

            if step % config.train['log_step'] == 0:
                print(
                    "epoch {} , step {} / {} ,  adv_D_loss_real {:.3f},  G_loss {:.3f}, adv_G_loss_real {:.3f}, pixel_loss {:.3f}, tv_loss {:.3f}".format(
                        epoch, step, len(dataloader), adv_D_loss_real.data,  L_G.data.cpu().numpy(), adv_G_loss_real.data,
                        pixel_loss.data, tv_loss.data))

        torch.save(G1.state_dict(), '{}/G_checkpoint.pth'.format(config.train['save_path']))
        torch.save(D1.state_dict(), '{}/D_checkpoint.pth'.format(config.train['save_path']))
        torch.save(optimizer_D.state_dict(), '{}/optimizer_D_checkpoint.pth'.format(config.train['save_path']))
        torch.save(optimizer_G.state_dict(), '{}/optimizer_G_checkpoint.pth'.format(config.train['save_path']))
        print("Save done at {}".format(config.train['save_path']))
