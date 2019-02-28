import torch
from dataset import *
import numpy as np
from models import *
import config
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import *
import os

if __name__=="__main__":
    #test_set = getTestData(config.test['data_root'])
    #test_set = TestDataset('/media/arthur/新加卷1/KK/sketch_synthesis/pointing04/')
    test_set = TestDataset('/media/arthur/新加卷/KK/sketch_synthesis/sketch_test/lighting/')
    dataloader = torch.utils.data.DataLoader(dataset=test_set, num_workers=8, batch_size=config.test['batch_size'],
                                             shuffle=False)
    #img_list = [x for x in listdir(os.path.join(config.test['data_root'], 'Testing/Photos')) if isImageFile(x)]
    #img_list = [x for x in listdir('/media/arthur/新加卷1/KK/sketch_synthesis/pointing04/') if isImageFile(x)]
    img_list = [x for x in listdir('/media/arthur/新加卷/KK/sketch_synthesis/sketch_test/lighting/') if isImageFile(x)]
    '''G1 = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                         use_dropout=True).cuda()'''
    G1 = sketchGenerator(input_nc=1, output_nc=3).cuda()
    D1 = D(in_nc=3, out_nc=3).cuda()
    G1.load_state_dict(torch.load('./saved/G_checkpoint.pth'), strict=True)
    set_requires_grad(G1, False)

    for step, batch in enumerate(dataloader):
        for k in batch:
            batch[k] = Variable(batch[k].cuda(async=True))

        real_img = batch['photo']
        fake_imgs = G1(batch['photo'])

        vis_real_img = tensor2im(real_img.data)
        vis_fake_img_unmasked = tensor2im(fake_imgs.data)

        '''plt.subplot(321), plt.imshow(vis_real_img), plt.title("vis_real_img")
        plt.subplot(322), plt.imshow(vis_fake_img_unmasked), plt.title("vis_fake_img_unmasked")

        plt.show()'''
        for i in range(fake_imgs.shape[0]):
            img_name = img_list[step * config.test['batch_size'] + i]
            save_img(fake_imgs.data, img_name)
