train =  {}
train['data_root'] = '/media/arthur/新加卷/KK/sketch_synthesis/CUFS/CUHK/CUHKStudent/'
train['learning_rate'] = 1e-4
train['num_epochs'] = 350
train['batch_size'] = 1
train['log_step'] = 100
train['save_path'] = './saved'
train['train_D_step'] = 5

test = {}
test['data_root'] = '/media/arthur/新加卷/KK/sketch_synthesis/CUFS/CUHK/CUHKStudent/'
test['batch_size'] = 1
test['result_path'] = './result_light'

G = {}
D = {}

loss = {}
loss['weight_gradient_penalty'] = 10

loss['weight_pixel_loss'] = 1.0
loss['weight_pixel_local'] = 1.0
loss['weight_tv_loss'] = 1e-4
loss['weight_adv_G'] = 1.0
loss['weight_kp_loss'] = 1.0