import os
import numpy as np
from torchvision.utils import save_image
import json
import time
from torch.autograd import Variable
from model_v2 import *
from utils import Config
from data_v2 import get_dataloader
import torch
from tqdm import tqdm


def save_sample(batches_done, dataloaders, Tensor, generator):
    masked_samples, samples, _ = next(iter(dataloaders['test']))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    # Generate inpainted image
    generator.eval()
    gen_samples = generator(masked_samples)

    # Save sample
    sample = torch.cat((masked_samples.data, gen_samples.data, samples.data), -2)
    save_image(sample, "images/test_images/%d.png" % (batches_done / Config['sample_interval']), nrow=8,
               normalize=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def extract_parts(imgs, centroids, Tensor, requires_grad=False):
    imgs_part = torch.empty(imgs.shape[0], 3, 128, 128)
    for i in range(imgs.shape[0]):
        imgs_part[i] = imgs[i, :, int(centroids[1][i]) - 64: int(centroids[1][i]) + 64,
                       int(centroids[0][i]) - 64: int(centroids[0][i]) + 64]
    imgs_part = Variable(imgs_part.type(Tensor), requires_grad=requires_grad)
    return imgs_part

def train_model(dataloaders, dataset_size, adversarial_loss, L1_loss, pixelwise_loss,
                generator, global_discriminator, local_discriminator,
                optimizer_G, optimizer_Dg, optimizer_Dl, num_epochs=Config['num_epochs']):
    cuda = True if torch.cuda.is_available() else False

    if cuda:
        generator.cuda()
        local_discriminator.cuda()
        global_discriminator.cuda()
        adversarial_loss.cuda()
        L1_loss.cuda()
        pixelwise_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    local_discriminator.apply(weights_init_normal)
    global_discriminator.apply(weights_init_normal)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # ----------
    #  Training
    # ----------
    loss_dict = {'d_loss_global': [],
                 'd_loss_local': [],
                 'g_adv_local': [],
                 'g_adv_global': [],
                 'g_l1': [],
                 'g_pixel': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)

        running_loss1, running_loss2, running_loss3, running_loss4, running_loss5, running_loss6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (masked_imgs, imgs, centroids) in tqdm(enumerate(dataloaders['train'])):


            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = extract_parts(imgs, centroids, Tensor, requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(masked_imgs)
            gen_parts = extract_parts(gen_imgs, centroids, Tensor, requires_grad=True)

            # Adversarial and pixelwise loss
            g_adv_local = adversarial_loss(local_discriminator(gen_parts), valid)
            g_adv_global = adversarial_loss(global_discriminator(gen_imgs), valid)

            g_l1_global = L1_loss(gen_imgs, imgs)
            g_l1_local = L1_loss(masked_parts, gen_parts)
            g_pixel_global = pixelwise_loss(gen_imgs, imgs)
            g_pixel_local = pixelwise_loss(masked_parts, gen_parts)
            
            g_l1 = g_l1_local+g_l1_global
            g_pixel = 0.8 * g_pixel_local + 0.2 * g_pixel_global
            # Total loss
            g_loss = 0.001 * g_adv_local + 0.001 * g_adv_global + 0.999 * g_l1 + 0.005 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_Dl.zero_grad()

            # Measure discriminator's ability to classify real from generated samples

            real_loss_local = adversarial_loss(local_discriminator(masked_parts.detach()), valid)
            fake_loss_local = adversarial_loss(local_discriminator(gen_parts.detach()), fake)
            d_loss_local = 0.5 * (real_loss_local + fake_loss_local)

            d_loss_local.backward()
            optimizer_Dl.step()

            optimizer_Dg.zero_grad()
            real_loss_global = adversarial_loss(global_discriminator(imgs.detach()), valid)
            fake_loss_global = adversarial_loss(global_discriminator(gen_imgs.detach()), fake)
            d_loss_global = 0.5 * (real_loss_global + fake_loss_global)

            d_loss_global.backward()
            optimizer_Dg.step()


            # Generate sample at sample interval
            batches_done = epoch * len(dataloaders['train']) + i
            if batches_done % Config['sample_interval'] == 0:
                # Generate train
                train_sample = torch.cat((masked_imgs.data, gen_imgs.data, imgs.data), -2)
                save_image(train_sample, "images/train_images/%d.png" % (batches_done / Config['sample_interval']),
                           nrow=8, normalize=True)
                # Generate test
                save_sample(batches_done, dataloaders, Tensor, generator)

            running_loss1 += d_loss_local.item() * imgs.size(0)
            running_loss2 += d_loss_global.item() * imgs.size(0)
            running_loss3 += g_adv_local.item() * imgs.size(0)
            running_loss4 = g_adv_global.item() * imgs.size(0)
            running_loss5 += g_l1.item() * imgs.size(0)
            running_loss6 += g_pixel.item() * imgs.size(0)

        torch.save(generator, f'model/generator_model_{epoch}.pth')
        torch.save(global_discriminator, f'model/global_discriminator_{epoch}.pth')
        torch.save(local_discriminator, f'model/local_discriminator_{epoch}.pth')
        ########## loss record ##################
        epoch_loss1 = running_loss1 / dataset_size['train']
        epoch_loss2 = running_loss2 / dataset_size['train']
        epoch_loss3 = running_loss3 / dataset_size['train']
        epoch_loss4 = running_loss4 / dataset_size['train']
        epoch_loss5 = running_loss5 / dataset_size['train']
        epoch_loss6 = running_loss6 / dataset_size['train']

        print(
            "[D local loss: %f] [D global loss: %f] [G local adv: %f, G global adv: %f, L1: %f, pixel: %f]"
            % (epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5, epoch_loss6)
        )
        loss_dict['d_loss_local'].append(epoch_loss1)
        loss_dict['d_loss_global'].append(epoch_loss2)
        loss_dict['g_adv_local'].append(epoch_loss3)
        loss_dict['g_adv_global'].append(epoch_loss4)
        loss_dict['g_l1'].append(epoch_loss5)
        loss_dict['g_pixel'].append(epoch_loss6)


        ###################  Save Result ###################
        with open('result/result_{}.json'.format(time.strftime("%Y_%m_%d_%H", time.localtime(time.time()))), 'w') as f:
            json.dump(loss_dict, f)




if __name__ == '__main__':
    dataloaders, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    L1_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()



    # Initialize generator and discriminator
    generator = Generator(channels=Config['channels'])
    global_discriminator = Global_Discriminator(channels=Config['channels'])
    local_discriminator = Local_Discriminator(channels=Config['channels'])

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=Config['lr'],
                                   betas=(Config['b1'], Config['b2']))
    optimizer_Dg = torch.optim.Adam(global_discriminator.parameters(),
                                   lr=Config['lr'],
                                   betas=(Config['b1'], Config['b2']))
    optimizer_Dl = torch.optim.Adam(local_discriminator.parameters(),
                                   lr=Config['lr'],
                                   betas=(Config['b1'], Config['b2']))


    train_model(dataloaders, dataset_size, adversarial_loss, L1_loss, pixelwise_loss,
                generator, global_discriminator, local_discriminator,
                optimizer_G, optimizer_Dg, optimizer_Dl, num_epochs=Config['num_epochs'])



