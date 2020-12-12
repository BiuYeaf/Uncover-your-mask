from torchvision.utils import save_image
from torch.autograd import Variable
from model_v3 import *
from utils import Config
from data import get_dataloader
import torch
from tqdm import tqdm
import json
import time

def save_sample(batches_done, dataloaders, Tensor, generator):
    masked_samples, samples = next(iter(dataloaders['test']))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    # Generate inpainted image
    generator.eval()
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, 128: , 64 : 64 + Config['mask_size']] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/test_images/%d.png" % (batches_done/Config['sample_interval']), nrow=8, normalize=True)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_model(dataloaders, dataset_size, patch, adversarial_loss, L1_loss, pixelwise_loss,
                generator, discriminator, optimizer_G, optimizer_D, num_epochs):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        L1_loss.cuda()
        pixelwise_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # ----------
    #  Training
    # ----

    loss_dict = {'d_loss': [],
                 'g_adv': [],
                 'g_l1': [],
                 'g_pixel': []}    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)
        
        running_loss1, running_loss2, running_loss3, running_loss4 = 0.0, 0.0, 0.0, 0.0
        for i, (masked_imgs, imgs) in tqdm(enumerate(dataloaders['train'])):
            masked_parts = imgs[:, :, 128:, 64:192]

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))
            # masked_parts = Variable(masked_parts.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked_imgs)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_l1 = L1_loss(gen_parts, masked_parts)
            g_pixel = pixelwise_loss(torch.sigmoid(gen_parts), masked_parts*0.5+0.5)
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_l1 + 0.005 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples

            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # print(
            #     "[Batch %d/%d] [D loss: %f] [G adv: %f, L1: %f, pixel: %f]"
            #     % (i, len(dataloaders['test']), d_loss.item(), g_adv.item(), g_l1.item(), g_pixel.item())
            # )

            # Generate sample at sample interval
            batches_done = epoch * len(dataloaders['train']) + i
            if batches_done % Config['sample_interval'] == 0:
                # Generate train
                filled_imgs = masked_imgs.clone()
                filled_imgs[:, :, 128: , 64: 64 + Config['mask_size']] = gen_parts
                train_sample = torch.cat((masked_imgs.data, filled_imgs.data, imgs.data), -2)
                save_image(train_sample, "images/train_images/%d.png" % (batches_done/Config['sample_interval']), nrow=8, normalize=True)
                # Generate test
                save_sample(batches_done, dataloaders, Tensor, generator)

            running_loss1 += d_loss.item() * imgs.size(0)
            running_loss2 += g_adv.item() * imgs.size(0)
            running_loss3 += g_l1.item() * imgs.size(0)
            running_loss4 += g_pixel.item() * imgs.size(0)

        torch.save(generator, 'generator_model.pth')
        torch.save(discriminator, 'discriminator.pth')
        ########## loss record ##################
        epoch_loss1 = running_loss1 / dataset_size['train']
        epoch_loss2 = running_loss2 / dataset_size['train']
        epoch_loss3 = running_loss3 / dataset_size['train']
        epoch_loss4 = running_loss4 / dataset_size['train']
        print(
            "[D loss: %f] [G adv: %f, L1: %f, pixel: %f]"
            % (epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4)
        )
        loss_dict['d_loss'].append(epoch_loss1)
        loss_dict['g_adv'].append(epoch_loss2)
        loss_dict['g_l1'].append(epoch_loss3)
        loss_dict['g_pixel'].append(epoch_loss4)
        ###################  Save Result ###################
        if epoch%10 == 0:
        with open('result/result_{}.json'.format(time.strftime("%Y_%m_%d_%H",time.localtime(time.time()))), 'w') as f:
            json.dump(loss_dict, f)

    # ###################  Save Result ###################
    #
    # with open('result/result_{}.json'.format(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time()))), 'w') as f:
    #     json.dump(loss_dict, f)



if __name__ == '__main__':
    dataloaders, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(Config['mask_size'] / 2 ** 3), int(Config['mask_size'] / 2 ** 3)
    patch = (1, patch_h, patch_w)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    pixelwise_loss = torch.nn.BCELoss()


    # Initialize generator and discriminator
    generator = Generator(channels=Config['channels'])
    discriminator = Discriminator(channels=Config['channels'])

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=Config['lr'],
                                   betas=(Config['b1'], Config['b2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=Config['lr'],
                                   betas=(Config['b1'], Config['b2']))


    train_model(dataloaders, dataset_size,patch, adversarial_loss, L1_loss, pixelwise_loss,
                generator, discriminator, optimizer_G, optimizer_D, num_epochs=Config['num_epochs'])
