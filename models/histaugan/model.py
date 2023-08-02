from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.histaugan import networks

# from .networks import Discriminator


class HistAuGAN(pl.LightningModule):
    def __init__(self, opts, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.opts = opts
        self.dim_attribute = 8

        # manual optimization for more advanced training procedure
        self.automatic_optimization = False

        # initialize networks
        self.dis1 = networks.Discriminator(opts.input_dim, c_dim=opts.num_domains, image_size=216)
        self.dis2 = networks.Discriminator(opts.input_dim, c_dim=opts.num_domains, image_size=216)
        self.dis_c = networks.DiscriminatorContent(c_dim=opts.num_domains)
        self.enc_c = networks.EncoderContent(opts.input_dim)
        self.enc_a = networks.EncoderAttribute(opts.input_dim, output_nc=self.dim_attribute, c_dim=opts.num_domains,
                                               nl_layer=networks.get_non_linearity(layer_type='lrelu'))
        self.gen = networks.Generator(opts.input_dim, c_dim=opts.num_domains, nz=self.dim_attribute)
        self.gen = networks.Generator(opts.input_dim, c_dim=opts.num_domains, nz=self.dim_attribute)

        # initialize network weights
        self.dis1.apply(networks.gaussian_weights_init)
        self.dis2.apply(networks.gaussian_weights_init)
        self.dis_c.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)

    def forward(self, x: torch.Tensor) -> Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        image, domain = x
        assert image.size(0) % 2 == 0, "Even batch size is required"

        half_size = image.size(0) // 2
        real_a = image[0:half_size]
        real_b = image[half_size:]
        domain_a = domain[0:half_size]
        domain_b = domain[half_size:]

        # get encoded z_c
        real_img = torch.cat((real_a, real_b), 0)
        z_content = self.enc_c.forward(real_img)
        z_content_a, z_content_b = torch.split(z_content, half_size, dim=0)

        # get encoded z_a
        mu, log_var = self.enc_a.forward(real_img, domain)
        std = log_var.mul(0.5).exp_()
        eps = torch.randn(std.size(0), std.size(1)).to(image.device)
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, half_size, dim=0)

        # get random z_a
        z_random = torch.randn(half_size, self.dim_attribute).to(image.device)

        # first cross translation
        input_content_forA = torch.cat((z_content_b, z_content_a, z_content_b), 0)
        input_content_forB = torch.cat((z_content_a, z_content_b, z_content_a), 0)
        input_attr_forA = torch.cat((z_attr_a, z_attr_a, z_random), 0)
        input_attr_forB = torch.cat((z_attr_b, z_attr_b, z_random), 0)
        input_domain_forA = torch.cat((domain_a, domain_a, domain_a), 0)
        input_domain_forB = torch.cat((domain_b, domain_b, domain_b), 0)
        output_fakeA = self.gen.forward(input_content_forA, input_attr_forA, input_domain_forA)
        output_fakeB = self.gen.forward(input_content_forB, input_attr_forB, input_domain_forB)
        fake_A_encoded, fake_AA_encoded, fake_A_random = torch.split(output_fakeA, z_content_a.size(0), dim=0)
        fake_B_encoded, fake_BB_encoded, fake_B_random = torch.split(output_fakeB, z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c
        fake_encoded_img = torch.cat((fake_A_encoded, fake_B_encoded), 0)
        z_content_recon = self.enc_c.forward(fake_encoded_img)
        z_content_recon_b, z_content_recon_a = torch.split(z_content_recon, half_size, dim=0)

        # get reconstructed encoded z_a
        mu_recon, log_var_recon = self.enc_a.forward(fake_encoded_img, domain)
        std_recon = log_var_recon.mul(0.5).exp_()
        eps_recon = torch.randn(std_recon.size(
            0), std_recon.size(1)).to(image.device)
        z_attr_recon = eps_recon.mul(std_recon).add_(mu_recon)
        z_attr_recon_a, z_attr_recon_b = torch.split(z_attr_recon, half_size, dim=0)

        # second cross translation
        fake_A_recon = self.gen.forward(z_content_recon_a, z_attr_recon_a, domain_a)
        fake_B_recon = self.gen.forward(z_content_recon_b, z_attr_recon_b, domain_b)

        # for latent regression with random attribute
        fake_random_img = torch.cat((fake_A_random, fake_B_random), 0)
        with torch.no_grad():
            mu_random, _ = self.enc_a.forward(fake_random_img, domain)
        mu_random_a, mu_random_b = torch.split(mu_random, half_size, 0)

        return z_content, z_random, mu, log_var, fake_encoded_img, fake_AA_encoded, fake_BB_encoded, fake_A_recon, \
            fake_B_recon, fake_random_img, mu_random_a, mu_random_b

    # manual training step
    def training_step(self, batch, batch_idx):
        dis_c_opt, dis1_opt, dis2_opt, enc_c_opt, enc_a_opt, gen_opt = self.optimizers()

        image, domain = batch

        # log training images every opts.log_train_img_freq iterations
        half_size = image.size(0) // 2
        if batch_idx % self.opts.log_train_img_freq == 0:
            self.log_images(image[:1], image[half_size:half_size + 1], domain[:1], domain[half_size:half_size + 1],
                            'train')
            self.log_translated_images(image[:1], domain[:1], 'train', random_attr=True)

        # update D, Ec, Ea, and G every d_iter iterations and Dc else
        if (batch_idx + 1) % self.opts.d_iter != 0:
            # update Dc
            z_content = self.enc_c.forward(image)
            dis_c_opt.zero_grad()
            pred_cls = self.dis_c.forward(z_content.detach())
            loss_dis_c = F.binary_cross_entropy_with_logits(pred_cls, domain)
            loss_dis_c.backward()
            nn.utils.clip_grad_norm_(self.dis_c.parameters(), 5)
            dis_c_opt.step()

            self.log_dict({'loss_dis_c': loss_dis_c}, prog_bar=True)

        else:
            # run forward pass once in training step
            z_content, z_random, mu, log_var, fake_encoded_img, fake_AA_encoded, fake_BB_encoded, fake_A_recon, \
                fake_B_recon, fake_random_img, mu_random_a, mu_random_b = self.forward(batch)

            # update D
            dis1_opt.zero_grad()
            loss_dis1 = self.dis_loss(self.dis1, 1, image, fake_encoded_img.detach(), domain)
            loss_dis1.backward()
            dis1_opt.step()

            dis2_opt.zero_grad()
            loss_dis2 = self.dis_loss(self.dis2, 2, image, fake_random_img.detach(), domain)
            loss_dis2.backward()
            dis2_opt.step()

            # update enc_c, enc_a, gen
            enc_c_opt.zero_grad()
            enc_a_opt.zero_grad()
            gen_opt.zero_grad()

            loss_gen = self.gen_loss(
                image, domain, z_content, mu, log_var, fake_encoded_img, fake_AA_encoded, fake_BB_encoded, fake_A_recon,
                fake_B_recon, fake_random_img, mu_random_a, mu_random_b, z_random,
            )
            loss_gen = sum(loss_gen)
            loss_gen.backward()

            enc_a_opt.step()
            enc_c_opt.step()
            gen_opt.step()

            self.log_dict({
                'gen_loss': loss_gen,
                'dis_loss/1': loss_dis1,
                'dis_loss/2': loss_dis2,
            }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        image, domain = batch

        # run forward pass once in training step
        z_content, z_random, mu, log_var, fake_encoded_img, fake_AA_encoded, fake_BB_encoded, fake_A_recon, \
            fake_B_recon, fake_random_img, mu_random_a, mu_random_b = self.forward(batch)

        # log metric
        # -- self and cross-cycle recon
        loss_l1_cc = torch.mean(torch.abs(
            image - torch.cat((fake_A_recon, fake_B_recon), 0))) * self.opts.lambda_rec

        self.log_dict({
            'l1_cc_loss/val': loss_l1_cc,
        })

        # log images
        half_size = image.size(0) // 2
        if batch_idx % self.opts.log_val_img_freq == 0:
            self.log_images(image[:1], image[half_size:half_size + 1], domain[:1], domain[half_size:half_size + 1],
                            'val', batch_idx)
            self.log_translated_images(image[:1], domain[:1], 'val', random_attr=True, batch_idx=batch_idx)

    def dis_loss(self, dis, dis_id, image, fake_img, domain):
        # calculate loss for one discriminator
        pred_fake, pred_fake_cls = dis.forward(fake_img)
        pred_real, pred_real_cls = dis.forward(image)

        label_fake = torch.zeros_like(pred_fake)
        label_real = torch.ones_like(pred_real)

        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)
        loss_real = F.binary_cross_entropy_with_logits(pred_real, label_real)

        loss_dis_adv = (loss_fake + loss_real) / 2
        loss_dis_cls = F.binary_cross_entropy_with_logits(pred_real_cls, domain)
        loss_dis = loss_dis_adv + self.opts.lambda_cls * loss_dis_cls

        self.log_dict({
            f'loss_dis_adv/{dis_id}': loss_dis_adv,
            f'loss_dis_cls/{dis_id}': loss_dis_cls,
        })

        return loss_dis

    def gen_loss(self, image, domain, z_content, mu, log_var, fake_encoded_img, fake_aa_encoded, fake_bb_encoded,
                 fake_a_recon, fake_b_recon, fake_random_img, mu_random_a, mu_random_b, z_random):
        # -- content adversarial loss for generator
        pred_cls = self.dis_c.forward(z_content)
        loss_adv_content = F.binary_cross_entropy_with_logits(pred_cls, 1 - domain)  # loss_G_GAN_content

        # -- adversarial loss for generator from dis1
        pred_fake, pred_fake_cls = self.dis1.forward(fake_encoded_img)
        label_real = torch.ones_like(pred_fake)
        loss_adv1 = F.binary_cross_entropy_with_logits(pred_fake, label_real)
        # -- classification (from dis1) (loss_G_cls)
        loss_adv1_cls = F.binary_cross_entropy_with_logits(pred_fake_cls, domain) * self.opts.lambda_cls_G

        # -- self and cross-cycle recon
        loss_l1_self_recon = torch.mean(torch.abs(
            image - torch.cat((fake_aa_encoded, fake_bb_encoded), 0))) * self.opts.lambda_rec
        loss_l1_cc = torch.mean(torch.abs(
            image - torch.cat((fake_a_recon, fake_b_recon), 0))) * self.opts.lambda_rec

        # -- KL loss - z_c
        loss_kl_zc = self._l2_regularize(z_content) * 0.01

        # -- KL loss - z_a
        kl_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        loss_kl_za = (0.01 * torch.sum(kl_element).mul_(-0.5)) / (kl_element.shape[0])

        # -- adversarial loss for generator from dis2
        pred_fake, pred_fake_cls = self.dis2.forward(fake_random_img)
        label_real = torch.ones_like(pred_fake)
        loss_adv2 = F.binary_cross_entropy_with_logits(pred_fake, label_real)
        # -- classification (from dis2) (loss_G_cls)
        loss_adv2_cls = F.binary_cross_entropy_with_logits(pred_fake_cls, domain) * self.opts.lambda_cls_G

        # -- latent regression loss
        loss_z_L1_a = torch.mean(torch.abs(mu_random_a - z_random)) * 10
        loss_z_L1_b = torch.mean(torch.abs(mu_random_b - z_random)) * 10
        loss_z_L1 = loss_z_L1_a + loss_z_L1_b

        self.log_dict({
            'loss_adv_content': loss_adv_content,
            'loss_adv/1': loss_adv1,
            'loss_adv_cls/1': loss_adv1_cls,
            'loss_adv/2': loss_adv2,
            'loss_adv_cls/2': loss_adv2_cls,
            'l1_self_rec_loss': loss_l1_self_recon,
            'l1_cc_loss': loss_l1_cc,
            'kl_loss_zc': loss_kl_zc,
            'kl_loss_za': loss_kl_za,
            'l1_latent_loss': loss_z_L1,
        })

        return loss_adv_content, loss_adv1, loss_adv1_cls, loss_l1_self_recon, loss_l1_cc, loss_kl_zc, loss_kl_za, \
            loss_adv2, loss_adv2_cls, loss_z_L1

    def log_images(self, real_a, real_b, domain_a, domain_b, phase, batch_idx=0):
        # log cross-cycle translation

        # get encoded z_c
        real_img = torch.cat((real_a, real_b), 0)
        domain = torch.cat((domain_a, domain_b), dim=0)
        z_content = self.enc_c.forward(real_img)
        z_content_a, z_content_b = torch.split(z_content, 1, dim=0)

        # get encoded z_a
        mu, log_var = self.enc_a.forward(real_img, domain)
        std = log_var.mul(0.5).exp_()
        eps = torch.randn(std.size(0), std.size(1)).to(real_img.device)
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, 1, dim=0)

        # get random z_a
        z_random = torch.randn(1, self.dim_attribute).to(real_img.device)

        # first cross translation
        input_content_forA = torch.cat((z_content_b, z_content_a, z_content_b), 0)
        input_content_forB = torch.cat((z_content_a, z_content_b, z_content_a), 0)
        input_attr_forA = torch.cat((z_attr_a, z_attr_a, z_random), 0)
        input_attr_forB = torch.cat((z_attr_b, z_attr_b, z_random), 0)
        input_domain_forA = torch.cat((domain_a, domain_a, domain_a), 0)
        input_domain_forB = torch.cat((domain_b, domain_b, domain_b), 0)
        output_fakeA = self.gen.forward(
            input_content_forA, input_attr_forA, input_domain_forA)
        output_fakeB = self.gen.forward(
            input_content_forB, input_attr_forB, input_domain_forB)
        fake_A_encoded, fake_AA_encoded, fake_A_random = torch.split(output_fakeA, z_content_a.size(0), dim=0)
        fake_B_encoded, fake_BB_encoded, fake_B_random = torch.split(output_fakeB, z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c
        fake_encoded_img = torch.cat((fake_A_encoded, fake_B_encoded), 0)
        z_content_recon = self.enc_c.forward(fake_encoded_img)
        z_content_recon_b, z_content_recon_a = torch.split(z_content_recon, 1, dim=0)

        # get reconstructed encoded z_a
        mu_recon, log_var_recon = self.enc_a.forward(fake_encoded_img, domain)
        std_recon = log_var_recon.mul(0.5).exp_()
        eps_recon = torch.randn(std_recon.size(
            0), std_recon.size(1)).to(real_img.device)
        z_attr_recon = eps_recon.mul(std_recon).add_(mu_recon)
        z_attr_recon_a, z_attr_recon_b = torch.split(z_attr_recon, 1, dim=0)

        # second cross translation
        fake_A_recon = self.gen.forward(z_content_recon_a, z_attr_recon_a, domain_a)
        fake_B_recon = self.gen.forward(z_content_recon_b, z_attr_recon_b, domain_b)

        # original, first cc, first cc (random attribute), first cc (self-recon), second cc
        img = torch.cat((
            torch.cat((real_a.detach().cpu(), fake_A_encoded[:1].detach().cpu(), fake_A_random[:1].detach().cpu(),
                       fake_AA_encoded[:1].detach().cpu(), fake_A_recon[:1].detach().cpu()), dim=3),
            torch.cat((real_b.detach().cpu(), fake_B_encoded[:1].detach().cpu(), fake_B_random[:1].detach().cpu(),
                       fake_BB_encoded[:1].detach().cpu(), fake_B_recon[:1].detach().cpu()), dim=3)
        ), dim=2)

        step = self.global_step + batch_idx if phase == 'val' else self.global_step
        self.logger.log_image(key=f'cross-cycle/{phase}', images=[img, ], step=step)

    def log_translated_images(self, image, domain, phase, random_attr=False, batch_idx=0):
        # get z_content
        z_content = self.enc_c.forward(image)
        # z_content = z_content.repeat(self.opts.num_domains, 1, 1, 1)

        # get z_attr
        if not random_attr:
            mu, log_var = self.enc_a.forward(image, domain)
            std = log_var.mul(0.5).exp_()
            eps = torch.randn((std.size(0), std.size(1))).to(image.device)
            z_attr = eps.mul(std).add_(mu)
        else:
            z_attr = torch.randn((7, 8,)).to(image.device)
        # z_attr = z_attr.repeat(self.opts.num_domains, 1)

        # generate new histology image with same content as img
        new_domains = F.one_hot(torch.arange(self.opts.num_domains), num_classes=self.opts.num_domains).to(image.device)
        # out = self.gen(z_content, z_attr, new_domains).detach().squeeze(0)  # in range [-1, 1]
        # grid = torchvision.utils.make_grid(torch.cat((image, out), dim=0), normalize=True, range=(-1, 1))

        translated_images = []
        for i in range(self.opts.num_domains):
            z_attr = torch.randn((1, 8,)).to(image.device)
            out = self.gen(z_content, z_attr, new_domains[i].unsqueeze(0)).detach()
            translated_images.append(out)

        img = torch.cat((image, *translated_images), dim=3)
        # grid = torchvision.utils.make_grid(
        #     torch.cat((image, *translated_images), dim=0), normalize=True, range=(-1, 1))
        step = self.global_step + batch_idx if phase == 'val' else self.global_step
        # self.logger.experiment.add_image(f'translated_images/{phase}', grid, global_step=step)
        # self.logger.experiment.add_image(f'translated_images/{phase}', (img / 2 + 0.5).squeeze(0), global_step=step)
        self.logger.log_image(key=f'translated_images/{phase}', images=[img, ], step=step)

    def configure_optimizers(self):
        dis_c_opt = torch.optim.Adam(self.dis_c.parameters(
        ), lr=self.hparams.learning_rate / 2.5, betas=(0.5, 0.999), weight_decay=0.0001)
        dis1_opt = torch.optim.Adam(self.dis1.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        dis2_opt = torch.optim.Adam(self.dis2.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        enc_c_opt = torch.optim.Adam(self.enc_c.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        enc_a_opt = torch.optim.Adam(self.enc_a.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        gen_opt = torch.optim.Adam(self.gen.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)

        dis_c_sch = networks.get_scheduler(dis_c_opt, self.opts)
        dis1_sch = networks.get_scheduler(dis1_opt, self.opts)
        dis2_sch = networks.get_scheduler(dis2_opt, self.opts)
        enc_c_sch = networks.get_scheduler(enc_c_opt, self.opts)
        enc_a_sch = networks.get_scheduler(enc_a_opt, self.opts)
        gen_sch = networks.get_scheduler(gen_opt, self.opts)

        return [dis_c_opt, dis1_opt, dis2_opt, enc_c_opt, enc_a_opt, gen_opt], [dis_c_sch, dis1_sch, dis2_sch,
                                                                                enc_c_sch, enc_a_sch, gen_sch]

    @staticmethod
    def _l2_regularize(mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
