import os
import torch
import torch.nn as nn

import lightning.pytorch as pl

from sklearn.metrics import accuracy_score
from easydict import EasyDict

from utils.loss import CenterLoss


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: EasyDict):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator, self.discriminator = model

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.adversarial_loss = nn.BCELoss()

        self.args = args

    def forward(self, x):
        features, logits, representation = self.generator(x)

        return features, logits, representation

    def get_CE_loss(self, logits, label):
        return self.ce_loss(logits, label)

    def get_adv_loss(self, source_domain_logits):
        adv_loss = torch.mean((source_domain_logits - 1) ** 2) / 2

        return adv_loss

    def get_domain_loss(self, source_domain_logits, target_domain_logits):
        domain_loss = (
            torch.mean((target_domain_logits - 1) ** 2)
            + torch.mean(source_domain_logits**2)
        ) / 2

        return domain_loss

    def training_step(self, batch, batch_idx):
        (source_data, source_label), (target_data, target_label) = batch

        (optimizer_d, optimizer_g) = self.optimizers()

        """ Use of generator model """
        (source_features, source_class_logits, _) = self(source_data)
        (target_features, target_class_logits, target_representation) = self(
            target_data
        )
        """ Use of discriminator model """
        (_, source_domain_logits) = self.discriminator(source_features)
        (_, target_domain_logits) = self.discriminator(target_features)

        self.toggle_optimizer(optimizer_d)
        """ adv loss for discriminator """
        domain_loss = self.get_domain_loss(source_domain_logits, target_domain_logits)

        self.manual_backward(domain_loss, retain_graph=True)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)

        """ CE loss for classifier """
        source_class_loss = self.get_CE_loss(source_class_logits, source_label)
        target_class_loss = self.get_CE_loss(target_class_logits, target_label)

        """ adv loss for feature_extractor """
        adv_loss = self.get_adv_loss(source_domain_logits.detach())

        """ center loss for feature_extractor on target representation """
        if self.current_epoch == 0:
            self.center_loss = CenterLoss(
                num_classes=self.args.num_classes,
                feat_dim=target_representation.size(1),
            )
        center_loss = self.center_loss(target_representation, target_label)
        loss = (
            self.args.w_adv * adv_loss
            + (self.args.w_s * source_class_loss + self.args.w_t * target_class_loss)
            + self.args.w_ct * center_loss
        )

        self.manual_backward(loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        source_pred = torch.argmax(source_class_logits, dim=1)
        source_acc = accuracy_score(
            source_pred.cpu().numpy(), source_label.cpu().numpy()
        )

        target_pred = torch.argmax(target_class_logits, dim=1)
        target_acc = accuracy_score(
            target_pred.cpu().numpy(), target_label.cpu().numpy()
        )

        self.log(
            "loss/ce_s",
            source_class_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/ce_t",
            target_class_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/adv_loss",
            adv_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/domain_loss",
            domain_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/center_loss",
            center_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/source/acc",
            source_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/target/acc",
            target_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        data, label = batch
        _, logits, _ = self(data)
        loss = self.ce_loss(logits, label)

        prediction = torch.argmax(logits, dim=1)

        acc = accuracy_score(prediction.cpu().numpy(), label.cpu().numpy())

        self.log(
            "test/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        ckpt_path = f"models/ckpt/drda/{self.args.CKPT_NAME}"
        os.makedirs(ckpt_path, exist_ok=True)
        if self.args.SAVE_CKPT:
            torch.save(
                self.generator.state_dict(),
                f"{ckpt_path}/generator_s{self.args.subject_num}.pth",
            )
            torch.save(
                self.discriminator.state_dict(),
                f"{ckpt_path}/discriminator_s{self.args.subject_num}.pth",
            )

    def configure_optimizers(self):
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return [self.d_optimizer, self.g_optimizer]
