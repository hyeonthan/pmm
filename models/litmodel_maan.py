import os
import torch
import torch.nn as nn

import lightning.pytorch as pl

from itertools import chain
from sklearn.metrics import accuracy_score
from easydict import EasyDict


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: EasyDict):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = model

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.adversarial_loss = nn.BCELoss()
        self.args = args

    def forward(self, x):
        return self.model(x)

    def get_ce_loss(self, logits, label):
        return self.ce_loss(logits, label)

    def get_discirminator_loss(self, source_domain_prob, target_domain_prob):
        source_batch_size = source_domain_prob.size(0)
        target_batch_size = target_domain_prob.size(0)

        d_s_label = torch.cat(
            (
                torch.ones(source_batch_size, 1),
                torch.zeros(source_batch_size, 1),
            ),
            dim=1,
        ).type_as(source_domain_prob)

        d_t_label = torch.cat(
            (
                torch.zeros(target_batch_size, 1),
                torch.ones(target_batch_size, 1),
            ),
            dim=1,
        ).type_as(target_domain_prob)

        d_s_loss = self.bce_loss(source_domain_prob, d_s_label)
        d_t_loss = self.bce_loss(target_domain_prob, d_t_label)
        self.log(
            "loss/d_s_loss",
            d_s_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/d_t_loss",
            d_t_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        d_loss = (d_s_loss + d_t_loss) / 2

        return d_loss

    def training_step(self, batch, batch_idx):
        (source_data, source_label), (target_data, target_label) = batch

        source_class_logits, soruce_domain_prob = self(source_data)
        target_class_logits, target_domain_prob = self(target_data)

        (optimizer_d, optimizer_c) = self.optimizers()

        self.toggle_optimizer(optimizer_c)
        optimizer_c.zero_grad()

        """ CE loss for classifier """
        class_loss = self.get_ce_loss(source_class_logits, source_label)

        self.manual_backward(class_loss, retain_graph=True)
        optimizer_c.step()
        self.untoggle_optimizer(optimizer_c)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()

        """ Discriminator loss for discriminator """
        discriminator_loss = self.get_discirminator_loss(
            soruce_domain_prob, target_domain_prob
        )

        self.manual_backward(discriminator_loss, retain_graph=True)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        loss = class_loss - self.args.alpha * discriminator_loss

        source_pred = torch.argmax(source_class_logits, dim=1)
        source_acc = accuracy_score(
            source_pred.cpu().numpy(), source_label.cpu().numpy()
        )

        target_pred = torch.argmax(target_class_logits, dim=1)
        target_acc = accuracy_score(
            target_pred.cpu().numpy(), target_label.cpu().numpy()
        )

        self.log(
            "loss/class_loss",
            class_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/discriminator_loss",
            discriminator_loss.item(),
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
        logits, _ = self(data)
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
        ckpt_path = f"models/ckpt/maan/{self.args.CKPT_NAME}"
        os.makedirs(ckpt_path, exist_ok=True)
        if self.args.SAVE_CKPT:
            if self.args.is_apply:
                torch.save(
                    self.model.pmm_module.state_dict(),
                    f"{ckpt_path}/rsmm_s{self.args.subject_num}.pth",
                )
            torch.save(
                self.model.state_dict(),
                f"{ckpt_path}/model_s{self.args.subject_num}.pth",
            )

    def configure_optimizers(self):
        self.optimizer_d = torch.optim.AdamW(
            self.model.discriminator.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )
        self.optimizer_c = torch.optim.AdamW(
            chain(
                self.model.pmm_module.parameters(),
                self.model.feature_extractor.parameters(),
                self.model.classifier.parameters(),
            ),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return [self.optimizer_d, self.optimizer_c]
