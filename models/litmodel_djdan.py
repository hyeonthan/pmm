import os
import torch
import torch.nn as nn

import lightning.pytorch as pl

from itertools import chain
from sklearn.metrics import accuracy_score
from easydict import EasyDict

torch.autograd.set_detect_anomaly(True)


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

    def calculate_CE_loss(self, logits, label):
        return self.ce_loss(logits, label)

    def calculate_global_loss(self, source_marginal_logits, target_marginal_logits):
        valid = torch.cat(
            (
                torch.zeros(target_marginal_logits.size(0), 1),
                torch.ones(target_marginal_logits.size(0), 1),
            ),
            dim=1,
        )
        valid = valid.type_as(target_marginal_logits)

        global_real_loss = self.adversarial_loss(target_marginal_logits, valid)

        fake = torch.cat(
            (
                torch.ones(source_marginal_logits.size(0), 1),
                torch.zeros(source_marginal_logits.size(0), 1),
            ),
            dim=1,
        )
        fake = fake.type_as(source_marginal_logits)

        global_fake_loss = self.adversarial_loss(source_marginal_logits, fake)

        global_loss = (global_real_loss + global_fake_loss) / 2

        return global_loss

    def calculate_local_loss(
        self, source_conditional_logits, target_conditional_logits
    ):
        for i in range(len(target_conditional_logits)):
            valid = torch.cat(
                (
                    torch.zeros(target_conditional_logits[i].size(0), 1),
                    torch.ones(target_conditional_logits[i].size(0), 1),
                ),
                dim=1,
            )
            valid = valid.type_as(target_conditional_logits[i])

            local_real_loss = self.adversarial_loss(target_conditional_logits[i], valid)

            fake = torch.cat(
                (
                    torch.ones(source_conditional_logits[0].size(0), 1),
                    torch.zeros(source_conditional_logits[0].size(0), 1),
                ),
                dim=1,
            )
            fake = fake.type_as(source_conditional_logits[i])

            local_fake_loss = self.adversarial_loss(source_conditional_logits[i], fake)

            if i == 0:
                local_loss = (local_real_loss + local_fake_loss) / 2
            else:
                local_loss += (local_real_loss + local_fake_loss) / 2

        return local_loss / self.args.num_classes

    def calculate_omega(
        self,
        source_marginal_logits,
        source_conditional_logits,
        target_marginal_logits,
        target_conditional_logits,
    ):
        d_g = torch.cat((source_marginal_logits, target_marginal_logits))

        d_g_label = torch.cat(
            (
                torch.zeros(d_g.size(0), 1),
                torch.ones(d_g.size(0), 1),
            ),
            dim=1,
        ).type_as(d_g)

        d_g_loss = self.bce_loss(d_g, d_g_label)

        dA_g = 2 * (1 - 2 * d_g_loss)

        for i, (s_c_logit, t_c_logit) in enumerate(
            zip(source_conditional_logits, target_conditional_logits)
        ):
            d_c = torch.cat((s_c_logit, t_c_logit))

            d_c_label = torch.cat(
                (
                    torch.zeros(d_c.size(0), 1),
                    torch.ones(d_c.size(0), 1),
                ),
                dim=1,
            ).type_as(d_c)

            d_c_loss = self.bce_loss(d_c, d_c_label)

            if i == 0:
                dA_c = 2 * (1 - 2 * (d_c_loss))
            else:
                dA_c = dA_c + 2 * (1 - 2 * (d_c_loss))

        dA_c = dA_c / len(source_conditional_logits)

        omega = dA_g / (dA_g + dA_c)

        self.log(
            "omega/dA_g",
            dA_g,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "omega/dA_c",
            dA_c,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return omega

    def training_step(self, batch, batch_idx):
        (source_data, source_label), (target_data, target_label) = batch

        source_class_logits, source_conditional_logits, source_marginal_logits = self(
            source_data
        )
        target_class_logits, target_conditional_logits, target_marginal_logits = self(
            target_data
        )

        (optimizer_d, optimizer_c) = self.optimizers()

        self.toggle_optimizer(optimizer_c)
        optimizer_c.zero_grad()
        """ CE loss for classifier """
        class_loss = self.calculate_CE_loss(source_class_logits, source_label)

        self.manual_backward(class_loss, retain_graph=True)
        optimizer_c.step()
        self.untoggle_optimizer(optimizer_c)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        """ Global loss for marginal discriminator """
        global_loss = self.calculate_global_loss(
            source_marginal_logits, target_marginal_logits
        )

        """ Local loss for conditional discriminator """
        local_loss = self.calculate_local_loss(
            source_conditional_logits, target_conditional_logits
        )

        """ Omega (w) calculation """
        if self.current_epoch == 0:
            omega = 0.5
        else:
            omega = self.calculate_omega(
                source_marginal_logits,
                source_conditional_logits,
                target_marginal_logits,
                target_conditional_logits,
            )

        discriminator_loss = omega * global_loss + (1 - omega) * local_loss

        self.manual_backward(discriminator_loss, retain_graph=True)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.log(
            "omega/omega",
            omega,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/class_loss",
            class_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/global_loss",
            global_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/local_loss",
            local_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        alpha = self.args.alpha
        loss = class_loss + alpha * discriminator_loss

        source_pred = torch.argmax(source_class_logits, dim=1)
        source_acc = accuracy_score(
            source_pred.cpu().numpy(), source_label.cpu().numpy()
        )

        target_pred = torch.argmax(target_class_logits, dim=1)
        target_acc = accuracy_score(
            target_pred.cpu().numpy(), target_label.cpu().numpy()
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
        logits, _, _ = self(data)
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

        ckpt_path = f"models/ckpt/djdan/{self.args.CKPT_NAME}"
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
        self.d_optimizer = torch.optim.AdamW(
            chain(
                self.model.conditional_discriminator.parameters(),
                self.model.marginal_discriminator.parameters(),
            ),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )
        self.c_optimizer = torch.optim.AdamW(
            chain(
                self.model.pmm_module.parameters(),
                self.model.feature_extractor.parameters(),
                self.model.class_classifier.parameters(),
            ),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )
        return [self.d_optimizer, self.c_optimizer]
