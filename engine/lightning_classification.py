import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
import torch.optim.lr_scheduler


class LitClassification(pl.LightningModule):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__()

        self.learning_rate = args.lr

        # hyperparameters
        self.args = args
        #self.hparams = args
        hparams = {x:vars(args)[x] for x in vars(args).keys() if x not in args.not_tracking_hparams}
        hparams.pop('not_tracking_hparams', None)
        self.hparams.update(hparams)
        print(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.best_auc = 0

        # adding data
        self.train_dataloader = train_loader
        self.eval_dataloader = eval_loader

        # adding training components
        self.net = net
        self.loss_function = loss_function
        self.get_metrics = metrics
        self.optimizer = self.configure_optimizers()

        self.epoch = 0

        # parameters to optimize
        if self.args.legacy:
            for param in self.net.module.par_freeze:
                param.requires_grad = False
        else:
            for param in self.net.par_freeze:
                param.requires_grad = False
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))

        # Begin of training
        self.tini = time.time()
        self.all_label = []
        self.all_out = []
        self.best_loss = np.inf
        self.all_loss = []

    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        if self.args.op == 'adams':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.op == 'sgd':
            if self.args.legacy:
                par_freeze = set(self.net.module.par_freeze)
            else:
                par_freeze = set(self.net.par_freeze)
            optimizer = torch.optim.SGD(list(set(self.net.parameters()) - par_freeze),
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay)

            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        #if self.args['legacy']:
        #    params = list(set(self.net.parameters()) - set(self.net.module.par_freeze))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'best_auc': 0})

    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs = batch['img']
        labels = batch['labels']

        if self.args.legacy:
            imgs[0] = imgs[0].cuda()
            imgs[1] = imgs[1].cuda()
            labels[0] = labels[0].cuda()

        # repeat part
        if len(imgs) == 2:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)
        elif len(imgs) == 4:
            imgs = (torch.cat([imgs[1], imgs[0], imgs[0]], 1), torch.cat([imgs[3], imgs[2], imgs[2]], 1))
        elif len(imgs) == 6:
            imgs = (torch.cat([imgs[0], imgs[1], imgs[2]], 1), torch.cat([imgs[3], imgs[4], imgs[5]], 1))

        output = self.net(imgs)

        loss, _ = self.loss_function(output, labels)

        if not self.args.legacy:
            self.log('train_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx=0):
        imgs = batch['img']
        labels = batch['labels']

        if self.args.legacy:
            imgs[0] = imgs[0].cuda()
            imgs[1] = imgs[1].cuda()
            labels[0] = labels[0].cuda()

        # repeat part
        if len(imgs) == 2:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)
        elif len(imgs) == 4:
            imgs = (torch.cat([imgs[1], imgs[0], imgs[0]], 1), torch.cat([imgs[3], imgs[2], imgs[2]], 1))
        elif len(imgs) == 6:
            imgs = (torch.cat([imgs[0], imgs[1], imgs[2]], 1), torch.cat([imgs[3], imgs[4], imgs[5]], 1))

        output = self.net(imgs)

        loss, _ = self.loss_function(output, labels)
        if not self.args.legacy:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # metrics
        self.all_label.append(labels[0].cpu())
        self.all_out.append(output[0].cpu().detach())
        self.all_loss.append(loss.detach().cpu().numpy())

        return loss

    def validation_epoch_end(self, x):
        all_out = torch.cat(self.all_out, 0)
        all_label = torch.cat(self.all_label, 0)
        metrics = self.get_metrics(all_label, all_out)

        auc = torch.from_numpy(np.array(metrics)).cuda()
        if not self.args.legacy:
            for i in range(len(auc)):
                self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_label = []
        self.all_out = []
        self.tini = time.time()

        self.all_loss = np.mean(self.all_loss)
        print(self.all_loss)

        # saving checkpoints
        if 0:
            if (self.all_loss < self.best_loss) and (self.epoch >= 2):
                self.best_loss = self.all_loss
                print(self.best_loss)
                if not self.args.legacy:
                    self.log('best_auc', auc[0])
                file_name = os.path.join('checkpoints', self.args.prj, (str(self.epoch) + '_{:.3f}.pth').format(metrics[0]))
                torch.save(self.net, file_name)
                print('save model at: ' + file_name)
        else:
            if self.epoch % 10 == 0:
                file_name = os.path.join('checkpoints', self.args.prj, (str(self.epoch) + '_{:.3f}.pth').format(metrics[0]))
                torch.save(self.net, file_name)

        self.all_loss = []
        self.epoch += 1
        return metrics

    """ Legacy Pytorch Code """
    def training_loop(self, train_dataloader):
        self.net.train(mode=True)
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            loss = self.training_step(batch=batch)
            loss.backward()
            epoch_loss += loss
            if i % self.args.batch_update == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return epoch_loss / i

    def eval_loop(self, eval_dataloader):
        self.net.train(mode=False)
        self.net.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                loss = self.validation_step(batch=batch)
                epoch_loss += loss
            metrics = self.validation_epoch_end(x=None)
            return epoch_loss / i, metrics

    def overall_loop(self):
        for epoch in range(self.args.epochs):
            tini = time.time()
            train_loss = self.training_loop(self.train_dataloader)
            with torch.no_grad():
                eval_loss, eval_metrics = self.eval_loop(self.eval_dataloader)

            print_out = {
                'Epoch: {}': [epoch],
                'Time: {:.2f} ': [time.time() - tini],
                'Train Loss: ' + '{:.4f} ': [train_loss],
                'Loss (T/V): ' + '{:.4f} ': [eval_loss],
                'Acc: ' + '{:.4f} ' * len(eval_metrics): eval_metrics,
            }
            print(' '.join(print_out.keys()).format(*[j for i in print_out.values() for j in i]))