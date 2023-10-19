
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args

from gqa_data import get_loader
from utils import LossMeter, set_global_logging_level

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


from trainer_base import TrainerBase

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        from gqa_model import FewVLMGQA

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = FewVLMGQA

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)

        print(self.model)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters
        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        print("num grad param:",count_parameters(self.model))
        print("num total elements:",sum(p.numel() for p in self.model.parameters()))

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()

            best_valid = 0.
            best_epoch = 0

        if self.args.distributed:
            dist.barrier()

        # torch.autograd.set_detect_anomaly(True)

        # print(f'GPU{self.args.gpu} before training starts')

        global_step = 0
        patience = 200 # TODO
        early_stop_counter = 0 # TODO

        if not args.test_only:
            for epoch in range(self.args.epochs):
                if self.start_epoch is not None:
                    epoch += self.start_epoch
                self.model.train()
                if self.args.distributed:
                    self.train_loader.sampler.set_epoch(epoch)
                if self.verbose:
                    pbar = tqdm(total=len(self.train_loader), ncols=120)

                epoch_results = {
                    'loss': 0.,

                }


                for step_i, batch in enumerate(self.train_loader):

                    if self.args.fp16 and _use_native_amp:
                        with autocast():
                            if self.args.distributed:
                                results = self.model.module.train_step(batch)
                            else:
                                results = self.model.train_step(batch)
                    else:
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)

                    loss = results['loss']


                    if self.args.fp16 and _use_native_amp:
                        self.scaler.scale(loss).backward()
                    elif self.args.fp16 and _use_apex:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()


                    loss = loss.detach()

                    # Update Parameters
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.clip_grad_norm)
                        elif self.args.fp16 and _use_apex:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(
                                self.optim), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                    for param in self.model.parameters():
                        param.grad = None

                    global_step += 1

                    for k, v in results.items():
                        if k in epoch_results:
                            epoch_results[k] += v.item()

                    if self.lr_scheduler:
                        if version.parse(torch.__version__) >= version.parse("1.4"):
                            lr = self.lr_scheduler.get_last_lr()[0]
                        else:
                            lr = self.lr_scheduler.get_lr()[0]
                    else:
                        try:
                            lr = self.optim.get_lr()[0]
                        except AttributeError:
                            lr = self.args.lr

                    if self.verbose:
                        loss_meter.update(loss.item())
                        desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                        desc_str += f' | Loss {loss_meter.val:4f}'

                        pbar.set_description(desc_str)
                        pbar.update(1)

                    if self.args.distributed:
                        dist.barrier()

                if self.verbose:
                    pbar.close()

                    log_str = ''

                    # Validation
                    valid_score = self.evaluate(self.val_loader) * 100.
                    if valid_score > best_valid:
                        best_valid = valid_score
                        best_epoch = epoch
                        self.save("BEST")

                    if epoch - best_epoch > patience:
                        early_stop_counter += 1
                        if early_stop_counter == 1:
                            print(f'Validation score has not improved for {patience} epochs. '
                                  'Applying early stopping...')
                        if early_stop_counter > 1:
                            print(f'Validation score has not improved for {early_stop_counter * patience} epochs. '
                                  'Applying early stopping...')
                        break
                    else:
                        early_stop_counter = 0

                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_epoch, best_valid)

                    print(log_str)

                if self.args.distributed:
                    dist.barrier()
                    
            if self.verbose:
                self.save("LAST")

        if self.verbose:

            if not self.args.test_only:
                best_path = os.path.join(self.args.output, 'BEST')
                self.load(best_path)

            dump_path = os.path.join(self.args.output, 'submit.json')
            
            self.predict(self.test_loader, dump_path=dump_path)

            test_score = self.evaluate(self.test_loader) * 100.

            print("Testdev %0.2f" % test_score)


        if self.args.distributed:
            dist.barrier()
            exit()

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            if self.args.num_beams > 1:
                gen_kwargs['num_beams'] = self.args.num_beams

            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")

            for i, batch in enumerate(loader):

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)
                
            

            if dump_path is not None:
                if not os.path.isdir(self.args.output):
                    os.makedirs(self.args.output, exist_ok=True)
                print('\nsave dump at', dump_path)
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()

if __name__ == "__main__":

    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    main_worker(args.local_rank, args)
