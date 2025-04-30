# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta,datetime

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model, best_acc, best_epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_%d_%2.5f.bin" % (args.name, best_epoch, best_acc))
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def delete_model(args, best_acc, best_epoch):
    file_path = os.path.join(args.output_dir, "%s_%d_%2.5f.bin" % (args.name, best_epoch, best_acc))
    try:
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied to delete {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB":
        num_classes = 200
    elif args.dataset == "AgriculturalDisease":
        num_classes = 61
    elif args.dataset == "AppleLeaf9":
        num_classes = 9
    elif args.dataset == "WheatData":
        num_classes = 12
    elif args.dataset == "PlantPathology":
        num_classes = 12
    elif args.dataset == "RiceLeaf":
        num_classes = 9
    elif args.dataset == "PlantLeave":
        num_classes = 39
    elif args.dataset == "PlantDoc":
        num_classes = 28
    elif args.dataset == "Cotton":
        num_classes = 80

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                              smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, epoch, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Epoch = %d", epoch)
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0]
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

        # Update progress bar with the current loss
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    val_accuracy = accuracy

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Epoch: %d" % epoch)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)

    # Log results to TensorBoard
    if args.local_rank in [-1, 0]:
        writer.add_scalar("validation/loss", scalar_value=eval_losses.avg, global_step=epoch)
        writer.add_scalar("validation/accuracy", scalar_value=val_accuracy, global_step=epoch)

    return val_accuracy



def train(args, model):
    best_epoch = 0
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    num_training_steps_per_epoch = len(train_loader)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_training_steps_per_epoch * args.num_epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_training_steps_per_epoch * args.num_epochs)

    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs)
    logger.info("  Total steps per epoch = %d", num_training_steps_per_epoch)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    datasetName = args.dataset
    result_file = f'./output/{datasetName}_results.txt'
    now = datetime.now()

    # 格式化时间为指定格式
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(result_file, 'a') as file:
        file.write("******************** Training ********************")
        file.write('\n')
        file.write("  Current times = %s" % (formatted_time))
        file.write('\n')
        file.write("  Total epochs = %d" % (args.num_epochs))
        file.write('\n')
        file.write("  Instantaneous batch size per GPU = %d" % (args.train_batch_size))
        file.write('\n')
        file.write("Total Parameter: \t%2.1fM" % count_parameters(model))
        file.write('\n')

    model.zero_grad()
    set_seed(args)  # Ensure reproducibility
    losses = AverageMeter()
    best_acc = 0
    global_step = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Epoch %d/%d" % (epoch + 1, args.num_epochs),
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            loss, logits = model(x, y)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Epoch %d/%d - Step %d/%d (loss=%2.5f)" %
                    (epoch + 1, args.num_epochs, step + 1, num_training_steps_per_epoch, losses.val)
                )

        # Compute epoch-level metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        train_accuracy = simple_accuracy(all_preds, all_label)
        logger.info("Train accuracy for epoch %d: %f" % (epoch + 1, train_accuracy))

        # Validation after each epoch
        with torch.no_grad():
            val_accuracy = valid(args, model, writer, test_loader, epoch, global_step)

        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=losses.avg, global_step=global_step)
            writer.add_scalar("train/accuracy", scalar_value=train_accuracy, global_step=global_step)
            writer.add_scalar("valid/accuracy", scalar_value=val_accuracy, global_step=global_step)

            if best_acc < val_accuracy:
                delete_model(args, best_acc, best_epoch)
                best_acc = val_accuracy
                best_epoch = epoch + 1
                save_model(args, model, best_acc, best_epoch)
                logger.info("Best valid accuracy in epoch: %d" % best_epoch)
        losses.reset()

        with open(result_file, 'a') as file:
            file.write("******************** Epoch %d/%d ********************" % (epoch + 1, args.num_epochs))
            file.write('\n')
            file.write("Train accuracy for epoch %d: %f" % (epoch + 1, train_accuracy))
            file.write('\n')
            file.write("Valid accuracy for epoch %d: %f" % (epoch + 1, val_accuracy))
            file.write('\n')
            file.write("Best valid accuracy in epoch: %d" % best_epoch)
            file.write('\n')

    writer.close()
    logger.info("Best Accuracy: 	%f" % best_acc)
    logger.info("Best Epoch: 	%d" % best_epoch)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: 	%f hours" % ((end_time - start_time) / 3600))

    with open(result_file, 'a') as file:
        file.write("******************** End Training! ********************")
        file.write('\n')
        file.write("Best Accuracy: %f" % best_acc)
        file.write('\n')
        file.write("Best Epoch: %d" % best_epoch)
        file.write('\n')
        file.write("Total Training Time: %f hours" % ((end_time - start_time) / 3600))
        file.write('\n')
        file.write("******************** End Training! ********************")
        file.write('\n')

def main(config):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["CUB", "AgriculturalDisease", "AppleLeaf9", "WheatData", "PlantPathology", "RiceLeaf",
                                 "PlantLeave", "PlantDoc", "Cotton"], default="CUB",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--resize_size", default=600, type=int,
                        help="Resolution size")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=config.batch_size, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=config.batch_size, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)


if __name__ == "__main__":
    config = CONFIGS["ViT-B_16"]
    main(config)
