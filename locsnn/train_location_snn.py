"""Train the LOC-SNN models

Codes based on https://github.com/clear-nus/VT_SNN

python locsnn/train_location_snn.py \
 --epochs 500 \
 --lr 0.001 \
 --sample_file 1 \
 --batch_size 8 \
 --fingers both \
 --network_config /path/to/correct_config.yml \
 --data_dir /path/to/preprocessed \
 --hidden_size 32 \
 --loss NumSpikes \
 --mode location \
 --task cw \
 --checkpoint_dir /path/to/checkpoint

where mode is one of {location, location_fuse, only_location, location_cat_whorl, location_cat_arch, location_cat_random} and task is {cw, slip, ycb}.
"""
from pathlib import Path
import logging
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn

from locsnn.models.snn import LocationSlayer, LocationSlayerFuse, OnlyLocationSlayer, LocationSlayerWhorl, LocationSlayerArch, LocationSlayerRandom
from locsnn.dataset import ViTacDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

parser = argparse.ArgumentParser("Train LOC-SNN models.")

parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)
parser.add_argument(
    "--network_config",
    type=str,
    help="Path SNN network configuration.",
    required=True,
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)
parser.add_argument(
    "--fingers",
    type=str,
    help="Which fingers to use for tactile data.",
    choices=["left", "right", "both"],
    required=True)
parser.add_argument(
    "--mode",
    type=str,
    choices=["location", "location_fuse", "only_location", "location_cat_whorl", "location_cat_arch", "location_cat_random"],
    help="Type of model to run.",
    required=True,
)

parser.add_argument(
    "--task",
    type=str,
    help="The classification task.",
    choices=["cw", "slip", "ycb"],
    required=True,
)

parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)

parser.add_argument(
    "--loss",
    type=str,
    help="Loss function to use.",
    choices=["NumSpikes", "WeightedNumSpikes", "WeightedLocationNumSpikes"],
    required=True,
)

args = parser.parse_args()

LOSS_TYPES = ["NumSpikes", "WeightedNumSpikes", "WeightedLocationNumSpikes"]

params = snn.params(args.network_config)

if args.task == "cw":
    output_size = 20
elif args.task == "slip":
    output_size = 2
elif args.task == "ycb":
    output_size = 36
else:
    raise ValueError("Invalid args.task")

if args.mode == "location":
    model = LocationSlayer
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "location_cat_whorl":
    model = LocationSlayerWhorl
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "location_cat_arch":
    model = LocationSlayerArch
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "location_fuse":
    model = LocationSlayerFuse
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "only_location":
    model = OnlyLocationSlayer
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "location_cat_random":
    model = LocationSlayerRandom
    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }

updated_checkpoint_dir = f"{args.checkpoint_dir}/{args.task}_{args.mode}_{args.sample_file}"
Path(updated_checkpoint_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
writer = SummaryWriter(Path(updated_checkpoint_dir))
net = model(**model_args).to(device)


if args.loss == "NumSpikes":
    params["training"]["error"]["type"] = "NumSpikes"
    error = snn.loss(params).to(device)
    criteria = error.numSpikes
elif args.loss == "WeightedNumSpikes":
    params["training"]["error"]["type"] = "WeightedNumSpikes"
    error = snn.loss(params).to(device)
    criteria = error.weightedNumSpikes
elif args.loss == "WeightedLocationNumSpikes":
    params["training"]["error"]["type"] = "WeightedLocationNumSpikes"
    error = snn.loss(params).to(device)
    criteria = error.weightedLocationNumSpikes

optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.5)

train_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"train_80_20_{args.sample_file}.txt",
    output_size=output_size,
    spiking=True,
    mode=args.mode,
    fingers=args.fingers
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)
test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=output_size,
    spiking=True,
    mode=args.mode,
    fingers=args.fingers
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
)


def _train():
    correct = 0
    num_samples = 0
    net.train()
    for *data, target, label in train_loader:
        data = [d.to(device) for d in data]
        target = target.to(device)
        output = net.forward(*data)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        num_samples += len(label)
        spike_loss = criteria(output, target)

        optimizer.zero_grad()
        spike_loss.backward()
        optimizer.step()

    writer.add_scalar("loss/train", spike_loss / len(train_loader), epoch)
    writer.add_scalar("acc/train", correct / num_samples, epoch)


def _test():
    correct = 0
    num_samples = 0
    net.eval()
    with torch.no_grad():
        for *data, target, label in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = net.forward(*data)
            correct += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            num_samples += len(label)
            spike_loss = criteria(output, target)  

        writer.add_scalar("loss/test", spike_loss / len(test_loader), epoch)
        writer.add_scalar("acc/test", correct / num_samples, epoch)

    return correct / num_samples


def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(updated_checkpoint_dir) / f"weights_{epoch:03d}.pt"
    model_path = Path(updated_checkpoint_dir) / f"model_{epoch:03d}.pt"
    torch.save(net.state_dict(), checkpoint_path)
    torch.save(net, model_path)

def _save_best_model(epoch, best_test_acc):
    checkpoint_path = Path(updated_checkpoint_dir) / f"weights_best.pt"
    model_path = Path(updated_checkpoint_dir) / f"model_best.pt"
    torch.save(net.state_dict(), checkpoint_path)
    torch.save(net, model_path)
    state = {
        'net': net.state_dict(),
        'best_acc': best_test_acc,
        'epoch': epoch,
    }
    overall_state_path = Path(updated_checkpoint_dir) / f"state_best.pt"
    torch.save(state, overall_state_path)

best_test_acc = 0
last_acc = 0
for epoch in range(1, args.epochs + 1):
    _train()
    if epoch % 10 == 0:
        curr_test_acc = _test()
        if curr_test_acc > best_test_acc:
            best_test_acc = curr_test_acc
            _save_best_model(epoch, best_test_acc)
        if epoch == args.epochs:
            last_acc = curr_test_acc
    if epoch % 100 == 0:
        _save_model(epoch)
print("last test acc: ", last_acc)
print("best test acc: ", best_test_acc)

args_pkl_path = Path(updated_checkpoint_dir) / f"args.pkl"
with args_pkl_path.open("wb") as f:
    pickle.dump(args, f)
