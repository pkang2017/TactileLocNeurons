'''
Plot confusion matrices

Based on https://github.com/clear-nus/VT_SNN
'''

import argparse
import os
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.legend import Legend

from torch.utils.data import DataLoader
import slayerSNN as snn
from locsnn.dataset import ViTacDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--runs", action="store", type=str, help="Path containing the model file", required=True
)

run_args = parser.parse_args()

def plot_confusion(predicted, actual, model, args):
    "Plots the confusion matrix."
    fig, ax = plt.subplots(figsize=(6,4))
    data = { "predicted": predicted, "actual": actual}
    df = pd.DataFrame(data, columns=["predicted", "actual"])
    cfm = pd.crosstab(df["actual"], df["predicted"], rownames=["Actual"], colnames=["Predicted"])
    sns.heatmap(cfm, annot=True)
    output_dir = f"confusion/{args.task}/{args.mode}_{args.loss}_best_{args.sample_file}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"{model.name}.png")

def analyse_model(model_dir):
    "Saves the confusion matrix."
    device = torch.device("cuda")
    # net = torch.load(model_dir / "model_500.pt").to(device)
    net = torch.load(model_dir / "model_best.pt").to(device)

    state = torch.load(model_dir / "state_best.pt")
    print(state["best_acc"])
    pickled_args = Path(model_dir / "args.pkl")
    with open(pickled_args, "rb") as f:
        args = pickle.load(f)

    if args.task == "cw":
        output_size = 20
    elif args.task == "slip":
        output_size = 2
    else:
        output_size = 36

    test_dataset = ViTacDataset(
        path=args.data_dir,
        sample_file=f"test_80_20_{args.sample_file}.txt",
        output_size=output_size,
        spiking=True,
        mode=args.mode,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    predictions = []
    actual = []

    correct = 0
    num_samples = 0

    net.eval()
    with torch.no_grad():
        for *data, target, label in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = net.forward(*data)
            prediction = snn.predict.getClass(output)
            correct += torch.sum(
                prediction == label
            ).data.item()
            predictions.extend(prediction)
            actual.extend(label)
            num_samples += len(label)

    predictions = list(map(lambda v: v.item(), predictions))
    actual = list(map(lambda v: v.item(), actual))

    plot_confusion(predictions, actual, model_dir, args)
    print(correct / num_samples)

if __name__ == "__main__":
    print(run_args.runs)
    analyse_model(Path(run_args.runs))




