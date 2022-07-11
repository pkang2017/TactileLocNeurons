'''
Timestep-wise inference

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
from locsnn.models.snn_timestep import LocationSlayerWeighted
from locsnn.dataset import ViTacDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--runs", action="store", type=str, help="Path containing the model files.", required=True
)
parser.add_argument(
    "--save", action="store", type=str, help="Path to save the timestep-wise inference figure.", required=True
)
run_args = parser.parse_args()

def analyse_model(model_dir):
    device = torch.device("cuda")

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
    params = snn.params(args.network_config)

    model_args = {
        "params": params,
        "input_size": 156 if args.fingers == "both" else 78,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }

    net = LocationSlayerWeighted(**model_args).to(device)
    net.load_state_dict(state['net'])

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

    time_length = params["simulation"]["tSample"]
    correct_spike_output = torch.zeros(time_length)
    correct_location_spike_output = torch.zeros(time_length)
    correct_spikeAll = torch.zeros(time_length)
    correct_spikeAll_weighted = torch.zeros(time_length)

    num_samples = torch.zeros(time_length)
    acc_spike_output = torch.zeros(time_length)
    acc_location_spike_output = torch.zeros(time_length)
    acc_spikeAll = torch.zeros(time_length)
    acc_spikeAll_weighted = torch.zeros(time_length)

    net.eval()
    with torch.no_grad():
        for *data, _, label in test_loader:
            for t in range(time_length):
                data = [d.to(device) for d in data]
                input_data = []
                for d in data:
                    input_data.append(torch.cat((d[:, :, :, :, :t + 1],
                           torch.zeros((d.shape[0], d.shape[1], d.shape[2], d.shape[3], time_length - t - 1)).to(device)), dim=-1))

                spike_output, location_spike_output, spikeAll, spikeAll_weighted = net.forward(*input_data, t)

                prediction_spike_output = snn.predict.getClass(spike_output)
                prediction_location_spike_output = snn.predict.getClass(location_spike_output)
                prediction_spikeAll = snn.predict.getClass(spikeAll)
                prediction_spikeAll_weighted = snn.predict.getClass(spikeAll_weighted)

                correct_spike_output[t] += torch.sum(
                    prediction_spike_output == label
                ).data.item()
                correct_location_spike_output[t] += torch.sum(
                    prediction_location_spike_output == label
                ).data.item()
                correct_spikeAll[t] += torch.sum(
                    prediction_spikeAll == label
                ).data.item()
                correct_spikeAll_weighted[t] += torch.sum(
                    prediction_spikeAll_weighted == label
                ).data.item()

                num_samples[t] += len(label)

    for i in range(time_length):
        acc_spike_output[i] = correct_spike_output[i] / num_samples[i]
        acc_location_spike_output[i] = correct_location_spike_output[i] / num_samples[i]
        acc_spikeAll[i] = correct_spikeAll[i] / num_samples[i]
        acc_spikeAll_weighted[i] = correct_spikeAll_weighted[i] / num_samples[i]


    return acc_spike_output, acc_location_spike_output, acc_spikeAll, acc_spikeAll_weighted, args.task, args.sample_file

if __name__ == "__main__":
    acc_slayer, acc_location_only, acc_location_cat, acc_location_cat_time_weighted, task, sample_file = analyse_model(Path(run_args.runs))

    # print("SNN_SRM", acc_slayer)
    # print("SNN_Location_SRM", acc_location_only)
    # print("Hybrid", acc_location_cat)
    # print("Hybrid_time_weighted", acc_location_cat_time_weighted)
    time_length = len(acc_location_cat)
    plt.rc('axes', labelsize=14)
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels

    SSR = 0.02   # 0.02 for ycb and cw, 0.001 for slip
    plt.plot(np.arange(time_length)*SSR, acc_slayer, label="SNN_TSRM")
    plt.plot(np.arange(time_length)*SSR, acc_location_only, label="SNN_LSRM")
    plt.plot(np.arange(time_length)*SSR, acc_location_cat, label="Hybrid")
    plt.plot(np.arange(time_length)*SSR, acc_location_cat_time_weighted, label="Hybrid_time_weighted")
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(run_args.save + str(task) + '_' + str(sample_file) + '.png')




