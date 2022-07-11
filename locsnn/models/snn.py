'''
Models

Codes based on https://github.com/clear-nus/VT_SNN
'''

import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer


class LocationSlayer(torch.nn.Module):
    """2-layer LocationSlayer built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayer, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)

        return spikeAll

class LocationSlayerFuse(torch.nn.Module):
    """2-layer LocationSlayerFuse built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayerFuse, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)

        self.outputFc = self.slayer.dense(output_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)
        out = self.slayer.spike(self.slayer.psp(self.outputFc(spikeAll)))
        return out

class OnlyLocationSlayer(torch.nn.Module):
    """2-layer OnlyLocationSlayer built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(OnlyLocationSlayer, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        return location_spike_output


class LocationSlayerWhorl(torch.nn.Module):
    """2-layer LocationSlayerWhorl built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayerWhorl, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)
        self.perm = []
        inner_outer = [21, 15, 16, 23, 27, 24, 17, 6, 9, 12, 20, 29, 33, 34, 31, 28, 22, 14, 10, 1, 2, 7, 18, 30, 37,
                       39, 38, 32, 19, 8, 3, 4, 11, 25, 35, 36, 26, 13, 5]
        for i in range(39):                                                 # left
            self.perm.append(2 * inner_outer[i] - 2)
            self.perm.append(2 * inner_outer[i] - 1)
        for i in range(39):                                                 # right
            self.perm.append(2 * inner_outer[i] - 2 + 78)
            self.perm.append(2 * inner_outer[i] - 1 + 78)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        spike_input_trans_perm = spike_input_trans[:, :, :, :, torch.LongTensor(self.perm)]
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans_perm)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)

        return spikeAll


class LocationSlayerArch(torch.nn.Module):
    """2-layer LocationSlayerArch built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayerArch, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)
        self.perm = []
        top_down = [11, 25, 35, 4, 18, 30, 7, 2, 20, 37, 29, 12, 9, 33, 23, 16, 1, 6, 15, 21, 27, 34, 39, 24, 17, 10, 31, 38, 28, 14, 3, 22, 32, 8, 19, 36, 5, 13, 26]
        for i in range(39):  # left
            self.perm.append(2 * top_down[i] - 2)
            self.perm.append(2 * top_down[i] - 1)
        for i in range(39):  # right
            self.perm.append(2 * top_down[i] - 2 + 78)
            self.perm.append(2 * top_down[i] - 1 + 78)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        spike_input_trans_perm = spike_input_trans[:, :, :, :, torch.LongTensor(self.perm)]
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans_perm)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)
        
        return spikeAll


class LocationSlayerRandom(torch.nn.Module):
    """2-layer LocationSlayerArch built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayerRandom, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)
        self.perm = []
        import random
        random_list = list(range(1, 40))
        random.shuffle(random_list)
        print(random_list)
        for i in range(39):  # left
            self.perm.append(2 * random_list[i] - 2)
            self.perm.append(2 * random_list[i] - 1)
        for i in range(39):  # right
            self.perm.append(2 * random_list[i] - 2 + 78)
            self.perm.append(2 * random_list[i] - 1 + 78)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        spike_input_trans_perm = spike_input_trans[:, :, :, :, torch.LongTensor(self.perm)]
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans_perm)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))

        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)
        
        return spikeAll
