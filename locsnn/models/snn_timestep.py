import torch
import slayerSNN as snn
import math

class LocationSlayerWeighted(torch.nn.Module):
    """2-layer LocationSlayerWeighted built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(LocationSlayerWeighted, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

        self.locationFc1 = self.slayer.dense(params["simulation"]["tSample"], hidden_size)
        self.locationFc2 = self.slayer.dense(hidden_size, output_size)
 

    def forward(self, spike_input, t):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input[:,:,:,:,:t+1])))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))

        spike_input_trans = torch.transpose(spike_input, 1, 4)
        location_spike_1 = self.slayer.spike(self.slayer.psp(self.locationFc1(spike_input_trans)))
        location_spike_output = self.slayer.spike(self.slayer.psp(self.locationFc2(location_spike_1)))
        spikeAll = torch.cat([spike_output, location_spike_output], dim=4)

        time_length = spike_input_trans.shape[1]
        TIME_CONSTANT = 10                            # 15 for ycb; 2 for slip; 10 for cw
        lam = 1/(1+math.exp(-TIME_CONSTANT*(t/time_length - 1)))       
        spike_output_weighted = spike_output * (1-lam)
        location_spike_output_weighted = location_spike_output * lam
        spikeAll_weighted = torch.cat([spike_output_weighted, location_spike_output_weighted], dim=4)

        return spike_output, location_spike_output, spikeAll, spikeAll_weighted
