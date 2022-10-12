import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        retval =  th.sum(agent_qs, dim=2, keepdim=True)
        if len(retval.shape) == 4:
            retval=retval.squeeze(3)
        return retval