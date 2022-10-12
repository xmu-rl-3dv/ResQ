REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .central_rnn_agent import CentralRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent