import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from modules.mixers.qatten import QattenMixer
from torch.optim import Adam

def get_ws(resq_version, condition, td_error):
    if resq_version == "v3":
        ws = th.where(condition, th.zeros_like(td_error), th.ones_like(td_error)) 
    return ws

class RestQLearnerCentral:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        args.is_res_mixer = True;  # added 20220502
        if "rest_mixer" in vars(args):
            if args.rest_mixer == "vdn":
                self.rest_mixer = VDNMixer()
            elif args.rest_mixer == "qmix":
                self.rest_mixer = QMixer(args)
            elif args.rest_mixer == "qatten":
                self.rest_mixer = QattenMixer(args)
            else:
                self.rest_mixer = QMixerCentralFF(args)
        else:
            self.rest_mixer = QMixerCentralFF(args)

        args.is_res_mixer = False
        if self.args.central_mixer in ["ff", "atten", "vdn"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
                elif self.args.central_mixer == "vdn":
                    self.central_mixer = VDNMixer()
                else:
                    raise Exception("Error with central_mixer")

                assert args.central_mac == "basic_central_mac"
                self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
                self.target_central_mac = copy.deepcopy(self.central_mac)
                self.params += list(self.central_mac.parameters())

                self.rest_mac = copy.deepcopy(self.mac) #added for RESTQ
                self.rest_target_mac = copy.deepcopy(self.rest_mac)#added for RESTQ
                self.params += list(self.rest_mac.parameters())#added for RESTQ
        else:
            raise Exception("Error with qCentral")
        self.params += list(self.central_mixer.parameters())
        self.params += list(self.rest_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.rest_target_mixer = copy.deepcopy(self.rest_mixer) #added for RESTQ
        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters()) + list(self.rest_mixer.parameters())))

        if hasattr(self, "optimizer"):
            if getattr(self, "optimizer") == "Adam":
                self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_agent_outs = self.rest_mac.forward(batch, t=t)
            rest_mac_out.append(rest_agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Calculate the Q-Values necessary for the target
        rest_target_mac_out = []
        self.rest_target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_target_agent_outs = self.rest_target_mac.forward(batch, t=t)
            rest_target_mac_out.append(rest_target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        rest_target_mac_out = th.stack(rest_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        rest_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl


        # Max over target Q-Values
        if self.args.double_q:  # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach() #mac_out batch_size, seq_length, n_agents, n_commands
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)             #(max, max_indices) = torch.max(input, dim, keepdim=False)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)

            rest_mac_out_detach = rest_mac_out.clone().detach()
            rest_mac_out_detach[avail_actions == 0] = -9999999
            rest_cur_max_action_targets, rest_cur_max_actions = cur_max_action_targets, cur_max_actions #这一点要注意，RestQ的argmax必须和Q_tot一样的
            rest_target_max_agent_qvals = th.gather(rest_target_mac_out[:,:], 3, rest_cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
        #central_target_max_agent_qvals.shape torch.Size([128, 2, 2, 1])
        # print(rest_chosen_action_qvals.shape) #torch.Size([128, 1, 2])
        rest_chosen_action_qvals_ = rest_chosen_action_qvals.unsqueeze(3).repeat(1,1,1,self.args.central_action_embed).squeeze(3)
        # print(rest_chosen_action_qvals_.shape)
        Q_r = rest_chosen_action_qvals = self.rest_mixer(rest_chosen_action_qvals_, batch["state"][:,:-1])#added for RESTQ
        negative_abs = getattr(self.args, 'residual_negative_abs', False)
        if negative_abs:
            Q_r = - Q_r.abs()

        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Td-error
        """should clean this 4 lines"""
        td_error = (chosen_action_qvals - (targets.detach()))
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        """should clean this 4 lines"""

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1]) #这个就是Q^*(s,\tau, u)
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        # ws = th.ones_like(td_error) * self.args.w
        # ws = th.zeros_like(td_error)
        if self.args.hysteretic_qmix: # OW-QMIX
            w_r = th.where(td_error < 0, th.ones_like(td_error)*1, th.zeros_like(td_error)) # Target is greater than current max
            w_to_use = w_r.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            if self.args.condition == "max_action":
                condition = is_max_action
            elif self.args.condition == "max_larger":
                condition = is_max_action | qtot_larger
            # ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            nomask = getattr(self.args, 'nomask', False)
            if nomask:
                w_r = th.ones_like(td_error);
            else:
                w_r = get_ws(self.args.resq_version, condition, td_error)
            w_to_use = w_r.mean().item() # Average of ws for logging

        # print(chosen_action_qvals.shape, chosen_action_qvals.shape, ws.shape)
        td_error = (chosen_action_qvals + w_r.detach() * rest_chosen_action_qvals - (targets.detach()))
        # print("td_error.shape", td_error.shape)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask        # 0-out the targets that came from padded data
        qmix_loss = (masked_td_error ** 2).sum() / mask.sum()



        noopt_loss2 = None
        if self.args.resq_version in ["v3"]:
            Q_r_ = th.max(Q_r, th.zeros_like(Q_r))
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()
            noopt_loss = noopt_loss1 #if residual_negative_abs == True, then this loss should be zero
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item() ** 2
        agent_norm = agent_norm ** (1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm
        # self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            if noopt_loss1 is not None:
                self.logger.log_stat("noopt_loss1", noopt_loss1.item(), t_env)
            if noopt_loss2 is not None:
                self.logger.log_stat("noopt_loss2", noopt_loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.rest_target_mac.load_state(self.rest_mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.rest_target_mixer.load_state_dict(self.rest_mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.rest_mac.cuda()
        self.rest_target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.rest_mixer.cuda()
            self.rest_target_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
