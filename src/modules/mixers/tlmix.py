import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Tlmix(nn.Module):
    def __init__(self, args):
        super(Tlmix, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        #leader_net
        self.embed_dim = args.mixing_embed_dim
        self.fc1 = nn.Linear(self.state_dim, args.leader_hidden_dim)
        self.fc2 = nn.Linear(args.leader_hidden_dim, args.n_agents)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w1_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w1_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")
        
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_2 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w2_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w2_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))
        # State dependent bias for hidden layer
        self.hyper_b_2 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V2 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        
        # with leader
        x = F.relu(self.fc1(states))
        l = self.fc2(x)
        agent_qs = agent_qs + l
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w1_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V1(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        #Twin: First layer
        w2 = th.abs(self.hyper_w_2(states))
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.n_agents, self.embed_dim)
        b2 = b2.view(-1, 1, self.embed_dim)
        hidden2 = F.elu(th.bmm(agent_qs, w2) + b2)
        # Second layer
        w2_final = th.abs(self.hyper_w2_final(states))
        w2_final = w2_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v2 = self.V2(states).view(-1, 1, 1)
        # Compute final output
        y2 = th.bmm(hidden2, w2_final) + v2
        # Reshape and return
        q2_tot = y2.view(bs, -1, 1)

        return q_tot, q2_tot, l
