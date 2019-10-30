import torch
import torch.nn as nn
import torch.functional as F
from .highway import Highway

class InterUnit(nn.Module):
    def __init__(self, wordrep_dim, high_tagrep_dim,low_tagrep_dim, num_layers, f):
        super().__init__()
        print("build Interaction unit...")
        self.highway_low = Highway(low_tagrep_dim, low_tagrep_dim, num_layers, f)
        self.highway_high = Highway(high_tagrep_dim,  high_tagrep_dim, num_layers, f)
        self.highway_word = Highway(wordrep_dim,wordrep_dim,num_layers, f)

    def forward(self, wordrep,high_tag_rep,low_tag_rep):
        wordrep = self.highway_word(wordrep)
        new_low_tag_rep = self.highway_low(low_tag_rep)
        new_high_tag_rep = self.highway_high(high_tag_rep)

        high_rep = torch.cat((wordrep,high_tag_rep,new_low_tag_rep),dim = -1)
        low_rep = torch.cat((wordrep, new_high_tag_rep, low_tag_rep), dim=-1)

        return high_rep,low_rep