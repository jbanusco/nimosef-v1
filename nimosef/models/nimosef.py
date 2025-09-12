import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from nimosef.models.layers import INR


class MultiHeadNetwork(nn.Module):
    def __init__(self, num_subjects, num_labels, latent_size, motion_size, hidden_size=512, 
                 num_res_layers=2, linear_head=True):
        super(MultiHeadNetwork, self).__init__()
        self.num_labels = num_labels
        self.latent_size = latent_size
        self.motion_size = motion_size
        self.hidden_size = hidden_size
        self.linear_head = linear_head
        self.num_res_layers = num_res_layers

        # Latent code for each subject (learnable)
        self.h_init_std = 0.01 /  math.sqrt(self.latent_size)
        self.shape_code = torch.nn.Embedding(num_subjects, self.latent_size, max_norm=None).requires_grad_(True)
        torch.nn.init.normal_(self.shape_code.weight.data, 0.0, self.h_init_std)
        
        # ======= Embeddings =======

        # Coordinate embedding [coordinates + time] -- or position embedding
        self.coord_embedding = lambda x: x  # Identity for now

        # Latent space projection
        coordinates_size = 3  # X, Y, Z
        time_size = 1 # Time
        input_size = coordinates_size + time_size
        self.latent_projection = INR(input_size, self.latent_size, self.latent_size, self.latent_size, 
                                     num_hidden_layers=self.num_res_layers, use_residual=True)

        # ======= Prediction Heads =======

        # Segmentation Head
        if self.linear_head:
            self.segmentation_head = nn.Linear(self.latent_size, num_labels)
        else:
            self.segmentation_head = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, num_labels)
            )

        # Intensity Head
        if self.linear_head:
            self.intensity_head = nn.Linear(self.latent_size, 1)
        else:
            self.intensity_head = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 1)
            )

        # ------- Motion and Displacement Field Heads -------

        # Motion Code        
        self.motion_head = nn.Sequential(
            nn.Linear(self.latent_size + 1, motion_size),
            nn.SiLU(),
        )

        # Displacement Field Head
        if self.linear_head:
            self.displacement_head = nn.Linear(hidden_size + motion_size, 3)
        else:
            self.displacement_head = nn.Sequential(
                nn.Linear(self.latent_size + motion_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 3)  # Output displacement vector
            )

        # Initialize weights        
        # for m in self.modules():
        #     self.weights_init(m)
    

    def weights_init(self, m):
        """ Initialization of linear layers """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    

    def decode_latent(self, corrected_coords, time, h):
        # Embed coordinates and time 
        input_data = self.coord_embedding(torch.cat([corrected_coords.float(), time.float()], dim=1))

        # Latent projection
        latent_t = self.latent_projection((input_data, h))

        return latent_t


    def forward(self, coords, time, sample_idx):
        # Shape code
        h = self.shape_code(sample_idx)

        # Embed coordinates and time        
        input_data = self.coord_embedding(torch.cat([coords.float(), time.float()], dim=1))

        # Get the latent projection  
        latent_t = self.latent_projection((input_data, h))

        # Predict segmentation
        seg_pred = self.segmentation_head(latent_t)
        # seg_pred = nn.functional.softmax(seg_pred, dim=1)

        # Predict intensity
        intensity_pred = self.intensity_head(latent_t)

        # Apply softplus to keep values in (0,1)
        intensity_pred = F.sigmoid(intensity_pred)

        # Compute motion code
        motion_code = self.motion_head(torch.cat([latent_t, time.float()], dim=1))        

        # Compute displacement field
        motion_latent = torch.cat((latent_t, motion_code), dim=-1)
        displacement = self.displacement_head(motion_latent)
        displacement = 0.5 * torch.tanh(0.5 * displacement)  # Log-scaled tanh ~[-0.25, 0.25]

        return seg_pred, intensity_pred, displacement, h
