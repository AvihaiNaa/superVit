import math

import cv2
import torch
import torch.nn as nn
import numpy as np

from config import CONFIG


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

        # 1) Linear mapper
        if CONFIG.VIT.TYPE == "PATCHES":
            self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
            self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        elif CONFIG.VIT.TYPE == "SUPERPIXELS":
            self.input_d = 1

        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape

        if CONFIG.VIT.TYPE == "PATCHES":
            patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        elif CONFIG.VIT.TYPE == "SUPERPIXELS":
            patches = apply_superpixels(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution



class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])



class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    
    
def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_superpixels(image, sp_size):
    # Convert the tensor to a numpy array
    image = image.numpy()
    # Transpose the dimensions to change the shape from (c, h, w) to (h, w, c)
    image = np.transpose(image, (1, 2, 0))

    # Create the superpixel object
    superpixel = cv2.ximgproc.createSuperpixelSLIC(image, region_size=sp_size)

    # Iterate to get the desired number of superpixels
    iterations = 5
    superpixel.iterate(iterations)

    # Get the labels of each pixel indicating which superpixel it belongs to
    labels = superpixel.getLabels()

    # Create a mask to visualize the superpixels
    mask = labels.reshape(image.shape[0], image.shape[1])
    return mask


def get_average_pixel_values(image, superpixel_mask, n_superpixels):
    c, _, _ = image.shape
    flat_mask = superpixel_mask.flatten()
    flat_image = image.reshape(-1, c).numpy()

    # Create an array to store the counts of pixels for each superpixel
    pixel_counts = np.bincount(flat_mask, minlength=n_superpixels)

    # Create an array to store the sums of pixel values for each superpixel
    pixel_sums = np.zeros((n_superpixels, ), dtype=flat_image.dtype)

    # Accumulate the pixel values for each superpixel
    np.add.at(pixel_sums, flat_mask[:, None], flat_image)

    # Calculate the mean pixel values for each superpixel
    average_values = np.divide(pixel_sums, pixel_counts, out=np.zeros_like(pixel_sums), where=(pixel_counts != 0))

    # Handle cases where a superpixel has no pixels by copying the average values from the previous superpixel
    mask_empty_superpixels = (pixel_counts == 0)
    average_values[mask_empty_superpixels] = np.roll(average_values, 1, axis=0)[mask_empty_superpixels]

    return average_values[:, None]


def apply_superpixels(images, n_superpixels):
    n, c, h, w = images.shape

    assert h == w, "Superpixels method is implemented for square images only"

    superpixels = torch.zeros(n, n_superpixels ** 2, c)
    sp_size = int(math.sqrt((h * w) / n_superpixels**2))

    for idx, image in enumerate(images):
        superpixel_mask = get_superpixels(image, sp_size)
        avg_superpixels = get_average_pixel_values(image, superpixel_mask, n_superpixels**2)
        superpixels[idx] = torch.tensor(avg_superpixels)
    return superpixels


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
