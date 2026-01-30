import torch
import matplotlib.pyplot as plt


def visualize_attention_map(att_map, title="Attention Map"):
    if att_map.dim() == 4:
        att_map = att_map.mean(dim=1)

    att_map = att_map.squeeze().cpu().numpy()

    plt.imshow(att_map, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_triplet_branches(att_hw, att_cw, att_ch):
    visualize_attention_map(att_hw, "HW Attention")
    visualize_attention_map(att_cw, "CW Attention")
    visualize_attention_map(att_ch, "CH Attention")
