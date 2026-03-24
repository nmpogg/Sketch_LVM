import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

def visualize_tsne(visualize_classes, saved_features, mode="photo"):
    if mode == "sketch":
        X = np.concatenate([torch.stack(v["sketch"]).cpu().numpy()
                            for v in saved_features.values() if len(v["sketch"]) > 0], axis=0)
        y = sum([[k] * len(v["sketch"])
                for k, v in saved_features.items() if len(v["sketch"]) > 0], [])

        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1)).fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls in sorted(set(y)):
            idx = [i for i, t in enumerate(y) if t == cls]
            plt.scatter(
                Z[idx, 0], Z[idx, 1],
                s=20,
                marker="^",                  # tam giác
                label=visualize_classes[int(cls)]   # đổi số -> chữ
            )

        ax = plt.gca()
        ax.set_xticks([])   # bỏ trục tọa độ
        ax.set_yticks([])
        for spine in ax.spines.values():   # bỏ đường viền
            spine.set_visible(False)

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig("cvpr_sketch.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
    
    else:
        X = np.concatenate([torch.stack(v["photo"]).cpu().numpy()
                            for v in saved_features.values() if len(v["photo"]) > 0], axis=0)
        y = sum([[k] * len(v["photo"])
                for k, v in saved_features.items() if len(v["photo"]) > 0], [])

        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1)).fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls in sorted(set(y)):
            idx = [i for i, t in enumerate(y) if t == cls]
            plt.scatter(
                Z[idx, 0], Z[idx, 1],
                s=20,
                marker="o",                  # hình tròn
                label=visualize_classes[int(cls)]   # đổi số -> chữ
            )

        ax = plt.gca()
        ax.set_xticks([])   # bỏ trục tọa độ
        ax.set_yticks([])
        for spine in ax.spines.values():   # bỏ đường viền
            spine.set_visible(False)

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig("cvpr_photo.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()