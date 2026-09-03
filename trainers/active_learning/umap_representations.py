import os
import numpy as np
import torch

def save_data_for_umap(cfg, round, labeled_in_set, train_features, model_train_pse, turtle_train_pse, labels_org, val_features=None, val_model_pse=None, val_turtle_pse=None,
                    val_org=None, test_features=None, test_model_pse=None, test_turtle_pse=None, test_org=None):
        """Save representations fo dataset, ground truth labels and pseudolabels (Model and TURTLE)"""
        if round == 0:
            round = "initial"
        else:
            round-=1

        umap_dir = f"umap/turtle/{cfg.DATASET.NAME_ADJ}/{cfg.STRATEGY}"
        os.makedirs(umap_dir, exist_ok=True)
        save_path = f"{umap_dir}/umap_round{round}_seed{cfg.SEED}.npz"

        print(f">>> Saving complete dataset representations for UMAP to: {save_path}")

        def _to_np(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        train_feat = _to_np(train_features)
        val_feat = _to_np(val_features) if val_features is not None else None
        test_feat = _to_np(test_features) if test_features is not None else None

        all_features = [train_feat]
        all_real = [labels_org]
        all_model_pse = [model_train_pse]
        all_turtle_pse = [turtle_train_pse]

        split_tags = []
        n_labeled = len(labeled_in_set) if round != 0 else 0
        n_unlabeled = len(labels_org) - n_labeled

        split_tags.extend(["train_labeled"] * n_labeled)
        split_tags.extend(["train_unlabeled"] * n_unlabeled)

        if val_feat is not None:
            all_features.append(val_feat)
            all_real.append(val_org)
            all_model_pse.append(val_model_pse)
            all_turtle_pse.append(val_turtle_pse if val_turtle_pse is not None else val_model_pse)
            split_tags.extend(["val"] * len(val_org))

        if test_feat is not None:
            all_features.append(test_feat)
            all_real.append(test_org if test_org is not None else test_model_pse)
            all_model_pse.append(test_model_pse)
            all_turtle_pse.append(test_turtle_pse if test_turtle_pse is not None else test_model_pse)
            split_tags.extend(["test"] * len(test_model_pse))

        full_features = np.concatenate(all_features, axis=0)
        full_real = np.concatenate(all_real, axis=0)
        full_model_pse = np.concatenate(all_model_pse, axis=0)
        full_turtle_pse = np.concatenate(all_turtle_pse, axis=0)
        split_tags = np.array(split_tags)

        np.savez(
            save_path,
            features=full_features,
            real_labels=full_real,
            model_pse=full_model_pse,
            turtle_pse=full_turtle_pse,
            split_tags=split_tags,
        )