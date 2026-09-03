from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from dassl.data.data_manager import build_data_loader
from dassl.data.transforms.transforms import build_transform


class TurtleData:
    """Pipeline for extracting and saving representations and labels for TURTLE."""
    def __init__(self, model, cfg, unlabeled_dst, dataset, round, turtle_path="TURTLE"):
        self.model = model
        self.cfg = cfg
        self.unlabeled_dst = unlabeled_dst
        self.dataset = dataset
        self.dataset_name = cfg.DATASET.NAME_ADJ
        self.backbone_name = cfg.MODEL.BACKBONE.NAME_ADJ
        self.turtle_path = Path(turtle_path)
        self.strategy = cfg.STRATEGY
        self.round = round
        self.seed = self.cfg.SEED


    def build_dataloaders(self):
        """Creates DataLoaders for the labeled and unlabeled datasets."""
        train_loader = build_data_loader(
            self.cfg,
            data_source=self.unlabeled_dst,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=build_transform(self.cfg, is_train=False),
            is_train=False,
        )

        test_loader = build_data_loader(
            self.cfg,
            data_source=self.dataset.test,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=build_transform(self.cfg, is_train=False),
            is_train=False,
        )

        return train_loader, test_loader

    @torch.no_grad()
    def extract_representations(self, loader, description: str):
        """Extracts representations and labels from the model for a given DataLoader."""
        img_feats, dot_feats, labels = [], [], []

        for batch in tqdm(loader, desc=description):
            image = batch["img"].cuda()
            label = batch["label"].cuda()

            preds, img_features, txt_features = self.model.model_inference(image, get_feature=True)

            preds_prob = torch.nn.functional.softmax(preds, dim=1)
            dot_features = torch.matmul(preds_prob, txt_features)

            img_feats.append(img_features.cpu())
            dot_feats.append(dot_features.cpu())
            labels.append(label.cpu())

        combined_features = torch.cat([torch.cat(img_feats), torch.cat(dot_feats)], dim=1)
        labels_array = torch.cat(labels).numpy()
        raw_img_feats_array = torch.cat(img_feats).numpy()

        return combined_features.numpy(), labels_array, raw_img_feats_array

    def run(self):
        """Executes the full pipeline: builds dataloaders, extracts features, and saves them."""

        print(f"--- [TURTLE] Initializing data preparation ({self.dataset_name} | {self.backbone_name}) ---")

        representations_dir = self.turtle_path / "data" / "representations" / self.backbone_name / self.dataset_name
        labels_dir = self.turtle_path / "data" / "labels" / self.dataset_name

        representations_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader = self.build_dataloaders()

        train_feats, train_labels, _ = self.extract_representations(train_loader, "Train Features")
        test_feats, test_labels, _ = self.extract_representations(test_loader, "Test Features")

   
        print("Saving files .npy...")

        np.save(representations_dir / f"train_{self.strategy}_round{self.round}_seed{self.seed}.npy", train_feats)
        np.save(labels_dir / f"train_{self.strategy}_round{self.round}_seed{self.seed}.npy", train_labels)
        np.save(representations_dir / f"val_{self.strategy}_round{self.round}_seed{self.seed}.npy", test_feats)
        np.save(labels_dir / f"val_{self.strategy}_round{self.round}_seed{self.seed}.npy", test_labels)

        print(f"--- [TURTLE] Data saved to: {representations_dir} ---")
        return train_feats, train_labels



def prepare_turtle_data(model, cfg, unlabeled_dst, dataset, round):
    """Prepares and saves TURTLE representations."""

    pipeline = TurtleData(model=model, cfg=cfg, unlabeled_dst=unlabeled_dst, dataset=dataset, round=round)
    return pipeline.run()