import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch

class TurtleRunner:
    """Executes the TURTLE training routine and manages checkpoints and task encoder initialization."""

    def __init__(self, cfg, device, round, turtle_path="TURTLE"):
        self.cfg = cfg
        self.device = device
        self.turtle_path = Path(turtle_path)
        
        self.inner_lr = 0.001
        self.outer_lr = 0.005
        self.T = 6000
        self.M = 10
        self.gamma = 10.0
        self.batch_size = 10000
        
        self.dataset_name = cfg.DATASET.NAME_ADJ
        self.backbone_name = cfg.MODEL.BACKBONE.NAME_ADJ
        self.round = round

        turtle_path_str = str(self.turtle_path.resolve())
        if turtle_path_str not in sys.path:
            sys.path.insert(0, turtle_path_str)

        from run_turtle import run as train_turtle
        self.train_turtle = train_turtle

    def train(self, num_clusters):
        """Runs the TURTLE training routine with the specified number of clusters."""

        original_sys_path = sys.path.copy()

        turtle_data_dir = self.turtle_path / "data"

        cli_args = [
            "--dataset", self.dataset_name,
            "--phis", self.backbone_name,
            "--root_dir", str(turtle_data_dir),
            "--seed", str(self.cfg.SEED),
            "--warm_start",
            "--T", str(self.T),
            "--M", str(self.M),
            "--gamma", str(self.gamma),
            "--batch_size", str(self.batch_size),
            "--inner_lr", str(self.inner_lr),
            "--outer_lr", str(self.outer_lr),
            "--clusters", str(num_clusters),
            "--strategy", self.cfg.STRATEGY,
            "--round", str(self.round)
        ]

        print("\n>>> [TURTLE] Initializing training...")
        cluster_acc_train, cluster_acc_val, preds_train, preds_test = self.train_turtle(cli_args)
        sys.path = original_sys_path

        return cluster_acc_train, cluster_acc_val, preds_train, preds_test

    def load_checkpoint(self):
        """Loads the saved checkpoint of the TURTLE model for the specified dataset."""
        
        num_spaces = 1 
        
        checkpoint_dir = (
            self.turtle_path / "data" / "task_checkpoints" / 
            f"{num_spaces}space" / self.backbone_name / self.dataset_name
        )

        checkpoint_filename = (
            f"turtle_{self.backbone_name}"
            f"_innerlr{self.inner_lr}"
            f"_outerlr{self.outer_lr}"
            f"_T{self.T}_M{self.M}_warmstart_gamma{self.gamma}_bs{self.batch_size}"
            f"_{self.cfg.STRATEGY}"
            f"_round{self.round}"
            f"_seed{self.cfg.SEED}.pt"
        )
        full_checkpoint_path = checkpoint_dir / checkpoint_filename

        if not full_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado em: {full_checkpoint_path}")

        print(f">>> [TURTLE] Checkpoint loaded successfully: {full_checkpoint_path}")
        return torch.load(full_checkpoint_path, map_location=self.device)

    def init_task_encoder(self, input_dim: int, num_classes: int, checkpoint: Dict[str, Any]):
        """Initializes the linear layer (Task Encoder) without weight_norm."""
        """"""
        task_phi = torch.nn.Linear(input_dim, num_classes).to(self.device).to(torch.float32)
        
        task_phi.load_state_dict(checkpoint["phi1"])
        task_phi.eval()
        
        return task_phi
    
def run_turtle(self, num_clusters, dim, round):
    """Runs TURTLE training, loads the checkpoint, and initializes the task encoder."""

    turtle = TurtleRunner(self.cfg, self.device, round)

    cluster_acc_train, cluster_acc_val, preds_train, preds_val = turtle.train(num_clusters)

    checkpoint = turtle.load_checkpoint()

    task_phi = turtle.init_task_encoder(
        input_dim=dim,
        num_classes=num_clusters,
        checkpoint=checkpoint
    )

    return task_phi, cluster_acc_train, cluster_acc_val, preds_train, preds_val