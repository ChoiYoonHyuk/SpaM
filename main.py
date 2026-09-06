import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_float(value: torch.Tensor) -> torch.Tensor:
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.float()
    return value


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def graph_convolutions():
    try:
        from torch_geometric.nn import GATConv, GCNConv
    except ImportError as error:
        raise ImportError("Install torch-geometric to construct the graph encoders.") from error
    return GATConv, GCNConv


def load_dataset(data_id: int):
    from torch_geometric.datasets import Actor, HeterophilousGraphDataset, WebKB, WikipediaNetwork

    if data_id == 0:
        dataset = HeterophilousGraphDataset(root="/tmp/RomanEmpire", name="Roman-empire")
    elif data_id == 1:
        dataset = HeterophilousGraphDataset(root="/tmp/Minesweeper", name="Minesweeper")
    elif data_id == 2:
        dataset = HeterophilousGraphDataset(root="/tmp/AmazonRatings", name="Amazon-ratings")
    elif data_id == 3:
        dataset = WikipediaNetwork(root="/tmp/Chameleon", name="chameleon")
    elif data_id == 4:
        dataset = WikipediaNetwork(root="/tmp/Squirrel", name="squirrel")
    elif data_id == 5:
        dataset = Actor(root="/tmp/Actor")
    elif data_id == 6:
        dataset = WebKB(root="/tmp/Cornell", name="Cornell")
    elif data_id == 7:
        dataset = WebKB(root="/tmp/Texas", name="Texas")
    elif data_id == 8:
        dataset = WebKB(root="/tmp/Wisconsin", name="Wisconsin")
    else:
        raise ValueError("data must be an integer from 0 through 8.")
    return dataset, int(dataset.num_classes)


class StructuralEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_hidden_dim: int = 128,
        num_classes: int = 0,
        use_gat: bool = False,
        dropout: float = 0.5,
        use_labels: bool = True,
    ):
        super().__init__()
        if use_labels and num_classes <= 0:
            raise ValueError("Label-aware encoding requires a positive num_classes.")
        self.dropout = dropout
        self.use_labels = use_labels
        self.num_classes = num_classes
        label_dim = num_classes if use_labels else 0
        GATConv, GCNConv = graph_convolutions()
        if use_gat:
            self.conv1 = GATConv(in_dim + label_dim, hidden_dim, heads=1, concat=True)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)
        else:
            self.conv1 = GCNConv(in_dim + label_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_labels:
            if y is None or train_mask is None:
                raise ValueError("Label-aware encoding requires y and a training mask.")
            if train_mask.ndim != 1 or train_mask.dtype != torch.bool:
                raise ValueError("Select one Boolean training mask before calling the model.")
            label_features = x.new_zeros((x.size(0), self.num_classes))
            label_features[train_mask] = F.one_hot(
                y[train_mask], num_classes=self.num_classes
            ).to(dtype=x.dtype)
            x_in = torch.cat((x, label_features), dim=-1)
        else:
            x_in = x
        h = self.conv1(x_in, edge_index)
        h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        source, target = edge_index
        edge_features = torch.cat((h[target], h[source]), dim=-1)
        edge_logits = stable_float(self.edge_mlp(edge_features))
        log_probs = F.log_softmax(edge_logits, dim=-1)
        edge_probs = log_probs.exp()
        if edge_logits.size(0) == 0:
            struct_loss = edge_logits.sum() * 0.0
        else:
            kl = (edge_probs * (log_probs + math.log(3.0))).sum(dim=-1).mean()
            log_activity = torch.logsumexp(log_probs[:, (0, 2)], dim=-1)
            struct_loss = kl - log_activity.mean()
        return edge_logits, edge_probs, struct_loss


class S2Layer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        val_dim: int,
        init_gamma: float = 1.0,
        l1_lambda: float = 0.05,
    ):
        super().__init__()
        if l1_lambda < 0:
            raise ValueError("The coefficient threshold must be nonnegative.")
        self.l1_lambda = float(l1_lambda)
        self.W_v = nn.Linear(in_dim, val_dim, bias=False)
        self.W_t = nn.Linear(in_dim, val_dim, bias=False)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * val_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.W_self = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(val_dim, hidden_dim)
        self.gamma_param = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        role_gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if H.ndim != 2 or H.size(0) == 0:
            raise ValueError("H must contain at least one node.")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_routes].")
        if role_gates.shape != (edge_index.size(1), 3):
            raise ValueError("role_gates must have negative, inactive, positive columns.")
        source, target = edge_index
        V = self.W_v(H)
        T = self.W_t(H)
        values = stable_float(V[source])
        scorer_input = torch.cat((T[target], V[source]), dim=-1)
        scores = stable_float(self.alpha_mlp(scorer_input).squeeze(-1))
        masses = F.softshrink(scores, lambd=self.l1_lambda).abs()
        gates = role_gates.to(dtype=masses.dtype)
        gamma = stable_float(F.softplus(self.gamma_param))
        signed_gate = gates[:, 2] - gamma * gates[:, 0]
        activity_gate = gates[:, 2] + gates[:, 0]
        messages = (masses * signed_gate).unsqueeze(-1) * values
        aggregate = values.new_zeros((H.size(0), V.size(1)))
        aggregate = aggregate.index_add(0, target, messages)
        H_new = self.W_out(aggregate) + self.W_self(H)
        sparse_loss = (masses * activity_gate).sum() / H.size(0)
        return H_new, sparse_loss


class SpaM(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        K: int = 5,
        K_test: int | None = None,
        val_dim: int = 64,
        init_gamma: float = 1.0,
        l1_lambda: float = 0.05,
        dropout: float = 0.5,
        use_gat: bool = False,
        use_backbone_mp: bool = True,
        use_labels: bool = True,
        temperature: float = 0.5,
    ):
        super().__init__()
        if min(in_feats, hidden_dim, num_classes, num_layers, K, val_dim) <= 0:
            raise ValueError("Dimensions, depth, and sample count must be positive.")
        if K_test is not None and K_test <= 0:
            raise ValueError("K_test must be positive.")
        if not 0 <= dropout < 1 or temperature <= 0:
            raise ValueError("Require 0 <= dropout < 1 and temperature > 0.")
        self.K = int(K)
        self.K_test = int(K if K_test is None else K_test)
        self.dropout = float(dropout)
        self.temperature = float(temperature)
        self.use_backbone_mp = use_backbone_mp
        self.struct_encoder = StructuralEncoder(
            in_dim=in_feats,
            hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_gat=use_gat,
            dropout=dropout,
            use_labels=use_labels,
        )
        if use_backbone_mp:
            GATConv, _ = graph_convolutions()
            self.backbone1 = GATConv(in_feats, hidden_dim, heads=1, concat=True)
            self.backbone2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)
            self.backbone_proj = nn.Linear(in_feats, hidden_dim, bias=False)
            self.mlp_backbone = None
        else:
            self.backbone1 = None
            self.backbone2 = None
            self.backbone_proj = None
            self.mlp_backbone = nn.Sequential(
                nn.Linear(in_feats, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        self.layers = nn.ModuleList(
            S2Layer(hidden_dim, hidden_dim, val_dim, init_gamma, l1_lambda)
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _backbone(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_backbone_mp:
            h = F.relu(self.backbone1(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.backbone2(h, edge_index) + self.backbone_proj(x))
            return F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.mlp_backbone(x))
        return F.dropout(h, p=self.dropout, training=self.training)

    def _sample_roles(self, edge_logits: torch.Tensor) -> torch.Tensor:
        if edge_logits.size(0) == 0:
            return edge_logits.clone()
        if self.training:
            return F.gumbel_softmax(
                edge_logits, tau=self.temperature, hard=True, dim=-1
            )
        indices = torch.distributions.Categorical(logits=edge_logits).sample()
        return F.one_hot(indices, num_classes=3).to(dtype=edge_logits.dtype)

    def forward(
        self,
        data: Any,
        K: int | None = None,
        return_details: bool = False,
    ) -> dict[str, Any]:
        x, edge_index = data.x, data.edge_index
        y = getattr(data, "y", None)
        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None and (train_mask.ndim != 1 or train_mask.dtype != torch.bool):
            raise ValueError("Select one Boolean training mask before calling the model.")
        samples = (self.K if self.training else self.K_test) if K is None else int(K)
        if samples <= 0:
            raise ValueError("The number of branches must be positive.")
        H0 = self._backbone(x, edge_index)
        edge_logits, edge_probs, struct_loss = self.struct_encoder(x, edge_index, y, train_mask)
        branch_log_probs = []
        sparse_losses = []
        sampled_gates = []
        for _ in range(samples):
            role_gates = self._sample_roles(edge_logits)
            H = H0
            layer_penalties = []
            for layer in self.layers:
                H, sparse_layer = layer(H, edge_index, role_gates)
                H = F.dropout(F.relu(H), p=self.dropout, training=self.training)
                layer_penalties.append(sparse_layer)
            logits = stable_float(self.classifier(H))
            branch_log_probs.append(F.log_softmax(logits, dim=-1))
            sparse_losses.append(torch.stack(layer_penalties).mean())
            if return_details:
                sampled_gates.append(role_gates)
        log_probs_stack = torch.stack(branch_log_probs, dim=0)
        log_probs_mc = torch.logsumexp(log_probs_stack, dim=0) - math.log(samples)
        if self.training:
            if y is None or train_mask is None or not bool(train_mask.any()):
                raise ValueError("Training requires labels and a nonempty training mask.")
            cls_loss = F.nll_loss(log_probs_mc[train_mask], y[train_mask])
        else:
            cls_loss = None
        result = {
            "logits": log_probs_mc,
            "probs": log_probs_mc.exp(),
            "cls_loss": cls_loss,
            "sparse_loss": torch.stack(sparse_losses).mean(),
            "struct_loss": struct_loss,
        }
        if return_details:
            result.update(
                edge_logits=edge_logits,
                edge_probs=edge_probs,
                branch_log_probs=log_probs_stack,
                role_gates=torch.stack(sampled_gates),
                initial_features=H0,
            )
        return result


def total_objective(
    output: dict[str, Any],
    lambda_sp: float = 0.01,
    lambda_st: float = 0.1,
) -> torch.Tensor:
    if output["cls_loss"] is None:
        raise ValueError("The total training objective requires training-mode outputs.")
    return output["cls_loss"] + lambda_sp * output["sparse_loss"] + lambda_st * output["struct_loss"]


def split_count(data: Any) -> int:
    masks = [data.train_mask, data.val_mask, data.test_mask]
    if any(mask.dtype != torch.bool or mask.ndim not in (1, 2) for mask in masks):
        raise ValueError("Dataset masks must be one- or two-dimensional Boolean tensors.")
    counts = [1 if mask.ndim == 1 else mask.size(1) for mask in masks]
    if len(set(counts)) != 1 or counts[0] == 0:
        raise ValueError("Training, validation, and test masks must have matching split counts.")
    return counts[0]


def select_split(data: Any, split_idx: int):
    count = split_count(data)
    if not 0 <= split_idx < count:
        raise ValueError(f"split_idx must be between 0 and {count - 1}.")
    selected = data.clone()
    masks = []
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(selected, name)
        mask = mask if mask.ndim == 1 else mask[:, split_idx]
        if mask.numel() != selected.x.size(0) or not bool(mask.any()):
            raise ValueError(f"{name} must be nonempty and aligned with node features.")
        setattr(selected, name, mask)
        masks.append(mask)
    if bool(((masks[0] & masks[1]) | (masks[0] & masks[2]) | (masks[1] & masks[2])).any()):
        raise ValueError("Training, validation, and test masks must be disjoint.")
    return selected


@torch.no_grad()
def predict(
    model: nn.Module,
    data: Any,
    samples: int,
    seed: int,
    use_amp: bool = False,
) -> torch.Tensor:
    device = data.x.device
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(data, K=samples)
            return output["logits"].detach()
    finally:
        model.train(was_training)


def accuracy(log_probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((log_probs[mask].argmax(dim=-1) == labels[mask]).float().mean().item())


def final_metrics(log_probs: torch.Tensor, data: Any, data_id: int) -> dict[str, Any]:
    test_acc = accuracy(log_probs, data.y, data.test_mask)
    nll = float(F.nll_loss(log_probs[data.test_mask], data.y[data.test_mask]).item())
    if data_id == 1:
        from sklearn.metrics import roc_auc_score

        y_test = data.y[data.test_mask].detach().cpu().numpy()
        if np.unique(y_test).size != 2:
            raise ValueError("Minesweeper test ROC-AUC requires both classes in the test mask.")
        probabilities = log_probs[data.test_mask, 1].exp().float().cpu().numpy()
        score = float(roc_auc_score(y_test, probabilities))
        metric = "ROC-AUC"
    else:
        score = test_acc
        metric = "Accuracy"
    return {
        "test_metric": metric,
        "test_score_percent": 100.0 * score,
        "test_accuracy_percent": 100.0 * test_acc,
        "test_nll": nll,
    }


def train_one_split(
    data: Any,
    num_classes: int,
    split_idx: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    seed = args.seed + split_idx
    eval_seed = args.eval_seed + split_idx
    set_seed(seed)
    model = SpaM(
        in_feats=data.x.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        K=args.K_train,
        K_test=args.K_test,
        val_dim=args.val_dim,
        init_gamma=1.0,
        l1_lambda=args.lambda_sc,
        dropout=args.dropout,
        use_gat=args.use_gat,
        use_backbone_mp=args.data not in (3, 4),
        use_labels=not args.feature_only,
        temperature=args.temperature,
    ).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=50, min_lr=1e-5
    )
    use_amp = args.use_amp and data.x.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = -math.inf
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=data.x.device.type, dtype=torch.float16, enabled=use_amp):
            output = model(data)
            loss = total_objective(output, args.lambda_sp, args.lambda_st)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite objective at split {split_idx}, epoch {epoch}.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().item())
        del output, loss
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.max_epochs
        if not should_evaluate:
            continue
        log_probs = predict(model, data, args.K_test, eval_seed, use_amp)
        val_acc = accuracy(log_probs, data.y, data.val_mask)
        del log_probs
        scheduler.step(val_acc)
        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if improved or epoch % args.log_every == 0:
            print(
                f"Split {split_idx:02d} | Epoch {epoch:04d} | Loss {loss_value:.6f} | "
                f"Val {100 * val_acc:.4f} | Best Val {100 * best_val:.4f}",
                flush=True,
            )
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    log_probs = predict(model, data, args.K_test, eval_seed, use_amp)
    result = {
        "split_idx": split_idx,
        "seed": seed,
        "eval_seed": eval_seed,
        "best_epoch": best_epoch,
        "selection_validation_accuracy_percent": 100.0 * best_val,
        "final_validation_accuracy_percent": 100.0 * accuracy(log_probs, data.y, data.val_mask),
        **final_metrics(log_probs, data, args.data),
    }
    config = {
        **vars(args),
        "resolved_device": str(data.x.device),
        "resolved_use_amp": use_amp,
        "structural_encoder": "GATConv" if args.use_gat else "GCNConv",
        "backbone": "MLP" if args.data in (3, 4) else "two-layer GAT with feature projection",
        "input_modality": "feature-only" if args.feature_only else "label-aware",
        "role_coordinate_order": ["negative", "inactive", "positive"],
        "decoder_input_order": ["target", "source"],
        "signed_layer_residual": False,
        "backbone_dropout_shared_across_branches": True,
        "signed_dropout_site": "after each signed-layer ReLU, independent across branches",
        "mlp_backbone_dropout_sites": "inside the MLP and at the shared backbone output",
        "regularizers_active_from_epoch": 1,
        "min_epochs_semantics": "early-stopping floor, not classification-only warm-up",
        "graph_preprocessing": "L2-normalized features and coalesced supplied directed routes",
        "validation_sampling": "fixed evaluation seed, training RNG restored after evaluation",
        "checkpoint_selection": "strict improvement in validation accuracy",
        "num_nodes": int(data.x.size(0)),
        "num_observed_directed_routes": int(data.edge_index.size(1)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": importlib.metadata.version("torch-geometric"),
    }
    checkpoint_path = output_dir / f"split_{split_idx:02d}_seed_{seed}.pt"
    torch.save({"state_dict": best_state, "config": config, "result": result}, checkpoint_path)
    result["checkpoint"] = str(checkpoint_path)
    (output_dir / f"split_{split_idx:02d}_seed_{seed}.json").write_text(
        json.dumps({"config": config, "result": result}, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SpaM with manuscript-aligned signed propagation")
    parser.add_argument("data", type=int, choices=range(9))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=100000)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_gat", action="store_true")
    parser.add_argument("--feature_only", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split_idx", type=int, default=0)
    group.add_argument("--all_splits", action="store_true")
    parser.add_argument("--K_train", type=int, default=5)
    parser.add_argument("--K_test", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--val_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=0.05)
    parser.add_argument("--lambda_sp", type=float, default=0.01)
    parser.add_argument("--lambda_st", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--output_dir", default="spam_runs")
    args = parser.parse_args()
    positive = (args.max_epochs, args.patience, args.eval_every, args.log_every, args.K_train, args.K_test)
    if min(positive) <= 0 or args.min_epochs < 0:
        parser.error("Epoch limits, intervals, patience, and sample counts must be positive.")
    if min(args.lambda_sc, args.lambda_sp, args.lambda_st) < 0 or args.lr <= 0:
        parser.error("Require nonnegative penalty values and a positive learning rate.")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices are cpu and cuda.")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Using CPU.", flush=True)
        device = torch.device("cpu")
    dataset, num_classes = load_dataset(args.data)
    base_data = dataset[0].clone()
    base_data.x = F.normalize(base_data.x.float(), p=2, dim=-1)
    base_data.y = base_data.y.reshape(-1).long()
    from torch_geometric.utils import coalesce

    base_data.edge_index = coalesce(base_data.edge_index.long(), num_nodes=base_data.x.size(0))
    count = split_count(base_data)
    indices = list(range(count)) if args.all_splits else [args.split_idx]
    modality = "feature_only" if args.feature_only else "label_aware"
    output_dir = Path(args.output_dir) / f"data_{args.data}_{modality}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split_idx in indices:
        data = select_split(base_data, split_idx).to(device)
        results.append(train_one_split(data, num_classes, split_idx, args, output_dir))
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores = np.asarray([result["test_score_percent"] for result in results], dtype=float)
    summary = {
        "data_id": args.data,
        "input_modality": modality,
        "num_runs": len(results),
        "test_metric": results[0]["test_metric"],
        "test_score_mean_percent": float(scores.mean()),
        "test_score_std_percent": float(scores.std(ddof=1)) if scores.size > 1 else None,
        "runs": results,
    }
    summary_path = output_dir / f"summary_seed_{args.seed}_splits_{'_'.join(map(str, indices))}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
