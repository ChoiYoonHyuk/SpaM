#!/usr/bin/env python3
"""Standalone, model-only PCNet baseline runner.

Example: python pcnet.py 0
Dataset indices: 0=Cora, 1=Citeseer, 2=Pubmed.
Only the PCNet implementation and the shared data/training path needed by it are included.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MODEL_NAME = 'PCNet'
MODEL_SLUG = 'pcnet'

DATASET_NAMES: Dict[int, str] = {
    0: 'Cora',
    1: 'Citeseer',
    2: 'Pubmed',
    3: 'ChameleonFiltered',
    4: 'SquirrelFiltered',
    5: 'Actor',
    6: 'Cornell',
    7: 'Texas',
    8: 'Wisconsin',
}
PLANETOID_NAMES: Dict[int, str] = {0: 'Cora', 1: 'CiteSeer', 2: 'PubMed'}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_features(x: Tensor) -> Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=-1.0)


def coalesce_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    if edge_index.numel() == 0:
        return edge_index.long().reshape(2, 0)
    edge_index = edge_index.long()
    flat = edge_index[0] * int(num_nodes) + edge_index[1]
    flat = torch.unique(flat, sorted=True)
    return torch.stack([flat.div(int(num_nodes), rounding_mode='floor'), flat.remainder(int(num_nodes))], dim=0)


def to_undirected_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    return coalesce_edge_index(torch.cat([edge_index, edge_index.flip(0)], dim=1), num_nodes)


def add_remaining_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    device = edge_index.device
    edge_index = coalesce_edge_index(edge_index, num_nodes)
    present = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    src, dst = edge_index
    mask = src == dst
    if mask.any():
        present[src[mask]] = True
    missing = torch.nonzero(~present, as_tuple=False).view(-1)
    if missing.numel() > 0:
        loops = torch.stack([missing, missing], dim=0)
        edge_index = coalesce_edge_index(torch.cat([edge_index, loops], dim=1), num_nodes)
    return edge_index


def propagate(x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor]=None, reduce: str='sum') -> Tensor:
    src, dst = edge_index
    msg = x[src]
    if edge_weight is not None:
        msg = msg * edge_weight.to(dtype=x.dtype).view(-1, *[1] * (msg.dim() - 1))
    out = x.new_zeros((x.size(0),) + x.shape[1:])
    out.index_add_(0, dst, msg)
    if reduce == 'mean':
        deg = x.new_zeros(x.size(0))
        deg.index_add_(0, dst, torch.ones(dst.numel(), device=x.device, dtype=x.dtype))
        out = out / deg.clamp_min(1.0).view(-1, *[1] * (out.dim() - 1))
    elif reduce != 'sum':
        raise ValueError(f'unsupported reduction: {reduce}')
    return out


def charlier_values(max_degree: int, x: int, lam: float) -> List[float]:
    vals = [1.0]
    if max_degree <= 1:
        return vals
    vals.append(float(x) - lam)
    for n in range(2, max_degree):
        vals.append((float(x) - n - lam + 1.0) * vals[-1] - (n - 1.0) * lam * vals[-2])
    return vals


@torch.no_grad()
def score_accuracy(scores: Tensor, y: Tensor, mask: Tensor) -> float:
    mask = mask.bool()
    if mask.sum().item() == 0:
        return float('nan')
    pred = scores.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


@dataclass
class GraphData:
    data_id: int
    name: str
    split_idx: int
    num_splits: int
    x: Tensor
    y: Tensor
    edge_index: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    num_nodes: int
    num_features: int
    num_classes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> 'GraphData':
        return GraphData(data_id=self.data_id, name=self.name, split_idx=self.split_idx, num_splits=self.num_splits, x=self.x.to(device), y=self.y.to(device), edge_index=self.edge_index.to(device), train_mask=self.train_mask.to(device), val_mask=self.val_mask.to(device), test_mask=self.test_mask.to(device), num_nodes=self.num_nodes, num_features=self.num_features, num_classes=self.num_classes, metadata=dict(self.metadata))


class FilteredWikipediaDataset:

    def __init__(self, root: str, name: str):
        name = name.lower()
        if name not in ('chameleon', 'squirrel'):
            raise ValueError(f'unsupported filtered Wikipedia dataset: {name}')
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        filename = f'{name}_filtered_directed.npz'
        path = root_path / filename
        if not path.exists():
            url = 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/' + filename
            print(f'[download] {url} -> {path}')
            urllib.request.urlretrieve(url, path)
        with np.load(path, allow_pickle=False) as arrays:
            self.x = torch.from_numpy(arrays['node_features']).float()
            self.y = torch.from_numpy(arrays['node_labels']).long().view(-1)
            self.edge_index = torch.from_numpy(arrays['edges']).long().t().contiguous()
            self.train_mask = torch.from_numpy(arrays['train_masks']).bool()
            self.val_mask = torch.from_numpy(arrays['val_masks']).bool()
            self.test_mask = torch.from_numpy(arrays['test_masks']).bool()
        if self.train_mask.dim() == 2 and self.train_mask.size(0) != self.x.size(0):
            self.train_mask = self.train_mask.t().contiguous()
            self.val_mask = self.val_mask.t().contiguous()
            self.test_mask = self.test_mask.t().contiguous()
        expected = {'chameleon': 890, 'squirrel': 2223}
        if self.x.size(0) != expected[name]:
            raise ValueError(f'{filename}: expected {expected[name]} nodes, got {self.x.size(0)}')
        self.num_node_features = int(self.x.size(1))
        self.num_classes = int(self.y.max().item()) + 1
        self.name = name


def _import_pyg_datasets():
    try:
        from torch_geometric.datasets import Actor, Planetoid, WebKB
        from torch_geometric.transforms import NormalizeFeatures
    except Exception as exc:
        raise RuntimeError(
            'torch-geometric is needed for dataset loading. Install a build matching '
            'your PyTorch, for example: pip install torch-geometric'
        ) from exc
    return Planetoid, NormalizeFeatures, Actor, WebKB


def _select_mask(mask: Tensor, split_idx: int, num_nodes: int) -> Tuple[Tensor, int]:
    if mask.dim() == 1:
        if int(split_idx) != 0:
            raise ValueError('this dataset exposes a single split; use --split 0 or the default --split -1')
        return mask.bool(), 1
    if mask.size(0) != num_nodes and mask.size(1) == num_nodes:
        mask = mask.t().contiguous()
    num_splits = int(mask.size(1))
    if not 0 <= split_idx < num_splits:
        raise ValueError(f'split_idx={split_idx} outside [0,{num_splits})')
    return mask[:, split_idx].bool(), num_splits


def load_graph(data_id: int, split_idx: int = 0, root: str = './data') -> GraphData:
    data_id = int(data_id)
    if data_id not in DATASET_NAMES:
        valid = ', '.join(f'{idx}={name}' for idx, name in DATASET_NAMES.items())
        raise ValueError(f'unknown dataset index {data_id}; available: {valid}')

    Planetoid, NormalizeFeatures, Actor, WebKB = _import_pyg_datasets()
    root_path = Path(root)

    if data_id in PLANETOID_NAMES:
        pyg_name = PLANETOID_NAMES[data_id]
        name = DATASET_NAMES[data_id]
        dataset = Planetoid(
            root=str(root_path / name),
            name=pyg_name,
            split='public',
            transform=NormalizeFeatures(),
        )
        data = dataset[0]
    elif data_id == 3:
        dataset = FilteredWikipediaDataset(
            str(root_path / 'ChameleonFilteredDirected'), 'chameleon'
        )
        data = dataset
        name = 'ChameleonFiltered'
    elif data_id == 4:
        dataset = FilteredWikipediaDataset(
            str(root_path / 'SquirrelFilteredDirected'), 'squirrel'
        )
        data = dataset
        name = 'SquirrelFiltered'
    elif data_id == 5:
        dataset = Actor(root=str(root_path / 'Actor'))
        data = dataset[0]
        name = 'Actor'
    elif data_id == 6:
        dataset = WebKB(root=str(root_path / 'Cornell'), name='Cornell')
        data = dataset[0]
        name = 'Cornell'
    elif data_id == 7:
        dataset = WebKB(root=str(root_path / 'Texas'), name='Texas')
        data = dataset[0]
        name = 'Texas'
    else:
        dataset = WebKB(root=str(root_path / 'Wisconsin'), name='Wisconsin')
        data = dataset[0]
        name = 'Wisconsin'

    x = sanitize_features(data.x)
    y = data.y.long().view(-1)
    edge_index = data.edge_index.long().contiguous()
    num_nodes = int(x.size(0))
    train_mask, n1 = _select_mask(data.train_mask, split_idx, num_nodes)
    val_mask, n2 = _select_mask(data.val_mask, split_idx, num_nodes)
    test_mask, n3 = _select_mask(data.test_mask, split_idx, num_nodes)
    num_splits = max(n1, n2, n3)
    edge_index = to_undirected_edges(edge_index, num_nodes)
    edge_index = add_remaining_self_loops(edge_index, num_nodes)
    num_classes = int(getattr(dataset, 'num_classes', int(y.max().item()) + 1))
    num_features = int(x.size(1))
    protocol = (
        'planetoid-public+bidirectional+self_loops'
        if data_id in PLANETOID_NAMES
        else 'bidirectional+self_loops'
    )
    metadata = {'benchmark_protocol': protocol, 'metric': 'accuracy'}
    return GraphData(
        data_id=data_id,
        name=name,
        split_idx=int(split_idx),
        num_splits=num_splits,
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        metadata=metadata,
    )


def synthetic_graph(seed: int=0) -> GraphData:
    set_seed(seed)
    n, f, c = (36, 10, 3)
    x = torch.randn(n, f)
    y = torch.arange(n) % c
    src = torch.randint(0, n, (140,))
    dst = torch.randint(0, n, (140,))
    edge = add_remaining_self_loops(to_undirected_edges(torch.stack([src, dst]), n), n)
    perm = torch.randperm(n)
    train = torch.zeros(n, dtype=torch.bool)
    train[perm[:18]] = True
    val = torch.zeros(n, dtype=torch.bool)
    val[perm[18:27]] = True
    test = ~(train | val)
    return GraphData(8, 'Synthetic', 0, 1, x, y, edge, train, val, test, n, f, c, {'metric': 'accuracy'})


class PCPropagation(nn.Module):

    def __init__(self, conv_layer: int, n_poly: int, alpha: float, a: float, b: float, c: float):
        super().__init__()
        self.k = int(conv_layer)
        self.n_poly = int(n_poly)
        self.a, self.b, self.c = (float(a), float(b), float(c))
        self.temp = nn.Parameter(torch.full((self.k + 1,), float(alpha)))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.size(0)
        loops = torch.arange(n, device=x.device)
        edge0 = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
        row, col = edge0
        one = x.new_ones(row.numel())
        deg1 = x.new_zeros(n)
        deg1.index_add_(0, row, one)
        deg2 = x.new_zeros(n)
        deg2.index_add_(0, col, one)
        norm = deg1.clamp_min(1.0).pow(-self.c)[row] * deg2.clamp_min(1.0).pow(-self.c)[col]
        edge1 = torch.cat([edge0, torch.stack([loops, loops])], dim=1)
        norm1 = torch.cat([-norm, x.new_full((n,), self.b)])
        powers = [x]
        h = x
        for _ in range(self.n_poly):
            h = propagate(-h, edge1, norm1)
            powers.append(h)
        out = self.temp[0].to(x.dtype) * x
        for i in range(1, self.k + 1):
            coeffs = charlier_values(self.n_poly, i, self.a)
            basis = x.new_zeros(x.shape)
            for j in range(self.n_poly):
                basis = basis + coeffs[j] / math.factorial(j) * powers[j]
            out = out + self.temp[i].to(x.dtype) * basis
        return out


class PCNetModel(nn.Module):

    def __init__(self, graph: GraphData, params: Mapping[str, float]):
        super().__init__()
        self.params_cfg = dict(params)
        self.lin1 = nn.Linear(graph.num_features, 64)
        self.lin2 = nn.Linear(64, graph.num_classes)
        self.prop = PCPropagation(int(params['conv_layer']), int(params['n_poly']), float(params['alpha']), float(params['a']), float(params['b']), float(params['c']))
        self.dropout1 = float(params['dpb1'])
        self.dropout2 = float(params['dpb2'])
        self.dprate = float(params['dpt1'])

    def forward(self, graph: GraphData) -> Tensor:
        x = F.dropout(graph.x, p=self.dropout1, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop(x, graph.edge_index)

    def optimizer_groups(self, config: 'TrainConfig') -> List[Dict[str, Any]]:
        p = self.params_cfg
        return [{'params': self.lin1.parameters(), 'lr': float(p['lr1']), 'weight_decay': float(p['wd1'])}, {'params': self.lin2.parameters(), 'lr': float(p['lr3']), 'weight_decay': float(p['wd3'])}, {'params': self.prop.parameters(), 'lr': float(p['lr2']), 'weight_decay': float(p['wd2'])}]


PCNET_PARAMS: Dict[int, Dict[str, float]] = {3: dict(conv_layer=6, n_poly=4, alpha=0.5, a=2.0, b=1.8, c=0.35, dpb1=0.8, dpt1=0.05, dpb2=0.9, lr1=0.1, lr2=0.05, lr3=0.005, wd1=0.001, wd2=0.0005, wd3=0.001, early_stop=200), 4: dict(conv_layer=5, n_poly=4, alpha=0.5, a=1.7, b=1.8, c=0.45, dpb1=0.85, dpt1=0.0, dpb2=0.0, lr1=0.1, lr2=0.1, lr3=0.005, wd1=5e-05, wd2=5e-06, wd3=5e-05, early_stop=300), 5: dict(conv_layer=6, n_poly=6, alpha=2.0, a=0.0, b=1.0, c=0.75, dpb1=0.0, dpt1=0.0, dpb2=0.85, lr1=0.1, lr2=0.1, lr3=0.05, wd1=0.005, wd2=5e-06, wd3=0.001, early_stop=300), 6: dict(conv_layer=4, n_poly=4, alpha=0.5, a=3.7, b=8.3, c=0.65, dpb1=0.05, dpt1=0.15, dpb2=0.8, lr1=0.05, lr2=0.005, lr3=0.005, wd1=0.002, wd2=0.005, wd3=0.0001, early_stop=200), 7: dict(conv_layer=4, n_poly=5, alpha=2.0, a=0.0, b=1.3, c=0.8, dpb1=0.1, dpt1=0.15, dpb2=0.7, lr1=0.01, lr2=0.01, lr3=0.01, wd1=0.002, wd2=5e-05, wd3=5e-06, early_stop=300), 8: dict(conv_layer=5, n_poly=4, alpha=0.5, a=1.6, b=7.3, c=0.8, dpb1=0.0, dpt1=0.05, dpb2=0.7, lr1=0.05, lr2=0.05, lr3=0.005, wd1=0.001, wd2=0.0001, wd3=0.0001, early_stop=300)}


def default_pc_params(data_id: int) -> Dict[str, float]:
    if data_id in PCNET_PARAMS:
        return dict(PCNET_PARAMS[data_id])
    return dict(conv_layer=6, n_poly=6, alpha=1.0, a=1.0, b=1.5, c=0.5, dpb1=0.5, dpt1=0.5, dpb2=0.5, lr1=0.01, lr2=0.01, lr3=0.01, wd1=0.0005, wd2=0.0, wd3=0.0005, early_stop=200)


@dataclass
class TrainConfig:
    epochs: int = 1000
    patience: int = 200
    lr: float = 0.01
    weight_decay: float = 0.0005
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.5
    grad_clip: float = 5.0
    eval_every: int = 1
    extras: Dict[str, Any] = field(default_factory=dict)


def model_config(
    data_id: int,
    quick: bool = False,
    epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
) -> TrainConfig:
    params = PCNET_PARAMS.get(data_id, {})
    cfg = TrainConfig(
        epochs=1500,
        patience=int(params.get('early_stop', 200)),
        hidden=64,
        layers=int(params.get('conv_layer', 6)),
        dropout=0.5,
    )
    if quick:
        cfg.epochs = min(cfg.epochs, 25)
        cfg.patience = min(cfg.patience, 8)
        cfg.hidden = min(cfg.hidden, 64)

    if epochs_override is not None and epochs_override > 0:
        cfg.epochs = int(epochs_override)
    if patience_override is not None and patience_override > 0:
        cfg.patience = int(patience_override)
    return cfg


def build_model(graph: GraphData, config: TrainConfig) -> nn.Module:
    extras = config.extras
    return PCNetModel(graph, default_pc_params(graph.data_id))


@dataclass
class TrainOutcome:
    model: str
    run: int
    seed: int
    split: int
    metric: str
    train: float
    val: float
    test: float
    best_epoch: int
    elapsed_sec: float
    status: str
    error: str = ''
    config: Dict[str, Any] = field(default_factory=dict)


def clone_state_cpu(module: nn.Module) -> Dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.optimizer_groups(config))


@torch.no_grad()
def evaluate_instance(
    model: nn.Module,
    graph: GraphData,
) -> Tuple[float, float, float]:
    model.eval()
    logits = model(graph)
    scores = logits
    return (
        score_accuracy(scores, graph.y, graph.train_mask),
        score_accuracy(scores, graph.y, graph.val_mask),
        score_accuracy(scores, graph.y, graph.test_mask),
    )


def train_model_instance(
    graph_cpu: GraphData,
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    run: int,
    split: int,
    verbose: bool = True,
) -> TrainOutcome:
    set_seed(seed)
    graph = graph_cpu.to(device)
    model = model.to(device)
    optimizer = make_optimizer(model, config)

    best_key = -float('inf')
    best_state: Optional[Dict[str, Tensor]] = None
    best_epoch = 0
    bad_epochs = 0
    start = time.time()
    for epoch in range(1, config.epochs + 1):

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'{MODEL_NAME}: non-finite loss at epoch {epoch}: {loss.item()}'
            )
        loss.backward()
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=False
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f'{MODEL_NAME}: non-finite gradient at epoch {epoch}'
                )
        optimizer.step()

        if epoch % config.eval_every != 0 and epoch != config.epochs:
            continue
        train_score, val_score, test_score = evaluate_instance(model, graph)
        key = val_score if math.isfinite(val_score) else -float('inf')
        improved = key > best_key
        if improved:
            best_key = key
            best_state = clone_state_cpu(model)
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += config.eval_every
        if verbose and (
            improved
            or epoch == 1
            or epoch == config.epochs
            or epoch % max(1, config.epochs // 10) == 0
        ):
            print(
                f'[{MODEL_NAME} e{epoch:04d}] loss={float(loss.item()):.4f} '
                f'train={train_score:.4f} val={val_score:.4f} '
                f'test={test_score:.4f} best={best_key:.4f}@{best_epoch}'
            )
        if bad_epochs >= config.patience:
            break
    if best_state is None:
        best_state = clone_state_cpu(model)
    model.load_state_dict(best_state)
    final_train, final_val, final_test = evaluate_instance(model, graph)
    elapsed = time.time() - start
    del optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return TrainOutcome(
        model=MODEL_NAME,
        run=run,
        seed=seed,
        split=split,
        metric='accuracy',
        train=final_train,
        val=final_val,
        test=final_test,
        best_epoch=best_epoch,
        elapsed_sec=elapsed,
        status='ok',
        config=asdict(config),
    )


def train_one(
    graph: GraphData,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    run: int,
    split: int,
) -> TrainOutcome:
    set_seed(seed)
    model = build_model(graph, config)
    return train_model_instance(
        graph, model, config, device, seed, run, split, verbose=True,
    )


def write_results(
    outcomes: Sequence[TrainOutcome],
    output_dir: Path,
    dataset_name: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{dataset_name}_{MODEL_SLUG}.csv'
    json_path = output_dir / f'{dataset_name}_{MODEL_SLUG}.json'
    rows = [asdict(outcome) for outcome in outcomes]
    fieldnames = [
        'model', 'run', 'seed', 'split', 'metric', 'train', 'val', 'test',
        'best_epoch', 'elapsed_sec', 'status',
        'error', 'config',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row['config'] = json.dumps(row['config'], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    payload = {
        'dataset': dataset_name,
        'model': MODEL_NAME,
        'generated_at_unix': time.time(),
        'runs': rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return csv_path, json_path


def print_summary(outcomes: Sequence[TrainOutcome]) -> None:
    print('\n' + '=' * 78)
    print('Final summary (test score at validation-selected checkpoint)')
    print('=' * 78)
    good = [
        outcome for outcome in outcomes
        if outcome.status == 'ok' and math.isfinite(outcome.test)
    ]
    if not good:
        errors = [outcome.error for outcome in outcomes if outcome.error]
        print(f'{MODEL_NAME}: ERROR - {errors[0] if errors else "unknown error"}')
    else:
        vals = np.asarray([outcome.val for outcome in good], dtype=float)
        tests = np.asarray([outcome.test for outcome in good], dtype=float)
        print(f'Model: {MODEL_NAME}')
        print(f'Runs: {len(good)}')
        print(f'Validation: {vals.mean():.4f} ± {vals.std(ddof=0):.4f}')
        print(f'Test:       {tests.mean():.4f} ± {tests.std(ddof=0):.4f}')
    print('=' * 78)


def parse_seed_list(spec: str) -> Optional[List[int]]:
    if not spec.strip():
        return None
    seeds = [int(value.strip()) for value in spec.split(',') if value.strip()]
    return seeds or None


def run_self_test() -> int:
    torch.set_num_threads(1)
    graph = synthetic_graph(7)
    device = torch.device('cpu')
    print(f'[self-test] {MODEL_NAME}')
    try:
        seed = 100
        set_seed(seed)
        config = model_config(
            graph.data_id, quick=True, epochs_override=1, patience_override=1
        )
        model = build_model(graph, config)
        outcome = train_model_instance(
            graph, model, config, device, seed, 0, 0,
            verbose=False,
        )
        if not math.isfinite(outcome.test):
            raise RuntimeError(f'non-finite test score: {outcome.test}')
        print(f'  ok: val={outcome.val:.4f}, test={outcome.test:.4f}')
        print('Self-test passed.')
        return 0
    except Exception as exc:
        print(f'  FAILED: {exc}')
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


def dataset_help() -> str:
    return ', '.join(f'{index}={name}' for index, name in DATASET_NAMES.items())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'Run the {MODEL_NAME} graph baseline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data', nargs='?', type=int, help=dataset_help())
    parser.add_argument('--runs', type=int, default=10, help='number of repeated runs')
    parser.add_argument('--seed', type=int, default=42, help='base seed')
    parser.add_argument(
        '--seeds', type=str, default='',
        help='explicit comma-separated seeds; overrides --runs and --seed',
    )
    parser.add_argument(
        '--split', type=int, default=-1,
        help='dataset split column; -1 cycles through available split columns',
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./baseline_results')
    parser.add_argument(
        '--epochs', type=int, default=-1,
        help="positive value overrides the method's epoch budget",
    )
    parser.add_argument(
        '--patience', type=int, default=-1,
        help="positive value overrides the method's patience",
    )
    parser.add_argument('--quick', action='store_true', help='small smoke-run profile')
    parser.add_argument(
        '--fail-fast', action='store_true', help='stop at the first failed run'
    )
    parser.add_argument(
        '--tracebacks', action='store_true',
        help='print full tracebacks for failed runs',
    )

    parser.add_argument(
        '--self-test', action='store_true',
        help='run a one-epoch synthetic test without downloading datasets',
    )
    return parser


def run_experiments(args: argparse.Namespace) -> int:
    data_id = int(args.data)
    if data_id not in DATASET_NAMES:
        raise ValueError(f'dataset index must be one of: {dataset_help()}')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but torch.cuda.is_available() is False')
    if device.type == 'cpu' and args.cpu_threads > 0:
        torch.set_num_threads(max(1, int(args.cpu_threads)))

    probe_split = 0 if args.split < 0 else int(args.split)
    probe = load_graph(data_id, probe_split, args.data_root)
    dataset_name = probe.name
    num_splits = probe.num_splits
    print(
        f'Dataset={dataset_name} id={probe.data_id} nodes={probe.num_nodes} '
        f'edges={probe.edge_index.size(1)} features={probe.num_features} '
        f'classes={probe.num_classes} splits={probe.num_splits} '
        f'metric=accuracy device={device}'
    )
    del probe

    explicit_seeds = parse_seed_list(args.seeds)
    seeds = (
        explicit_seeds
        if explicit_seeds is not None
        else [int(args.seed) + index for index in range(max(1, int(args.runs)))]
    )

    config = model_config(
        data_id,
        quick=bool(args.quick),
        epochs_override=args.epochs if args.epochs > 0 else None,
        patience_override=args.patience if args.patience > 0 else None,
    )
    print('\n' + '#' * 110)
    print(f'# {MODEL_NAME} | config={json.dumps(asdict(config), sort_keys=True)}')
    print('#' * 110)

    outcomes: List[TrainOutcome] = []
    output_dir = Path(args.output_dir)
    for run, seed in enumerate(seeds):
        split = int(args.split) if args.split >= 0 else run % max(1, num_splits)
        print(f'\n[{MODEL_NAME}] run={run + 1}/{len(seeds)} seed={seed} split={split}')
        graph: Optional[GraphData] = None
        try:
            graph = load_graph(data_id, split, args.data_root)
            set_seed(seed)
            outcome = train_one(graph, config, device, seed, run, split)
            outcomes.append(outcome)
            print(
                f'[{MODEL_NAME}] completed: val={outcome.val:.4f} '
                f'test={outcome.test:.4f} epoch={outcome.best_epoch} '
                f'time={outcome.elapsed_sec:.1f}s'
            )
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            outcomes.append(
                TrainOutcome(
                    model=MODEL_NAME,
                    run=run,
                    seed=seed,
                    split=split,
                    metric='accuracy',
                    train=float('nan'),
                    val=float('nan'),
                    test=float('nan'),
                    best_epoch=0,
                    elapsed_sec=0.0,
                    status='error',

                    error=error,
                    config=asdict(config),
                )
            )
            print(f'[{MODEL_NAME}] ERROR: {error}', file=sys.stderr)
            if args.tracebacks:
                traceback.print_exc()
            if args.fail_fast:
                raise
        finally:
            if graph is not None:
                del graph
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        csv_path, json_path = write_results(outcomes, output_dir, dataset_name)
        print(f'[checkpoint-results] {csv_path} | {json_path}')

    print_summary(outcomes)
    csv_path, json_path = write_results(outcomes, output_dir, dataset_name)
    print(f'CSV:  {csv_path.resolve()}')
    print(f'JSON: {json_path.resolve()}')
    return 0


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.data is None:
        parser.error('dataset index is required unless --self-test is used')
    if int(args.data) not in DATASET_NAMES:
        parser.error(f'dataset index must be one of: {dataset_help()}')
    try:
        return run_experiments(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == '__main__':
    raise SystemExit(run_cli())
