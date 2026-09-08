#!/usr/bin/env python3
"""Sequential launcher for the pruned standalone baseline files.

Each model remains implemented only in its own sibling Python file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASELINES: Dict[str, Tuple[str, str]] = {
    'mlp': ('MLP', 'mlp.py'),
    'gcn': ('GCN', 'gcn.py'),
    'gat': ('GAT', 'gat.py'),
    'gcnii': ('GCNII', 'gcnii.py'),
    'h2gcn': ('H2GCN', 'h2gcn.py'),
    'geom_gcn': ('Geom-GCN', 'geom_gcn.py'),
    'gprgnn': ('GPRGNN', 'gprgnn.py'),
    'fagcn': ('FAGCN', 'fagcn.py'),
    'acm_gcn': ('ACM-GCN', 'acm_gcn.py'),
    'glognn': ('GloGNN', 'glognn.py'),
    'gbk_gnn': ('GBK-GNN', 'gbk_gnn.py'),
    'auto_heg': ('Auto-HeG', 'auto_heg.py'),
    'pcnet': ('PCNet', 'pcnet.py'),
    'tfe_gnn': ('TFE-GNN', 'tfe_gnn.py'),
    'cgnn': ('CGNN', 'cgnn.py'),
    'l2dgcn': ('L2DGCN', 'l2dgcn.py'),
    'gesc': ('GESC', 'gesc.py'),
}
ALIASES: Dict[str, str] = {
    'mlp': 'mlp',
    'gcn': 'gcn',
    'gat': 'gat',
    'gcnii': 'gcnii',
    'h2gcn': 'h2gcn',
    'geomgcn': 'geom_gcn',
    'gprgnn': 'gprgnn',
    'fagcn': 'fagcn',
    'acmgcn': 'acm_gcn',
    'glognn': 'glognn',
    'gbkgnn': 'gbk_gnn',
    'autoheg': 'auto_heg',
    'pcnet': 'pcnet',
    'tfegnn': 'tfe_gnn',
    'cgnn': 'cgnn',
    'l2dgcn': 'l2dgcn',
    'gesc': 'gesc',
}


def normalize_model(value: str) -> str:
    key = value.strip().lower().replace('-', '').replace('_', '')
    if key in ALIASES:
        return ALIASES[key]
    if value.strip().lower() in BASELINES:
        return value.strip().lower()
    raise ValueError(
        f'unknown model {value!r}; available: {", ".join(BASELINES)}'
    )


def parse_models(spec: str) -> List[str]:
    if spec.strip().lower() in {'all', '*'}:
        return list(BASELINES)
    selected: List[str] = []
    for part in spec.split(','):
        if part.strip():
            slug = normalize_model(part)
            if slug not in selected:
                selected.append(slug)
    if not selected:
        raise ValueError('no models selected')
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run multiple pruned standalone graph baselines sequentially.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data', nargs='?', type=int)
    parser.add_argument('--models', default='all', help='all or comma-separated names')
    parser.add_argument(
        '--launcher-fail-fast', action='store_true',
        help='stop launching additional model files after the first nonzero exit code',
    )
    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    try:
        selected = parse_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    if args.data is None and '--self-test' not in passthrough:
        parser.error('dataset index is required unless --self-test is passed through')

    root = Path(__file__).resolve().parent
    failures: List[Tuple[str, int]] = []
    for slug in selected:
        model_name, filename = BASELINES[slug]
        script = root / filename
        if not script.is_file():
            parser.error(f'missing sibling baseline file: {script}')
        command = [sys.executable, str(script)]
        if args.data is not None:
            command.append(str(args.data))
        command.extend(passthrough)
        print('\n' + '=' * 88)
        print(f'Launching {model_name}: {" ".join(command)}')
        print('=' * 88)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append((model_name, result.returncode))
            if args.launcher_fail_fast:
                break

    if failures:
        print('\nFailed baselines:', file=sys.stderr)
        for model_name, returncode in failures:
            print(f'  {model_name}: exit code {returncode}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(run_cli())
