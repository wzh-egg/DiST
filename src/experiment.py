"""Load AIR inputs and initialize the model for training or evaluation."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from model import DiST
from src.dataset import source_from
from src.geometry import load_geometry
from src.utils import write_json


def setup(args):
    args.out.mkdir(parents=True,exist_ok=True);out=args.out;job=json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'air.json').read_text())
    if args.micro!=16 or args.eval_batch!=16:raise ValueError('Keep micro/eval16 for this experiment')
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'compile_cache/inductor')
    os.environ['TRITON_CACHE_DIR']=str(out/'compile_cache/triton')
    torch.set_num_threads(4);random.seed(job['seed']);np.random.seed(job['seed'])
    torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=source_from(args.stations, args.grid);data=source.station
    geometry=load_geometry(args.geometry)
    base=DiST(geometry,gate_mode=job['gate_mode'],fixed_alpha=job['fixed_alpha'],kv_mode='packed',position_bias=False,**job['model_config'])
    base.to(device)
    config=dict(job,**job['model_config'])
    config.update(micro=args.micro,micro_batch=args.micro,eval_batch=args.eval_batch,
                data=str(data.root),windows={k:len(v) for k,v in data.origins.items()},
                station_mean=data.mean,station_std=data.std,nodes=data.N,
                cell_count=len(geometry['group_ids']),cards_per_cell=1,
                parameters=sum(p.numel() for p in base.parameters()),active_parameters=sum(p.numel() for p in base.parameters() if p.requires_grad),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0'),
                history_steps=12,horizon_steps=12,step_hours=1,grid_stages=4,active_rounds=len(base.interactions),grid_card_scope='own_cell_one',card_exchange='local_ring/global_top50',grid_cards_per_block=1,
                grid_frame_projection='800 ->32 ->ReLU; original single-layer projection',
                initialization='from scratch; two independent states, shared weights; one prediction head')
    write_json(out/'config.json',config)
    print('AIR DATA:',config['data'],config['windows'],flush=True)
    print(f"MODEL: probability mixture + horizon path gates; local-ring/global Top50; own-cell one-card grid write; four rounds; four grid history states; parameters={config['parameters']}",flush=True)
    print('FUSION:', job['gate_mode'], 'global alpha=', job['fixed_alpha'], 'relative lambda=', job['relative_weight'], flush=True)
    print(f"Adam lr={job['lr']}; WD={job['weight_decay']}; dropout.1; micro/effective={args.micro}/32; eval={args.eval_batch}; AMP+compile={job['compile']}",flush=True)
    return job, device, source, base
