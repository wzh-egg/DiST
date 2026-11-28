"""Train DiST on AIR and select a checkpoint by validation MAE."""
import csv
import json
import math
import signal
import sys
import threading
import time

from src.cli import parse_args

def write_history(out,rows):
    if not rows:return
    temp=out/'history.csv.tmp'
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(out/'history.csv')


def train(args):
    import numpy as np
    import torch
    from src.scheduler import PlateauDecay
    from src.engine import train_update, evaluate, capture_rng, restore_rng, atomic_checkpoint
    from src.experiment import setup
    from src.input_pipeline import Pipeline
    from src.optimizer import make_optimizer
    from src.utils import batches, write_json
    job, device, source, base = setup(args)
    data, out = source.station, args.out
    phase=['starting training'];stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):print('RUNNING: '+phase[0],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        optimizer=make_optimizer(base,job)
        scheduler=PlateauDecay(optimizer, patience=job['lr_patience'], factor=job['lr_factor'], min_lr=job['min_lr'])
        scaler=torch.amp.GradScaler('cuda',init_scale=128.,enabled=job['amp'])
        rng=np.random.default_rng(job['seed']);rows=[];best=math.inf;best_epoch=stale=completed=0
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            base.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng'],rng);rows=saved['history'];best=saved['best']
            best_epoch=saved['best_epoch'];stale=saved['stale'];completed=saved['epoch']
            write_history(out,rows)
            if best_epoch==completed:
                atomic_checkpoint(out/'best.pt',saved)
                write_json(out/'best_validation.json',dict(epoch=completed,**saved['validation']))
            print(f'RESUME epoch{completed+1}; best={best_epoch}; stale={stale}/{job["patience"]}',flush=True)
            del saved
        model=torch.compile(base,fullgraph=False,dynamic=False) if job['compile'] else base
        pause=[False]
        def request_pause(*_):
            pause[0]=True;print('Pause requested; finish and save the current epoch.',flush=True)
        signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
        for name in ('PAUSE_REQUEST','PAUSED'):(out/name).unlink(missing_ok=True)
        for epoch in range(completed+1,job['epochs']+1):
            if stale>=job['patience']:break
            model.train();torch.cuda.reset_peak_memory_stats(device)
            origins=rng.permutation(data.origins['train']);groups=list(batches(origins,32))
            micro_groups=[m for group in groups for m in batches(group,args.micro)]
            absolute=count=retries=0; objective_sum=0.;began=time.perf_counter();lr=optimizer.param_groups[0]['lr']
            phase[0]=f'epoch{epoch} first batch; initial compilation may take several minutes'
            with Pipeline(source,micro_groups,'prefetch',device) as pipe:
                for index,group in enumerate(groups,1):
                    result=train_update(model,optimizer,source,group,args.micro,device,scaler,pipe,job['gradient_clip'],profile_update=False,relative_weight=job['relative_weight'],relative_floor=job['relative_floor'])
                    absolute+=result['absolute'];count+=result['count'];retries+=result['retries'];objective_sum+=result['objective']*result['count']
                    phase[0]=f'epoch{epoch} train{index}/{len(groups)}'
                    if index==1 or index%20==0 or index==len(groups):
                        print(f'Epoch{epoch} update{index}/{len(groups)}; train MAE={absolute/count:.6f}; elapsed={time.perf_counter()-began:.1f}s',flush=True)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize(device)
            train_seconds=time.perf_counter()-began;peak=torch.cuda.max_memory_allocated(device)/2**30
            phase[0]=f'epoch{epoch} validation'
            validation=evaluate(model,source,data.origins['val'],args.eval_batch,device)
            value=validation['model']['mae'];improved=value<best
            if improved:best,best_epoch,stale=value,epoch,0
            else:stale+=1
            scheduler.step(value)
            row=dict(epoch=epoch,train_mae=absolute/count,train_loss=objective_sum/count,relative_weight=job['relative_weight'],val_mae=value,val_rmse=validation['model']['rmse'],val_mape=validation['model']['mape_percent'],
                     lr=lr,next_lr=optimizer.param_groups[0]['lr'],best_epoch=best_epoch,stale=stale,
                     micro=args.micro,micro_batch=args.micro,eval_batch=args.eval_batch,train_seconds=train_seconds,
                     val_seconds=validation['seconds'],peak_allocated_GiB=peak,overflow_retries=retries)
            for hour in (3,6,12):
                for metric in ('mae','rmse','mape_percent'):
                    row[f'val_{hour}h_{metric}']=validation['model']['per_hour'][hour-1][metric]
            rows.append(row)
            saved=dict(epoch=epoch,model=base.state_dict(),optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=capture_rng(rng,True),
                       history=rows,best=best,best_epoch=best_epoch,stale=stale,validation=validation)
            atomic_checkpoint(out/'last.pt',saved)
            if improved:
                atomic_checkpoint(out/'best.pt',saved)
                write_json(out/'best_validation.json',dict(epoch=epoch,**validation))
            del saved;write_history(out,rows);write_json(out/'status.json',dict(row,complete=False))
            print(f'Epoch{epoch}: val MAE={value:.6f}; best={best_epoch}; stale={stale}/{job["patience"]}; train/val={train_seconds:.1f}/{validation["seconds"]:.1f}s; peak={peak:.3f}GiB; lr={lr:g}; next_lr={row["next_lr"]:g}',flush=True)
            if pause[0] or (out/'PAUSE_REQUEST').exists():
                (out/'PAUSED').write_text('Whole epoch checkpoint saved.\n');return 75
        write_json(out/'status.json',dict(rows[-1],complete=True,best_epoch=best_epoch,best_val_mae=best,epochs=len(rows)))
        (out/'COMPLETE').write_text('Training complete; validation-best checkpoint selected.\n')
        print('TRAIN COMPLETE:',out,flush=True);return 0
    finally:stop.set()


if __name__ == "__main__":
    args = parse_args(__doc__)
    import torch
    try:
        sys.exit(train(args))
    except torch.OutOfMemoryError:
        print("CUDA out of memory. Resume from the last completed epoch.", flush=True)
        sys.exit(42)
    except RuntimeError as error:
        if "CUDA out of memory" in str(error):
            print("CUDA out of memory. Resume from the last completed epoch.", flush=True)
            sys.exit(42)
        raise
