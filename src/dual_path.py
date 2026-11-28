"""Two independent state paths, shared weights; local-ring and global Top50."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.regional_attention import CardAttentionInteraction, DirectCardRound
from src.attention_helpers import prepare_mask, project_gather
from src.attention import one_card, pool_forward, single_query



@torch.compiler.disable
def spatial_bias_fp32(module, relative):
    # Keep this small MLP out of Inductor's mixed-precision graph.
    # No detach: both Linear layers and distance_raw still receive gradients.
    with torch.autocast(device_type=relative.device.type, enabled=False):
        return module(relative.float()).float()


class CellSummaries(nn.Module):
    def __init__(self, old, dim):
        super().__init__()
        self.spatial_score=old.spatial_score;self.history_card=old.history_card
        self.station_norm=old.station_norm;self.seed_norm=old.seed_norm
        self.station_pool=old.station_pool
        self.station_seed=nn.Parameter(old.station_seeds[:1].detach().clone())
        self.fuse=nn.Linear(2*dim,dim)

    def forward(self,station,grid,grid_valid,geo):
        grouped=grid[:,geo.group_ids]
        valid=grid_valid[:,geo.group_ids]&geo.group_valid[None,:,:,None]
        block_valid=valid.any(-1)
        score=self.spatial_score(grouped.flatten(-2)).squeeze(-1).float()
        alpha=score.masked_fill(~block_valid,-1e4).softmax(-1)*block_valid
        weights=alpha[...,None]*valid
        weights=weights/weights.sum(-2,keepdim=True).clamp_min(1e-12)
        history=(weights[...,None].to(grouped.dtype)*grouped).sum(-3)
        delta=history[:,:,1:]-history[:,:,:-1]
        grid_card=self.history_card(torch.cat((history.flatten(-2),delta.flatten(-2)),-1))
        gv=valid.any(-1).any(-1);grid_card=grid_card*gv[...,None]
        source=self.station_norm(station)[:,geo.cell_station_ids]
        smask=geo.cell_station_valid[None].expand(len(station),-1,-1)
        seed=self.seed_norm(self.station_seed)[None,None].expand(len(station),len(geo.group_ids),-1,-1)
        scard=pool_forward(self.station_pool,seed,source,smask).squeeze(-2)
        available=gv|smask.any(-1)
        cards=self.fuse(torch.cat((grid_card,scard),-1))*available[...,None]
        return cards,available


class CellRound(DirectCardRound):
    def __init__(self,old,dim):
        nn.Module.__init__(self)
        for name,module in old.named_children():self.add_module(name,module)
        self.regional=CellSummaries(old.regional,dim)

    def exchange(self,cards,valid,geo,external):
        update=self.regional_read;a=update.attention
        def heads(x):return x.unflatten(-1,(a.heads,-1)).transpose(1,2)
        q=heads(a.query(update.q_norm(cards)));k,v=a.project_kv(update.kv_norm(cards))
        with torch.autocast(device_type=cards.device.type,enabled=False):
            scores=q.float()@heads(k).float().transpose(-2,-1)/math.sqrt(q.shape[-1])
            scores=scores+spatial_bias_fp32(self.spatial_bias,geo.cell_relative)[None]
            eligible=valid[:,None,:].expand(-1,len(geo.group_ids),-1)
            if external:
                eligible=eligible&~torch.eye(len(geo.group_ids),device=cards.device,dtype=torch.bool)[None]
                ranked=scores.detach().mean(1).masked_fill(~eligible,-torch.inf)
                indices=ranked.topk(min(50,len(geo.group_ids)-1),dim=-1).indices
                keep=eligible&torch.zeros_like(eligible).scatter_(-1,indices,True)
            else:
                keep=eligible&geo.cell_adjacent[None]
            available=keep.any(-1)
            scores=torch.where(available[:,None,:,None],scores.masked_fill(~keep[:,None],-torch.inf),0.)
            weights=F.dropout(scores.softmax(-1),a.dropout,training=a.training)
            message=weights@heads(v).float()
        message=a.output(message.transpose(1,2).reshape_as(cards).to(cards.dtype))
        message=message*available[...,None]
        result=update.ffn(cards+update.drop(message))*valid[...,None]
        return result,keep

    def step(self,station,grid,valid,geo,masks,kv_mode,external,details=False):
        cards,cv=self.regional(station,grid,valid,geo)
        cards,keep=self.exchange(cards,cv,geo,external)
        reader=self.grid_read
        ids=geo.block_group[:,None]
        k,v=project_gather(reader.attention,reader.kv_norm(cards),ids,False,kv_mode)
        message=one_card(reader.attention,reader.q_norm(grid),k,v,prepare_mask(cv[:,ids]))
        grid=reader.ffn(grid+reader.drop(message))*valid[...,None]
        s=self.station_norm(station)
        k,v=project_gather(self.station_attention,s,geo.station_neighbors,False,kv_mode)
        sm=single_query(self.station_attention,s[:,:,None],k,v,masks[0]).squeeze(-2)
        k,v=project_gather(self.grid_attention,self.grid_norm(grid),geo.candidate_to_unique,True,kv_mode)
        gm=single_query(self.grid_attention,s[:,:,None],k,v,masks[1]).squeeze(-2)
        station=self.station_ffn(station+self.station_drop(self.station_fuse(torch.cat((sm,gm),-1))))
        return station,grid,keep if details else None


class DualPathModel(CardAttentionInteraction):
    def __init__(self,geometry,gate_mode='half',fixed_alpha=.5,**kwargs):
        super().__init__(geometry,**kwargs)
        self.interactions=nn.ModuleList(CellRound(old,self.hidden) for old in self.interactions)
        self.gate_mode=gate_mode
        self.fixed_alpha=float(fixed_alpha)
        if not 0 < self.fixed_alpha < 1:raise ValueError("alpha must be between 0 and 1")
        self.register_buffer('fixed_leads',torch.arange(1,13,dtype=torch.float32)/12)
        # Construct the learned network last so the two fixed controls share
        # identical initial backbone/head tensors with the learned trial.
        if gate_mode=='learned':
            with torch.random.fork_rng(devices=[]):
                self.gate_norm=nn.LayerNorm(2*self.hidden)
                self.lead_embedding=nn.Embedding(12,8)
                self.gate_net=nn.Sequential(nn.Linear(2*self.hidden+8,self.hidden),nn.GELU(),nn.Linear(self.hidden,self.hidden))
                nn.init.zeros_(self.gate_net[-1].weight);nn.init.zeros_(self.gate_net[-1].bias)
                prior=torch.linspace(.1,.9,12)
        elif gate_mode not in ('half','ramp','fixed'):raise ValueError(gate_mode)

    def gates(self,local,global_state):
        if self.gate_mode=='fixed':return local.new_full((*local.shape[:2],12,self.hidden),self.fixed_alpha)
        if self.gate_mode=='half':return local.new_full((*local.shape[:2],12,self.hidden),.5)
        if self.gate_mode=='ramp':return self.fixed_leads[None,None,:,None].expand(*local.shape[:2],12,self.hidden)
        features=self.gate_norm(torch.cat((local,global_state),-1)).unsqueeze(-2).expand(-1,-1,12,-1)
        lead=self.lead_embedding.weight[None,None].expand(*local.shape[:2],-1,-1)
        return (self.gate_net(torch.cat((features,lead),-1)).float()+self.gate_prior[None,None,:,None]).sigmoid()
