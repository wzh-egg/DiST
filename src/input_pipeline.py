"""Bounded exact-input staging. CUDA events guard host and device slot reuse.

No model, loss, candidate, mask, precision or sample order changes. A slot is
held until the consumer calls release(), after all uses have been enqueued.
"""
from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import torch

KEYS=('station','values','valid')


@dataclass
class HostBatch:
    origins: np.ndarray
    arrays: dict
    target: np.ndarray
    observed: np.ndarray
    last: np.ndarray
    read_seconds: float


class Source:
    def __init__(self,station,cache):
        self.station,self.cache=station,cache
        self.calls=0

    def shapes(self,batch):
        t,u,r,c=self.cache.values.shape
        return dict(station=(batch,2,self.station.N,self.station.H),
                    values=(batch,self.station.H,u,r,c),valid=(batch,self.station.H,u,r,c))

    def read(self,origins,out=None):
        began=time.perf_counter();origins=np.asarray(origins,dtype=np.int64)
        if origins.ndim!=1 or not len(origins):raise ValueError('Empty/invalid origins.')
        if (origins<self.station.H).any() or (origins+self.station.P>self.station.T).any():
            raise ValueError('History/forecast outside data timeline.')
        x,y,observed,last=self.station.batch(origins)
        if out is None:
            arrays=dict(station=x,
                values=np.stack([self.cache.values[int(o)-self.station.H:int(o)] for o in origins]),
                valid=np.stack([self.cache.valid[int(o)-self.station.H:int(o)] for o in origins]))
        else:
            arrays={k:out[k][:len(origins)] for k in KEYS}
            np.copyto(arrays['station'],x,casting='no')
            for i,o in enumerate(origins):
                np.copyto(arrays['values'][i],self.cache.values[int(o)-self.station.H:int(o)],casting='no')
                np.copyto(arrays['valid'][i],self.cache.valid[int(o)-self.station.H:int(o)],casting='no')
        self.calls+=1
        return HostBatch(origins.copy(),arrays,y,observed,last,time.perf_counter()-began)


def assemble(station,values,valid):
    patches=torch.stack((values.permute(0,2,1,3,4),
        valid.permute(0,2,1,3,4).to(dtype=values.dtype)),dim=3).contiguous()
    block_valid=valid.any(dim=(1,3,4))
    if not bool(torch.isfinite(patches).all()):raise ValueError('Nonfinite grid input.')
    return station,patches,block_valid


class Slot:
    def __init__(self,shapes,device,pinned=True):
        self.host={k:torch.empty(shape,dtype=torch.bool if k=='valid' else torch.float32,
                              pin_memory=pinned) for k,shape in shapes.items()}
        self.numpy={k:v.numpy() for k,v in self.host.items()}
        self.device={k:torch.empty(shape,dtype=self.host[k].dtype,device=device) for k,shape in shapes.items()}
        self.last_copy=None
        self.last_use=None


class Packet:
    def __init__(self,host,slot=None):
        self.host,self.slot=host,slot
        self.start=self.end=None
        self.device_arrays=None

    def tensors(self,device):
        if self.slot is None:
            # Keep the original three .to(device) calls as the baseline.
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            arrays={k:torch.from_numpy(self.host.arrays[k]).to(device) for k in KEYS}
            end.record();self.start,self.end=start,end
            return assemble(**arrays)
        stream=torch.cuda.current_stream(device)
        stream.wait_event(self.end)
        return assemble(**self.device_arrays)


class Pipeline:
    """original: pageable/serial; pinned_serial: one slot; prefetch: three slots.

    All GPU buffers are persistent until close() synchronizes both streams.
    Source pinned memory is not modified until its last H2D has completed;
    device memory is not overwritten until its last consumer has completed.
    """
    def __init__(self,source,groups,mode,device='cuda:0',slots=3):
        if mode not in ('original','pinned_serial','prefetch'):raise ValueError(mode)
        self.source=source;self.groups=[np.asarray(g,dtype=np.int64).copy() for g in groups]
        if not self.groups or any(not len(g) for g in self.groups):raise ValueError('Empty plan.')
        self.mode,self.device=mode,torch.device(device)
        self.position=0;self.active=None;self.packets=[];self.closed=False
        self.stop=threading.Event();self.thread=None
        self.free=queue.Queue();self.ready=queue.Queue(maxsize=slots)
        self.copy_stream=None;self.slots=[]
        if mode!='original':
            count=slots if mode=='prefetch' else 1
            self.slots=[Slot(source.shapes(max(map(len,self.groups))),self.device) for _ in range(count)]
            if mode=='prefetch':
                self.copy_stream=torch.cuda.Stream(device=self.device)
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                for slot in self.slots:self.free.put(slot)

    def __enter__(self):
        if self.mode=='prefetch':
            self.thread=threading.Thread(target=self._produce,name='aqi-pinned-prefetch',daemon=True)
            self.thread.start()
        return self

    def _put(self,q,item):
        while not self.stop.is_set():
            try:q.put(item,timeout=.1);return True
            except queue.Full:pass
        return False

    def _get(self,q):
        while not self.stop.is_set():
            try:return q.get(timeout=.1)
            except queue.Empty:pass
        raise RuntimeError('Pipeline stopped.')

    def _stage(self,slot,group,stream):
        # A completed consumer event alone does not authorize changing CPU memory
        # while DMA is reading it; keep an explicit last-copy guard as well.
        if slot.last_copy is not None:slot.last_copy.synchronize()
        host=self.source.read(group,slot.numpy)
        packet=Packet(host,slot);n=len(group)
        packet.device_arrays={k:v[:n] for k,v in slot.device.items()}
        with torch.cuda.stream(stream):
            if slot.last_use is not None:stream.wait_event(slot.last_use)
            packet.start=torch.cuda.Event(enable_timing=True);packet.end=torch.cuda.Event(enable_timing=True)
            packet.start.record(stream)
            for k in KEYS:packet.device_arrays[k].copy_(slot.host[k][:n],non_blocking=True)
            packet.end.record(stream)
        slot.last_copy=packet.end
        return packet

    def _produce(self):
        try:
            torch.cuda.set_device(self.device)
            for group in self.groups:
                slot=self._get(self.free)
                if not self._put(self.ready,self._stage(slot,group,self.copy_stream)):return
        except BaseException as error:
            if not self.stop.is_set():self._put(self.ready,error)

    def take(self,origins):
        if self.active is not None:raise RuntimeError('Release the previous packet first.')
        if self.position>=len(self.groups):raise StopIteration
        if not np.array_equal(origins,self.groups[self.position]):raise ValueError('Batch order changed.')
        if self.mode=='original':packet=Packet(self.source.read(origins))
        elif self.mode=='pinned_serial':
            packet=self._stage(self.slots[0],origins,torch.cuda.current_stream(self.device))
        else:
            packet=self._get(self.ready)
            if isinstance(packet,BaseException):raise packet
        if not np.array_equal(packet.host.origins,origins):raise ValueError('Prefetch origin order changed.')
        self.active=packet;self.position+=1;self.packets.append(packet)
        return packet

    def release(self):
        if self.active is None:raise RuntimeError('No active packet.')
        packet=self.active
        if packet.slot is not None:
            event=torch.cuda.Event();event.record(torch.cuda.current_stream(self.device))
            packet.slot.last_use=event
            if self.mode=='prefetch':self.free.put(packet.slot)
        # Do not retain pageable input/GPU arrays across steps through diagnostics.
        packet.device_arrays=None
        packet.host.arrays={}
        packet.host.target=packet.host.observed=packet.host.last=None
        self.active=None

    def stats(self):
        torch.cuda.synchronize(self.device)
        return dict(read_seconds=sum(p.host.read_seconds for p in self.packets),
                    h2d_cuda_interval_seconds=sum(p.start.elapsed_time(p.end) for p in self.packets)/1000,
                    batches=len(self.packets),windows=sum(len(p.host.origins) for p in self.packets),
                    pinned_host_bytes=sum(t.numel()*t.element_size() for s in self.slots for t in s.host.values()),
                    persistent_device_bytes=sum(t.numel()*t.element_size() for s in self.slots for t in s.device.values()))

    def close(self):
        if self.closed:return
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():raise RuntimeError('Prefetch reader did not stop within 30 seconds.')
        if self.copy_stream is not None:self.copy_stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.closed=True

    def __exit__(self,*exc):self.close()


class TrainingData:
    """Adapter for the unmodified frozen train_group; labels retain its transfer."""
    def __init__(self,station,pipeline):self.station,self.pipeline=station,pipeline

    def __getattr__(self,name):return getattr(self.station,name)

    def batch(self,origins):
        packet=self.pipeline.take(origins)
        h=packet.host
        return packet,h.target,h.observed,h.last
