"""Decay after exactly N validations without a new minimum; checkpointable."""
import math

class PlateauDecay:
    def __init__(self, optimizer, patience=5, factor=.5, min_lr=1e-5):
        if patience < 1 or not 0 < factor < 1 or min_lr <= 0:
            raise ValueError('Invalid schedule')
        self.optimizer=optimizer
        self.patience=patience
        self.factor=factor
        self.min_lr=min_lr
        self.best=math.inf
        self.bad_epochs=0
        self.steps=0

    def step(self, value):
        value=float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite validation metric; refusing scheduler update')
        self.steps+=1
        if value < self.best:
            self.best=value
            self.bad_epochs=0
        else:
            self.bad_epochs+=1
        if self.bad_epochs >= self.patience:
            for group in self.optimizer.param_groups:
                group['lr']=max(group['lr']*self.factor,self.min_lr)
            self.bad_epochs=0

    def state_dict(self):
        return {k:getattr(self,k) for k in ('patience','factor','min_lr','best','bad_epochs','steps')}

    def load_state_dict(self, state):
        for k in ('patience','factor','min_lr'):
            if state[k]!=getattr(self,k):raise ValueError('Scheduler configuration changed: '+k)
        for k in ('best','bad_epochs','steps'):setattr(self,k,state[k])
