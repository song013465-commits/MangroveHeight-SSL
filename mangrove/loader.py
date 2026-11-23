from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch
from torch.utils.data._utils.collate import default_collate

class ThreadedDataLoader:
    def __init__(self,dataset,batch_size,shuffle,threads=4,prefetch=2,pin_memory=True,seed=0):
        self.dataset,self.batch_size,self.shuffle=dataset,batch_size,shuffle
        self.threads,self.prefetch,self.pin_memory=threads,prefetch,pin_memory
        self.seed,self.epoch=seed,0
    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size
    def __iter__(self):
        indices=torch.randperm(len(self.dataset)).tolist() if self.shuffle else list(range(len(self.dataset)))
        groups=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        executor=ThreadPoolExecutor(max_workers=self.threads)
        pending=deque();offset=0
        epoch=self.epoch;self.epoch+=1
        def submit(group):
            return [executor.submit(self.dataset.__getitem__,(i,self.seed+epoch*1000003+i)) for i in group]
        try:
            while offset<len(groups) and len(pending)<self.prefetch:
                pending.append(submit(groups[offset]));offset+=1
            while pending:
                batch=default_collate([future.result() for future in pending.popleft()])
                if offset<len(groups):
                    pending.append(submit(groups[offset]));offset+=1
                if self.pin_memory:
                    batch={k:v.pin_memory() for k,v in batch.items()}
                yield batch
        finally:
            for group in pending:
                for f in group:f.cancel()
            executor.shutdown(wait=True)
