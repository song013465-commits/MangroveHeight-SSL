import copy,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np

def cpu_snapshot(value):
    
    if isinstance(value,torch.Tensor):
        result=value.detach().cpu()
        return result.clone() if value.device.type=="cpu" else result
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):
        result=copy.copy(value)
        for key,item in value.items():result[key]=cpu_snapshot(item)
        if hasattr(value,"_metadata"):result._metadata=copy.deepcopy(value._metadata)
        return result
    if isinstance(value,list):return [cpu_snapshot(x) for x in value]
    if isinstance(value,tuple):return tuple(cpu_snapshot(x) for x in value)
    return copy.deepcopy(value)

class AsyncCheckpointer:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.future=None
    def save(self,checkpoint,improved):

        if self.future is not None:self.future.result()
        snapshot=cpu_snapshot(checkpoint)
        self.future=self.executor.submit(self._write,snapshot,improved)
    def _write(self,state,improved):
        started=time.perf_counter()
        temporary=self.folder/"last.tmp"
        torch.save(state,temporary);temporary.replace(self.folder/"last.pth")
        if improved:

            best={k:v for k,v in state.items() if k not in ("optimizer","scaler","rng")}
            temporary=self.folder/"best.tmp"
            torch.save(best,temporary);temporary.replace(self.folder/"best.pth")
        status={"epoch":state["epoch"]+1,"saved":True,"improved":improved,
                "write_seconds":time.perf_counter()-started}
        (self.folder/"checkpoint_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    def close(self):
        try:
            if self.future is not None:self.future.result()
        finally:self.executor.shutdown(wait=True)
