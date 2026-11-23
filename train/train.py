"""Pretrain a Sentinel-2 teacher using DINO, iBOT and KoLeo."""
import argparse
import json
import math
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import RandomResizedCrop,RandomHorizontalFlip,Compose
from osgeo import gdal
from mangrove import build_dpt_encoder
from mangrove.layers import DINOHead
from mangrove.loss import DINOLoss,iBOTPatchLoss,KoLeoLoss
from mangrove.masking import MaskingGenerator

class SSLDataset(Dataset):
    def __init__(self,args):
        self.paths = sorted(p for p in Path(args.dataset).glob("*") if p.suffix.lower() in (".tif", ".tiff"))
        if not self.paths: raise ValueError("No Sentinel-2 tiles in --dataset")
        self.global_transform=Compose([RandomResizedCrop(args.global_size,scale=(.32,1.0)),RandomHorizontalFlip()])
        self.local_transform=Compose([RandomResizedCrop(args.local_size,scale=(.05,.32)),RandomHorizontalFlip()])
        self.local_crops=args.local_crops
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        image=gdal.Open(str(self.paths[i])).ReadAsArray().astype(np.float32)
        if image.shape != (10,512,512): raise ValueError("Expected 10 bands at 512 x 512")
        image=np.nan_to_num(image)
        image=torch.from_numpy(image)
        return [self.global_transform(image) for _ in range(2)]+[self.local_transform(image) for _ in range(self.local_crops)]

def build_network(args):
    backbone,_=build_dpt_encoder(args.arch,10,16,0)
    backbone.gradient_checkpointing=True
    return nn.ModuleDict({"backbone":backbone,"dino_head":DINOHead(backbone.embed_dim,args.prototypes)})
def seed_worker(worker):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)

def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device);amp=args.amp and device.type=="cuda"
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if not args.resume and (output/"last.pth").exists():
        raise FileExistsError("Resume or choose a new output directory")
    loader=DataLoader(SSLDataset(args),batch_size=args.batch_size,shuffle=True,drop_last=True,
                      num_workers=args.workers,worker_init_fn=seed_worker)
    student=build_network(args).to(device)
    teacher=build_network(args).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False);teacher.eval()
    dino=DINOLoss(args.prototypes).to(device)
    ibot=iBOTPatchLoss(args.prototypes).to(device)
    koleo=KoLeoLoss().to(device)
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.04)
    scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128)
    start=0;step=0
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu")
        if saved.get("s2_channels") != 10:
            raise ValueError("Resume requires a 10-band pretraining checkpoint from this version")
        for name,model in [("student",student),("teacher",teacher),("dino",dino),("ibot",ibot)]:
            model.load_state_dict(saved[name],strict=True)
        optimizer.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
        start=saved["epoch"]+1;step=saved["step"]
        torch.set_rng_state(saved["torch_rng"]);np.random.set_state(saved["numpy_rng"]);random.setstate(saved["python_rng"])
        if device.type=="cuda":torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    grid=args.global_size//16;tokens=grid*grid
    mask_generator=MaskingGenerator((grid,grid),min_num_patches=min(4,tokens),
                                    max_num_patches=max(1,tokens//2))
    total_steps=max(1,len(loader)*args.epochs)
    for epoch in range(start,args.epochs):
        student.train();loss_sum=0.;updates=0
        for batches in loader:
            crops=[x.to(device) for x in batches]
            masks=[]
            for crop in crops[:2]:
                batch_masks=[]
                for _ in range(len(crop)):
                    mask=mask_generator(max(1,int(tokens*random.uniform(.1,.5))))

                    if not mask.any():mask.flat[0]=True
                    batch_masks.append(torch.from_numpy(mask.flatten()))
                masks.append(torch.stack(batch_masks).to(device))
            progress=min(step/total_steps,1.)
            lr=args.lr*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group["lr"]=lr
            temp=.04+min(epoch/30.,1.)*.03
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                with torch.no_grad():
                    teacher_features=[teacher["backbone"].forward_features(x) for x in crops[:2]]
                    teacher_cls=teacher["dino_head"](torch.cat([x["x_norm_clstoken"] for x in teacher_features]))
                    teacher_patch=teacher["dino_head"](torch.cat([x["x_norm_patchtokens"][m] for x,m in zip(teacher_features,masks)]))
                    probs=dino.softmax_center_teacher(teacher_cls.float(),temp).chunk(2)
                    patch_probs=ibot.softmax_center_teacher(teacher_patch.float()[None],temp)[0]
                    dino.update_center(teacher_cls.float());ibot.update_center(teacher_patch.float()[None])
                student_features=[student["backbone"].forward_features(x,masks=m) for x,m in zip(crops[:2],masks)]
                local_features=[student["backbone"].forward_features(x) for x in crops[2:]]
                cls_outputs=[student["dino_head"](x["x_norm_clstoken"]) for x in student_features+local_features]
                dino_terms=[dino([logits.float()],[target]) for i,logits in enumerate(cls_outputs)
                            for j,target in enumerate(probs) if i!=j]
                loss_dino=sum(dino_terms)/len(dino_terms)
                student_patch=student["dino_head"](torch.cat([x["x_norm_patchtokens"][m] for x,m in zip(student_features,masks)]))
                loss_ibot=ibot.forward_masked(student_patch.float(),patch_probs,torch.cat(masks))
                loss_koleo=sum(koleo(x["x_norm_clstoken"].float()) for x in student_features)*.1 if len(crops[0])>1 else loss_dino*0
                loss=loss_dino+loss_ibot+loss_koleo
            if not torch.isfinite(loss):raise FloatingPointError("SSL loss NaN/Inf")
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),3.0)
            if not torch.isfinite(norm):
                if not amp:raise FloatingPointError("SSL gradient NaN/Inf")
                scaler.update();continue
            scaler.step(optimizer);scaler.update()
            momentum=1-(1-.996)*(.5*(1+math.cos(math.pi*progress)))
            with torch.no_grad():
                for target,value in zip(teacher.parameters(),student.parameters()):
                    target.mul_(momentum).add_(value,alpha=1-momentum)
            step+=1;updates+=1;loss_sum+=loss.item()
            if step%10==0 or args.smoke_test:
                print(json.dumps(dict(epoch=epoch+1,step=step,loss=loss.item(),
                                      dino=loss_dino.item(),ibot=loss_ibot.item(),koleo=loss_koleo.item())),flush=True)
            if args.smoke_test:break
        if not updates:raise RuntimeError("No successful SSL updates")
        dino.apply_center_update();ibot.apply_center_update()
        torch.save({"teacher":teacher.state_dict(), "s2_channels":10},output/"teacher_checkpoint.pth")
        if not args.smoke_test:
            state=dict(s2_channels=10,epoch=epoch,step=step,student=student.state_dict(),teacher=teacher.state_dict(),
                       dino=dino.state_dict(),ibot=ibot.state_dict(),optimizer=optimizer.state_dict(),
                       scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),
                       numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None)
            torch.save(state,output/"last.tmp");(output/"last.tmp").replace(output/"last.pth")
        with (output/"metrics.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(epoch=epoch+1,loss=loss_sum/updates,updates=updates,smoke_only=args.smoke_test))+"\n")
        if args.smoke_test:return
def get_args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True)
    p.add_argument("--output-dir",default=str(ROOT/"outputs/ssl"))
    p.add_argument("--arch",choices=["small","base","large"],default="large")
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--global-size",type=int,default=512);p.add_argument("--local-size",type=int,default=192)
    p.add_argument("--local-crops",type=int,default=8);p.add_argument("--prototypes",type=int,default=65536)
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--workers",type=int,default=0)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda");p.add_argument("--no-amp",dest="amp",action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--resume");p.add_argument("--smoke-test",action="store_true")
    return p
def main(args=None):train(get_args_parser().parse_args() if args is None else args)
if __name__=="__main__":main()
