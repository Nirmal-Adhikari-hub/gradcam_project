# ───────── gradcam_perlayer_slowfast.py  (replace the old file) ──────────
import sys, glob, cv2, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ─── make pytorch-grad-cam resize robust ─────────────────────────────────
import pytorch_grad_cam.utils.image as _im
import pytorch_grad_cam.base_cam    as _bc
_orig = _im.scale_cam_image
def _safe(img, tgt=None):
    if img.ndim == 3: img = img.mean(0)
    if img.ndim == 1: img = img[None]
    if isinstance(tgt,(list,tuple)) and len(tgt)>=2:
        tgt = (int(tgt[-1]), int(tgt[-2]))      # (W,H)
    return _orig(img, tgt)
_im.scale_cam_image = _safe
_bc.scale_cam_image = _safe
# ─────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).resolve().parent))
from slowfast_setup import parse_args, load_dataset, load_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MEAN = np.array([0.485,0.456,0.406])[None,None,:]
STD  = np.array([0.229,0.224,0.225])[None,None,:]
denorm = lambda x:(x*STD+MEAN).clip(0,1)

def reshape(t):
    if isinstance(t,tuple): t=t[0]
    if isinstance(t,PackedSequence):t=t.data
    if t.dim()==5:
        B,C,T,H,W=t.shape
        return t.permute(0,2,1,3,4).reshape(B*T,C,H,W)
    if t.dim()==3:
        B,C,T=t.shape
        return t.permute(0,2,1).reshape(B*T,C,1,1)
    if t.dim()==2:
        B,C=t.shape
        return t.view(B*C,1,1)
    return t.flatten().view(-1,1,1)

def load_frames(info,args):
    rel = info.split('|')[1]
    root=Path(args.dataset_info['dataset_root'])
    patt=root/'features'/f"fullFrame-256x256px/{args.mode}"/rel
    paths=sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:,:,::-1].astype(np.float32)/255 for p in paths])

# ─────────────────────────────────────────────────────────────────────────
def main():
    args=parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gd=np.load(args.dataset_info['dict_path'],allow_pickle=True).item()
    feeder=load_dataset(args=args,gloss_dict=gd)
    model=load_model(args=args,gloss_dict=gd,checkpoint_path=args.slowfast_ckpt).to(dev)

    data,len_x,_,_,info=feeder.collate_fn([feeder[args.index]])
    frames=load_frames(info[0],args); inp=data.to(dev)

    # ── make first BiLSTM CAM-friendly ────────────────────────────────────
    class ROut(nn.Module):
        def __init__(s,r): super().__init__(); s.r=r
        def forward(s,x,h=None):
            p,_=s.r(x,h); pad,_=rnn_utils.pad_packed_sequence(p,batch_first=False)
            return pad
    model.temporal_model[0].rnn=ROut(model.temporal_model[0].rnn)

    import types
    def fwd(self,src,lens,h=None):
        p=rnn_utils.pack_padded_sequence(src,lens); o=self.rnn(p,h)
        if isinstance(o,torch.Tensor): out,hid=o,None
        elif isinstance(o,rnn_utils.PackedSequence): out,_=rnn_utils.pad_packed_sequence(o); hid=None
        else:
            po,hid=o; out,_=rnn_utils.pad_packed_sequence(po)
        if self.bidirectional and hid is not None:
            hid=self._cat_directions(hid)
        if isinstance(hid,tuple): hid=torch.cat(hid,0)
        return {'predictions':out,'hidden':hid}
    model.temporal_model[0].forward = types.MethodType(fwd,model.temporal_model[0])

    # ── wrapper that returns *last-step* logits (better grad signal) ──────
    class CAMWrap(nn.Module):
        def __init__(s,core,L): super().__init__(); s.c=core; s.L=L
        def forward(s,x):
            l=s.L.repeat(x.size(0)); train=s.c.training; s.c.train()
            with torch.set_grad_enabled(True):
                logits=s.c(x,l)['sequence_logits'][0]     # [T,B,C] / [B,C]
                if logits.dim()==3: logits=logits[-1]     # take final timestep
            s.c.train(train); return logits
    cam_model=CAMWrap(model,len_x.to(dev)).to(dev)

    # ── target layers ─────────────────────────────────────────────────────
    layers=[
        ('stem_fast', model.conv2d.s1.pathway1_stem),
        ('res3_slow', model.conv2d.s3.pathway0_res1.branch2.c),
        ('res5_fast', model.conv2d.s5.pathway1_res2.branch2.c),
        ('bilstm0',   model.temporal_model[0].rnn),
    ]

    with torch.no_grad():
        top_id=int(cam_model(inp)[0].argmax())
    print(f'\nTop-1 predicted gloss: {top_id} ({gd.get(top_id,"<unk>")})\n')

    out_root=Path('slowfast/work_dir/gradcam_per_layer')
    H_vid,W_vid=frames.shape[1:3]
    alpha=0.4                                  # overlay strength

    for lname,layer in layers:
        print(f'→ Running Grad-CAM on [{lname}]')
        cam=GradCAMPlusPlus(cam_model,[layer],reshape_transform=reshape)
        heat=cam(inp,targets=[ClassifierOutputTarget(top_id)],
                 eigen_smooth=True,aug_smooth=False)    # (N,Hm,Wm) or (N,P)

        if heat.ndim==2: heat=heat[:,None]              # (N,P)→(N,1,P)
        N=heat.shape[0]; frame_ids=np.linspace(0,len(frames)-1,N).round().astype(int)
        save_dir=out_root/lname; save_dir.mkdir(parents=True,exist_ok=True)

        for i,(hm,fi) in enumerate(zip(heat,frame_ids)):
            hm=hm.squeeze()
            if hm.ndim==1: hm=hm[:,None]
            hm=cv2.resize(hm.astype(np.float32),(W_vid,H_vid))
            hm=(hm-hm.min())/(hm.max()-hm.min()+1e-7)    # re-normalise
            vis=show_cam_on_image(denorm(frames[fi]),hm,use_rgb=True,image_weight=1-alpha)
            cv2.putText(vis, f'{gd.get(top_id,"<unk>")}', (8,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(save_dir/f'frame{i:03d}.jpg'), vis)

        print(f'   saved {N} maps → {save_dir}')
    print('\nDone!  See slowfast/work_dir/gradcam_per_layer/')
# -------------------------------------------------------------------------
if __name__=='__main__': main()
