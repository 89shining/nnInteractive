#!/usr/bin/env python
"""Five-fold nnInteractive Stage-1 CTV lasso fine-tuning."""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import patient_dirs, patient_id, load_case
from interaction_builder import joint_axial_lasso
from prompt_sampler import positive_slices, sample_train
from prediction_wrapper import JointLassoSession, differentiable_predict_joint
from losses import unprompted_dice_ce, unprompted_hard_dice, whole_volume_hard_dice

DATA = Path('/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
SPLITS = Path('/home/intern/ftp/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json')
PLAN = Path('/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json')
MODEL = Path('/home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0')
OUT = Path('/home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults')
TRAINING_CODE_FILES=(
  'coordinate_adapter.py','dataset.py','differentiable_buffer.py','interaction_builder.py',
  'losses.py','prediction_wrapper.py','prompt_sampler.py','train.py',
)

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
      for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def git_commit(repo: Path) -> str | None:
    try: return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): return None

def training_code_manifest(stage1_dir: Path) -> dict[str,str]:
    missing=[name for name in TRAINING_CODE_FILES if not (stage1_dir/name).is_file()]
    if missing: raise FileNotFoundError(f'Missing Stage-1 training files: {missing}')
    return {name:sha256(stage1_dir/name) for name in TRAINING_CODE_FILES}

def verify_audit(audit_path: Path, model_dir: Path) -> None:
    if not audit_path.is_file(): raise FileNotFoundError(f'Missing protocol approval record: {audit_path}')
    audit=json.loads(audit_path.read_text())
    if audit.get('result') != 'PASS': raise RuntimeError(f'Formal training requires audit PASS, got {audit.get("result")}')
    # Runtime Audit is an implementation-feasibility gate. Its audited_fold
    # remains recorded as evidence, but a PASS approves this fixed code/model
    # bundle for the five-fold training protocol rather than only that split.
    fingerprints=audit.get('fingerprints',{}); repo=Path(audit.get('repo',''))
    expected={
      'repository_git_commit':git_commit(repo),
      'inference_session_py_sha256':sha256(repo/'nnInteractive'/'inference'/'inference_session.py'),
      'crop_py_sha256':sha256(repo/'nnInteractive'/'utils'/'crop.py'),
      'model_checkpoint_sha256':sha256(model_dir/'fold_0'/'checkpoint_final.pth'),
      'dataset_json_sha256':sha256(model_dir/'dataset.json'),
      'plans_json_sha256':sha256(model_dir/'plans.json'),
      'inference_session_class_json_sha256':sha256(model_dir/'inference_session_class.json'),
      'training_code_sha256':training_code_manifest(Path(__file__).resolve().parent),
    }
    mismatched={k:(fingerprints.get(k),v) for k,v in expected.items() if fingerprints.get(k)!=v}
    if mismatched: raise RuntimeError(f'Audit fingerprint mismatch; re-audit before training: {mismatched}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=DATA); p.add_argument('--split-path', type=Path, default=SPLITS)
    p.add_argument('--validation-plan', type=Path, default=PLAN); p.add_argument('--model-dir', type=Path, default=MODEL)
    p.add_argument('--output-root', type=Path, default=OUT); p.add_argument('--fold', type=int, default=None)
    p.add_argument('--audit-result', type=Path, default=Path(__file__).resolve().parent/'Audit'/'audit_result.json')
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4); p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--device', default='cuda'); p.add_argument('--no-resume', action='store_true')
    return p.parse_args()

def load_folds(path: Path, root: Path):
    by_name = {p.name:p for p in patient_dirs(root)}; data=json.loads(path.read_text())['folds']; result=[]
    for item in data:
        result.append({'fold':int(item['fold']), 'train':[by_name[x] for x in item['train']], 'val':[by_name[x] for x in item['val']]})
    return result

def checkpoint(path: Path, epoch, model, optim, sched, best, best_epoch, patience, args):
    torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optim.state_dict(),
      'scheduler_state_dict':sched.state_dict(),'best_metric':best,'best_epoch':best_epoch,'patience_counter':patience,
      'config':vars(args)}, path)

def gradient_audit(model):
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    bad_grad=[n for n,p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad_grad: raise RuntimeError(f'Gradient audit failed: non-finite gradients in {bad_grad[:10]}')
    with_grad=[(n,p) for n,p in params if p.grad is not None]
    nonzero=[(n,p) for n,p in with_grad if bool((p.grad != 0).any())]
    if not nonzero: raise RuntimeError('Gradient audit failed: no finite nonzero trainable gradient')
    groups={}
    for n,p in nonzero: groups[n.split('.')[0]]=groups.get(n.split('.')[0],0)+1
    if not any('encoder' in n.lower() for n,_ in nonzero): raise RuntimeError('Gradient audit failed: encoder has no nonzero gradient')
    if not any('decoder' in n.lower() for n,_ in nonzero): raise RuntimeError('Gradient audit failed: decoder has no nonzero gradient')
    print('[nnInteractive gradient audit] PASS', {'trainable':len(params),'with_grad':len(with_grad),'nonzero':len(nonzero),'groups':groups})
    return nonzero[0]

def validate(session, val_cases, plan, fold, device):
    model=session.network; model.eval(); by_k={k:[] for k in range(1,6)}; whole=[]
    with torch.inference_mode():
      for idx, case in enumerate(val_cases,1):
        image,gt=load_case(case); record=plan['folds'][str(fold)][str(patient_id(case))]
        for k in range(1,6):
          scores=[]
          for placement in record['placements'][str(k)]:
            prompts=list(map(int,placement['prompt_frame_ids']))
            session.set_joint_lassos(image,joint_axial_lasso(gt,prompts))
            pred=session.native_predict_joint().to(device)
            scores.append(unprompted_hard_dice(pred,torch.from_numpy(gt).to(device),prompts))
            whole.append(whole_volume_hard_dice(pred,torch.from_numpy(gt).to(device)))
          by_k[k].append(float(np.mean(scores)))
        print(f'[val fold{fold} patient {idx}/{len(val_cases)}] {case.name}')
    means={f'D{k}':float(np.mean(by_k[k])) for k in by_k}; means['S_val']=float(np.mean(list(means.values())))
    means['whole_volume_dice']=float(np.mean(whole))
    return means

def run_fold(args, spec, plan, device):
    fold=spec['fold']; run=args.output_root/'Mixed_K1_5'/f'fold_{fold}'; ckpt=run/'checkpoints'; ckpt.mkdir(parents=True,exist_ok=True)
    if (run/'completed.flag').exists(): print(f'fold {fold}: completed; skip'); return
    session=JointLassoSession(device=device,do_autozoom=True,verbose=False,use_pinned_memory=True)
    session.initialize_from_trained_model_folder(str(args.model_dir),use_fold=0)
    model=session.network
    for parameter in model.parameters(): parameter.requires_grad_(True)
    total_params=sum(p.numel() for p in model.parameters()); trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params != total_params: raise RuntimeError('Full-network fine-tuning requires every parameter to be trainable')
    print(f'[full-network FT] total={total_params:,} trainable={trainable_params:,} frozen={total_params-trainable_params:,}')
    model.train(); optim=AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); sched=CosineAnnealingLR(optim,T_max=args.epochs)
    start,best,best_epoch,patience=1,-1.,-1,0; latest=ckpt/'latest.pth'
    if latest.exists() and not args.no_resume:
      state=torch.load(latest,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict']); optim.load_state_dict(state['optimizer_state_dict']); sched.load_state_dict(state['scheduler_state_dict']); start=state['epoch']+1; best=state['best_metric']; best_epoch=state['best_epoch']; patience=state['patience_counter']
    if patience >= 2:
      (run/'completed.flag').write_text('completed\n'); print(f'fold {fold}: resumed patience exhausted; mark completed'); return
    for epoch in range(start,args.epochs+1):
      model.train(); losses=[]; begun=time.time(); first=True
      for case in spec['train']:
        image,gt=load_case(case); rng=random.Random(args.seed+fold*10000019+epoch*1000003+patient_id(case)); prompts=sample_train(positive_slices(gt),rng,2)
        session.set_joint_lassos(image,joint_axial_lasso(gt,prompts)); optim.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
          logits,coverage,_=differentiable_predict_joint(session, complete_coverage=True, activation_checkpointing=True, prompt_slices=prompts)
          loss_mask=torch.ones(gt.shape,dtype=torch.bool,device=device); loss_mask[prompts]=False
          if not bool(coverage[loss_mask].all()): raise RuntimeError(f'{case.name}: native path did not cover all unprompted loss voxels')
          loss=unprompted_dice_ce(logits,torch.from_numpy(gt).to(device),prompts)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward()
        if first:
          audit_name,audit_param=gradient_audit(model)
          before=audit_param.detach().clone()
        optim.step()
        if first and not bool((audit_param.detach()-before).abs().max()>0): raise RuntimeError(f'Optimizer update audit failed for {audit_name}')
        first=False; losses.append(float(loss.detach()))
      sched.step(); metrics=None
      if epoch==1 or epoch%10==0:
        metrics=validate(session,spec['val'],plan,fold,device); value=metrics['S_val']; improved=value>best
        if improved: best,best_epoch,patience=value,epoch,0; checkpoint(ckpt/'best.pth',epoch,model,optim,sched,best,best_epoch,patience,args)
        else: patience+=1
      checkpoint(latest,epoch,model,optim,sched,best,best_epoch,patience,args)
      print(f'[fold {fold}] epoch {epoch}/{args.epochs} loss={np.mean(losses):.5f} val={metrics or "SKIP"} best={best:.5f}@{best_epoch} time={time.time()-begun:.1f}s')
      if patience>=2: break
    (run/'completed.flag').write_text('completed\n')

def main():
    args=parse_args(); device=torch.device(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    plan=json.loads(args.validation_plan.read_text()); folds=load_folds(args.split_path,args.data_root)
    required=[args.model_dir/'dataset.json',args.model_dir/'plans.json',args.model_dir/'inference_session_class.json',args.model_dir/'fold_0'/'checkpoint_final.pth']
    if not all(x.is_file() for x in required): raise FileNotFoundError('Incomplete official nnInteractive_v1.0 bundle')
    verify_audit(args.audit_result,args.model_dir)
    print({'data_root':str(args.data_root),'model_dir':str(args.model_dir),'output_root':str(args.output_root),'K':'1-5','gap':'2 slices (10 mm)','interaction':'joint positive axial closed lasso','decay':1.0,'loss':'0.5 soft Dice + 0.5 CE, unprompted only','torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})
    for spec in folds:
      if args.fold is None or spec['fold']==args.fold: run_fold(args,spec,plan,device)
if __name__=='__main__': main()
