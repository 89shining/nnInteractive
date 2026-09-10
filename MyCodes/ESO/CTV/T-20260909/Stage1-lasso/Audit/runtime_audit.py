#!/usr/bin/env python
"""Runtime Go/No-Go audit; does not optimize or write model weights."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from dataset import patient_dirs, patient_id, load_case
from interaction_builder import joint_axial_lasso, connected_components_4
from prediction_wrapper import JointLassoSession, differentiable_predict_joint
from nnInteractive.inference import inference_session as native_session_module

DATA=Path('/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train')
PLAN=Path('/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json')
MODEL=Path('/home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0')

class TraceSession(JointLassoSession):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.native_trace=[]; self._capture_refinement=False
    def _build_network_input(self, center, zoom):
        out=super()._build_network_input(center,zoom)
        bbox=[[int(a),int(b)] for a,b in out[2]]
        self.native_trace.append({'type':'coarse','center':[int(v) for v in center],'zoom':float(zoom),'bbox':bbox}); return out
    def _refine_coarse(self, diff_map, prediction_with_coarse):
        # Call the unmodified official refinement implementation. The module
        # hook below observes only its executed crop bboxes.
        self._capture_refinement=True
        try: return super()._refine_coarse(diff_map,prediction_with_coarse)
        finally: self._capture_refinement=False

class NativeRefinementTrace:
    """Temporarily observe official refinement bboxes without changing its logic."""
    def __init__(self, session):
        self.session=session; self.original=None
        self.active_bbox=None; self.remaining_calls=0
    def __enter__(self):
        self.original=native_session_module.crop_and_pad_into_buffer
        def hooked(target, bbox, source, *args, **kwargs):
            if self.session._capture_refinement:
                frozen=[[int(a),int(b)] for a,b in bbox]
                # Official refinement calls this helper exactly three times
                # per refinement bbox (image, prediction, interactions).
                # Count calls rather than globally de-duplicating bboxes, so
                # two consecutive official refinement steps at the same bbox
                # remain distinguishable in the trace.
                if self.remaining_calls == 0 or frozen != self.active_bbox:
                    self.session.native_trace.append({'type':'refine','bbox':frozen})
                    self.active_bbox=frozen
                    self.remaining_calls=2
                else:
                    self.remaining_calls-=1
            return self.original(target,bbox,source,*args,**kwargs)
        native_session_module.crop_and_pad_into_buffer=hooked
        return self
    def __exit__(self, exc_type, exc, tb):
        native_session_module.crop_and_pad_into_buffer=self.original

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=DATA); p.add_argument('--plan',type=Path,default=PLAN); p.add_argument('--model-dir',type=Path,default=MODEL); p.add_argument('--fold',type=int,default=0); p.add_argument('--max-patients',type=int,default=0); args=p.parse_args()
 plan=json.loads(args.plan.read_text()); cases=patient_dirs(args.data_root); records=plan['folds'][str(args.fold)]
 eligible=[p for p in cases if str(patient_id(p)) in records]
 selected=list(eligible)
 if args.max_patients: selected=selected[:args.max_patients]
 expected_episodes=len(eligible)*5*2
 s=TraceSession(device=torch.device('cuda'),do_autozoom=True,verbose=False); s.initialize_from_trained_model_folder(str(args.model_dir),use_fold=0); s.network.eval()
 failures=[]; multi=[]; checked=0
 # This observation is data-wide by design. It documents every axial mask
 # that expands to multiple component-wise lassos, independently of the
 # single fold used for the GPU trajectory/coverage gate.
 for component_case in cases:
  _,component_gt=load_case(component_case)
  for z in range(component_gt.shape[0]):
   if component_gt[z].any() and connected_components_4(component_gt[z])!=1:
    multi.append(f'{component_case.name}:z={z}')
 for case in selected:
  image,gt=load_case(case)
  for k in range(1,6):
   for place in records[str(patient_id(case))]['placements'][str(k)]:
    prompts=list(map(int,place['prompt_frame_ids'])); lasso=joint_axial_lasso(gt,prompts)
    with torch.inference_mode():
      s.set_joint_lassos(image,lasso); s.native_trace=[]
      with NativeRefinementTrace(s): s.native_predict_joint()
      native=list(s.native_trace)
    s.set_joint_lassos(image,lasso)
    # Trace is collected only from the native interaction-driven phase. The
    # deterministic completion grid then fills missing logits without changing
    # session.interactions, queues, AutoZoom, or the trace.
    # Match the actual training wrapper's numerical/control environment.
    # Native validation remains untouched and manages its own native path.
    with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_available()):
      _,coverage,wrapper=differentiable_predict_joint(s, complete_coverage=True, prompt_slices=prompts)
    mask=torch.ones(gt.shape,dtype=torch.bool,device=coverage.device); mask[prompts]=False
    if not bool(coverage[mask].all()): failures.append(f'coverage {case.name} K{k} placement{place["placement_id"]}')
    if native!=wrapper: failures.append(f'trajectory {case.name} K{k} placement{place["placement_id"]}: native={native}, wrapper={wrapper}')
    checked+=1; print(f'checked {checked}: {case.name} K{k} p{place["placement_id"]}',flush=True)
 # Multi-component axial CTV masks are valid: every component becomes one
 # closed contour in the same prompted-slice interaction. Keep them as an
 # audit observation, never as a feasibility failure.
 complete=(args.max_patients == 0 and len(eligible) > 0 and expected_episodes > 0 and checked == expected_episodes)
 result='PASS' if complete and not failures else 'BLOCKED'
 report={
   'result':result,
   'eligible_patients':len(eligible),
   'selected_patients':len(selected),
   'audited_fold':args.fold,
   'expected_episodes':expected_episodes,
   'checked_episodes':checked,
   'complete':complete,
   'coverage_or_trajectory_failures':failures,
   'multi_component_slice_observations':multi,
 }
 out=Path(__file__).resolve().parent/'runtime_audit_report.json'; out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
