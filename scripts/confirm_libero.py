"""Restartable n0008 confirmation: train, select held-out checkpoint, evaluate LIBERO."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUNTIME = ROOT/'outputs/confirmation/n0008-20261001'
EVIDENCE = ROOT/'studies/confirmation/n0008-20261001'
PYTHON = ROOT/'.venv/bin/python'
ARMS = [('RSI-n0008', '.rsi/attempts/n0008/workspace'), ('SmolVLM-GPU-base', '.rsi/base')]


def read(path): return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_hashes(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*.py'))}


def plan():
    if (EVIDENCE/'plan.json').exists(): return read(EVIDENCE/'plan.json')
    from rsi.runner import Runner
    from rsi.core import manifest, digest
    runner=Runner(ROOT,ROOT/'.rsi'); runner.verify()
    event=next(e for e in runner.events('outcome') if e['attempt']=='n0008')
    if manifest(ROOT/ARMS[0][1]) != event['workspace_hash']:
        raise RuntimeError('n0008 architecture changed')
    arms=[]
    for label,relative in ARMS:
        destination=RUNTIME/label/'source'
        shutil.copytree(ROOT/relative/'src',destination/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        arms.append({'label':label,'source':str(destination.relative_to(ROOT)),
                     'source_manifest':source_hashes(destination/'src')})
    result={'frozen_at':dt.datetime.now(dt.timezone.utc).isoformat(),'candidate':'n0008',
            'study_id':digest(runner.events('initialized')[0]),'arms':arms,
            'optimizer_steps':10000,'checkpoints':[1500,5000,10000],
            'recipe':read(ROOT/'evaluation/training_config.json'),
            'image_processor_backend':'torchvision',
            'selection':'Lowest sample-weighted held-out loss on 200 fixed samples at 1500, 5000, 10000; ties choose fewer steps.',
            'development_episodes':100,'final_episodes':500,
            'final_evaluation':'After development review; initial campaign does not claim final task performance.',
            'benchmark':{'predictions':500,'repetitions':3},
            'limits':'Initial 10k budget; substantially less training exposure than the paper. Qwen reference is evaluation-only. Development success does not select checkpoints.',
            'retention':'Retain milestone inference weights and latest optimizer state; export hashes before deleting other checkpoints.'}
    write(EVIDENCE/'plan.json',result)
    return result


def run_child(argv, label, stage, source):
    logs=RUNTIME/label/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/f'{stage}-{time.time_ns()}.log'
    env=dict(os.environ,PYTHONPATH=str(source/'src')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl')
    status={'status':'running','arm':label,'stage':stage,'log':str(log.relative_to(ROOT)),
            'started_at':dt.datetime.now(dt.timezone.utc).isoformat()}
    write(RUNTIME/'status.json',status)
    print(label,stage,flush=True)
    with log.open('w') as handle:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=handle,stderr=handle)
        while process.poll() is None:
            if time.time()-log.stat().st_mtime>2400:
                process.terminate();process.wait(timeout=40)
                raise RuntimeError('No log progress for 40 minutes')
            if shutil.disk_usage(ROOT).free < 50*1024**3:
                cleanup(label)
            if shutil.disk_usage(ROOT).free < 30*1024**3:
                process.terminate();process.wait(timeout=40)
                raise RuntimeError('Disk guard: less than 30 GiB free')
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{stage} exit {process.returncode}; inspect {log.relative_to(ROOT)}')
    return log


def training_records(label):
    return sorted([p for p in (ROOT/'studies/evaluation/training').glob('*/run.json')
                   if read(p).get('variant')==label])


def checkpoint_candidates(label):
    candidates=[]
    for path in training_records(label):
        run_id=read(path)['run_id']
        for p in (ROOT/'outputs/evaluation/training'/run_id/'train/checkpoints').glob('*/pretrained_model'):
            state=p.parent/'training_state'
            required=['optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors','scheduler_state.json','training_step.json']
            if p.parent.name.isdigit() and (p/'model.safetensors').exists() and all((state/name).is_file() for name in required):
                try:
                    from safetensors import safe_open
                    for file in [p/'model.safetensors',state/'optimizer_state.safetensors',state/'rng_state.safetensors']:
                        with safe_open(file,framework='pt',device='cpu') as handle: handle.keys()
                    if int(read(state/'training_step.json')['step'])!=int(p.parent.name):continue
                except Exception:continue
                candidates.append((int(p.parent.name),p))
    return sorted(candidates,key=lambda x:x[0])


def select(label):
    from evaluation.select_checkpoint import choose
    from evaluation.common import file_hash
    batches={};expected=plan()['recipe']
    for record_path in training_records(label):
        record=read(record_path)
        if record.get('recipe')!=expected:raise RuntimeError('Training recipe differs')
        metrics=record_path.parent/'metrics.jsonl'
        if not metrics.exists():continue
        for line in metrics.read_text().splitlines():
            row=json.loads(line)
            if row['phase']=='validation':batches[row['step'],row['batch']]=row
    curve=[]
    for step in sorted({k[0] for k in batches}):
        rows=[r for (s,b),r in batches.items() if s==step];count=sum(r['samples'] for r in rows)
        if count==200:
            curve.append({'step':step,'samples':count,**{key:sum(r[key]*r['samples'] for r in rows)/count for key in ['loss','action_loss','wm_loss']}})
    winner=choose(curve,[1500,5000,10000]);checkpoints=[]
    for path in training_records(label):
        run_id=read(path)['run_id']
        for weights in (ROOT/'outputs/evaluation/training'/run_id/'train/checkpoints').glob('*/pretrained_model/model.safetensors'):
            if not weights.parents[1].name.isdigit() or int(weights.parents[1].name)!=winner['step']:continue
            checkpoints.append({'step':winner['step'],'sha256':file_hash(weights),'bytes':weights.stat().st_size,'artifact':str(weights.parent.relative_to(ROOT))})
    if not checkpoints or len({c['sha256'] for c in checkpoints})!=1:raise RuntimeError('Selected checkpoint missing or ambiguous')
    result={'rule':plan()['selection'],'selected_step':winner['step'],'validation':winner,
            'checkpoint':checkpoints[-1],'validation_history':curve,'simulator_scores_used':False}
    write(EVIDENCE/f'{label}-selection.json',result)
    return ROOT/result['checkpoint']['artifact']


def cleanup(label):
    from evaluation.common import file_hash
    candidates=checkpoint_candidates(label)
    if not candidates:return
    latest=candidates[-1][0];retained={1500,5000,10000,latest}
    removed=[]
    for step,path in candidates:
        targets=[path.parent] if step not in retained else ([path.parent/'training_state'] if step!=latest else [])
        for target in targets:
            if target.is_dir() and not target.is_symlink():
                removed.append({'path':str(target.relative_to(ROOT)),'step':step,'weights_sha256':file_hash(path/'model.safetensors')})
    if removed:
        write(EVIDENCE/'retention'/f'{label}-{time.time_ns()}.json',{'removed':removed,'keep_steps':sorted(retained)})
        for row in removed:shutil.rmtree(ROOT/row['path'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();campaign=plan()
    if args.prepare_only:print(json.dumps(campaign,indent=2));return
    for arm in campaign['arms']:
        label=arm['label'];source=ROOT/arm['source']
        if source_hashes(source/'src')!=arm['source_manifest']:raise RuntimeError('Frozen source changed')
        for stop in campaign['checkpoints']:
            records=training_records(label)
            if any(read(p).get('status')=='completed' and read(p).get('completed_steps',0)>=stop for p in records):continue
            checkpoints=checkpoint_candidates(label)
            argv=[str(PYTHON),'-m','evaluation.train','--dataset-root',str(args.dataset_root.resolve()),
                  '--steps',str(stop),'--run-label',label,'--architecture-source',str(source)]
            if checkpoints:argv+=['--resume',str(checkpoints[-1][1])]
            run_child(argv,label,f'train-{stop}',source)
            cleanup(label)
        checkpoint=select(label)
        for command,extra in [('rollout',['--phase','development','--episodes','10']),
                              ('benchmark',['--predictions','500','--repetitions','3'])]:
            marker=EVIDENCE/f'{label}-{command}.json'
            if marker.exists():continue
            before=set((ROOT/'studies/evaluation/runs').glob('*/run.json'))
            argv=[str(PYTHON),'-m','evaluation.run',command,'--variant','S500','--label',label,'--checkpoint',str(checkpoint),*extra]
            run_child(argv,label,command,source)
            records=[p for p in set((ROOT/'studies/evaluation/runs').glob('*/run.json'))-before if read(p).get('variant')==label and read(p).get('status')=='completed']
            if len(records)!=1:raise RuntimeError('Evaluation evidence ambiguous')
            write(marker,{'run':str(records[0].relative_to(ROOT)),'summary':read(records[0])['summary']})
    write(RUNTIME/'status.json',{'status':'completed','completed_at':dt.datetime.now(dt.timezone.utc).isoformat()})
    print('Confirmation training, development LIBERO and timing completed.',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        prior=read(RUNTIME/'status.json') if (RUNTIME/'status.json').exists() else {}
        write(RUNTIME/'status.json',{**prior,'status':'failed','error':str(exc),'failed_at':dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
