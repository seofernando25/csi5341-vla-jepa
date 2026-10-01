"""Export the separate n0008 cloud continuation from recorded measurements."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from evaluation.common import ROOT, read_json, write_json, file_hash
from evaluation.figures import style, save

OUTPUT = ROOT / 'studies/confirmation/n0008-20261001/cloud-results'

def validation(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'validation': groups[row['step']].append(row)
    result = []
    for step, group in sorted(groups.items()):
        count = sum(r['samples'] for r in group)
        if count != 200 or len({r['batch'] for r in group}) != len(group):
            raise ValueError('Incomplete or duplicated held-out measurement')
        result.append({'step':step, 'samples':count, **{k:sum(r[k]*r['samples'] for r in group)/count for k in ['loss','action_loss','wm_loss']}})
    return result

def build(training_run, benchmark_run, rollout_run, output):
    training = read_json(training_run/'run.json')
    benchmark = read_json(benchmark_run/'run.json')
    rollout = read_json(rollout_run/'run.json')
    if any(r['status'] != 'completed' for r in [training,benchmark,rollout]):
        raise ValueError('Completed records required')
    if benchmark['checkpoint_sha256'] != rollout['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoints differ')
    rows = [json.loads(l) for l in (training_run/'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in rows if r['phase']=='training']
    if [r['step'] for r in updates] != list(range(5101,10001)):
        raise ValueError('Expected native continuation from 5100 to 10000')
    curve = validation(rows)
    if curve[-1]['step'] != 10000: raise ValueError('Final validation missing')
    timings = list(csv.DictReader((benchmark_run/'timings.csv').open()))
    episodes = list(csv.DictReader((rollout_run/'episodes.csv').open()))
    for mode in ['policy','pipeline']:
        for repetition in range(3):
            selected = [r for r in timings if r['mode']==mode and int(r['repetition'])==repetition]
            if len(selected)!=500 or {int(r['prediction']) for r in selected}!=set(range(500)):
                raise ValueError('Incomplete benchmark repetition')
    if len(episodes)!=10 or {int(r['task_id']) for r in episodes}!=set(range(10)) or any(r['status']!='completed' for r in episodes):
        raise ValueError('Expected ten valid development episodes')
    evidence = {str(p.relative_to(ROOT)):file_hash(p) for directory,files in [(training_run,['run.json','metrics.jsonl']),(benchmark_run,['run.json','timings.csv']),(rollout_run,['run.json','episodes.csv'])] for p in [directory/f for f in files]}
    export = read_json(ROOT/'outputs/cloud/export-status.json')
    job = read_json(ROOT/'outputs/cloud/remote-job.json')
    if export['status']!='exported_and_destroyed' or export['verified_files']!=12:
        raise ValueError('Verified export and rental cleanup required')
    summary = {'candidate':'n0008', 'label':benchmark['variant'], 'scope':'Separate RTX5090 continuation; not final LIBERO or matched hardware replication',
        'resume_step':5000,'final_step':10000,'source_manifest':training['architecture_source_manifest'],
        'recipe':training['recipe'],'environment':benchmark['environment'],
        'checkpoint_sha256':benchmark['checkpoint_sha256'],'validation':curve,
        'benchmark':benchmark['summary'],'development':rollout['summary'],
        'continuation_optimizer_gpu_hours':sum(r['update_seconds'] for r in updates)/3600,
        'peak_training_allocated_bytes':max(r['peak_allocated_bytes'] for r in updates),
        'export':{'verified_files':12,'rental_deleted':True,'checkpoint_files':job['export_files']},
        'evidence':evidence,
        'interpretation':'Prediction loss decreased; 0/10 valid closed-loop development successes. Neither insufficient training nor frozen-backbone mismatch is isolated by this experiment. Matched base confirmation and full development/final evaluations remain pending.'}
    output.mkdir(parents=True,exist_ok=True)
    closure = ROOT/'outputs/cloud/closure.json'
    if closure.exists(): summary['operations'] = read_json(closure)
    write_json(output/'summary.json',summary)
    with (output/'validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
    # Portable copies retain original record hashes while removing host-specific paths.
    def portable(x):
        if isinstance(x,dict):return {k:portable(v) for k,v in x.items()}
        if isinstance(x,list):return [portable(v) for v in x]
        if isinstance(x,str):return x.replace(str(ROOT)+'/', '')
        return x
    for directory in [benchmark_run,rollout_run]:
        original=read_json(directory/'run.json');original.setdefault('original_metadata_sha256',file_hash(directory/'run.json'))
        write_json(directory/'run.json',portable(original))
    render(summary,timings,output)
    print(json.dumps({'final_validation_loss':curve[-1]['loss'],'development_successes':rollout['summary']['successes'],'verified_checkpoint_files':12}))


def render(summary,timings,output):
    curve=summary['validation']
    style()
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    for key,label,color,line in [('loss','Total','#80549A','-'),('action_loss','Action','#264D73','--'),('wm_loss','World model','#007F79',':')]:
        axes[0].plot([r['step']/1000 for r in curve],[r[key] for r in curve],marker='o',ms=3,lw=1.3,ls=line,color=color,label=label)
    axes[0].set(title='a   Held-out prediction loss',xlabel='Optimizer updates (thousands)',ylabel='Mean loss (200 fixed samples)',ylim=(0,.34))
    axes[0].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=8,loc='center right')
    for mode,label,color in [('policy','Policy native','#264D73'),('pipeline','Full pipeline','#80549A')]:
        values=np.sort([float(r['latency_ms']) for r in timings if r['mode']==mode])
        axes[1].plot(values,np.arange(1,len(values)+1)/len(values),color=color,lw=1.5,label=label)
    axes[1].set(title='b   Inference latency',xlabel='Action-chunk latency (ms, log scale)',ylabel='Cumulative fraction',ylim=(0,1.02),xscale='log')
    from matplotlib.ticker import ScalarFormatter
    axes[1].set_xticks([250,500,1000,2000])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True)
    save(fig,output/'figures','F6_cloud')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['training-run','benchmark-run','rollout-run']:p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--from-summary',action='store_true',help='Replot from published compact evidence and timing CSV')
    a=p.parse_args()
    if a.from_summary:
        summary=read_json(a.output/'summary.json')
        timing_path=next(ROOT/name for name in summary['evidence'] if name.endswith('/timings.csv'))
        timings=list(csv.DictReader(timing_path.open()))
        render(summary,timings,a.output);return
    if any(value is None for value in [a.training_run,a.benchmark_run,a.rollout_run]):
        p.error('Provide all three run directories, or --from-summary')
    build(a.training_run.resolve(),a.benchmark_run.resolve(),a.rollout_run.resolve(),a.output.resolve())
if __name__=='__main__':main()
