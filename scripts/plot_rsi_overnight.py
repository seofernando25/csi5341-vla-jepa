#!/usr/bin/env python3
"""Export measured overnight RSI figures; replay from the compact JSON snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, GRAY, ORANGE = '#1764b5', '#64748b', '#b35c19'

def extract(journal):
    rows, failures = [], []
    for path in sorted(journal.glob('*.json')):
        raw = path.read_bytes()
        event = json.loads(raw)
        kind = event.get('kind')
        if kind not in ('baseline', 'promotion_baseline', 'outcome', 'promotion'):
            continue
        node = event.get('node', {})
        metrics = node.get('metrics', {}) if kind == 'outcome' else event.get('metrics', {})
        if 'eval_loss' not in metrics:
            if kind == 'outcome':
                failures.append({'id': event['attempt'], 'status': node.get('status'), 'event_seq': event['seq']})
            continue
        rows.append(dict(id=event.get('attempt', 'baseline'), protocol='screen' if kind in ('baseline', 'outcome') else 'promotion',
                         loss=metrics['eval_loss'], steps=metrics['optimizer_steps'],
                         peak_gib=metrics['hardware']['peak_allocated_bytes'] / 2**30,
                         wall_minutes=metrics['wall_seconds'] / 60,
                         eval_samples=metrics.get('eval_samples'), gpu=metrics['hardware']['gpu'],
                         batch=event.get('batch_id'), ended_at=event.get('ended_at'),
                         event_seq=event['seq'], event_sha256=hashlib.sha256(raw).hexdigest(),
                         workspace_hash=event.get('workspace_hash')))
    return {'description': 'Current GPU-processor study, overnight September 30–October 1, 2026. Single seed; logged loss precision 4 decimals. Wall time includes setup, training and validation; not inference latency.', 'rows': rows, 'unmeasured_attempts': failures}

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#cbd5e1')
    ax.tick_params(color='#cbd5e1', labelcolor='#334155')
    ax.grid(axis='y', color='#e2e8f0', linewidth=.7)
    ax.set_axisbelow(True)

def save(fig, out, name, footer):
    fig.text(.02, .015, footer, fontsize=8, color=GRAY)
    fig.savefig(out / (name + '.png'), dpi=220, facecolor='white')
    fig.savefig(out / (name + '.pdf'), facecolor='white')
    plt.close(fig)

def plot(data, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    screens = [r for r in data['rows'] if r['protocol'] == 'screen' and r['id'] != 'baseline']
    base = next(r for r in data['rows'] if r['protocol'] == 'screen' and r['id'] == 'baseline')
    promotions = [r for r in data['rows'] if r['protocol'] == 'promotion']
    pbase = next(r for r in promotions if r['id'] == 'baseline')
    assert len(screens) == 23 and len(promotions) == 7
    assert all(r['steps'] == 500 for r in screens) and all(r['steps'] == 1500 for r in promotions)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=.09, right=.98, bottom=.23, top=.84)
    xs = [int(r['id'][1:]) for r in screens]
    ys = [r['loss'] for r in screens]
    ax.scatter(xs, ys, s=36, color=GRAY, zorder=3, label='Measured candidate')
    ax.step(xs, np.minimum.accumulate(ys), where='post', color=BLUE, linewidth=1.4, alpha=.7, label='Best screen so far')
    ax.axhline(base['loss'], color=ORANGE, linestyle='--', linewidth=1.2, label=f"Matched baseline · {base['loss']:.4f}")
    n8 = next(r for r in screens if r['id'] == 'n0008')
    ax.scatter([8], [n8['loss']], marker='*', s=180, color=BLUE, edgecolor='white', zorder=4)
    ax.annotate('n0008', (8, n8['loss']), xytext=(5, 12), textcoords='offset points', color=BLUE, fontsize=9)
    for n, dy in ((1,-16),(24,-16)):
        r=next(r for r in screens if int(r['id'][1:])==n)
        ax.annotate(f"{r['id']} · {r['loss']:.4f}",(n,r['loss']),xytext=(4,dy),textcoords='offset points',fontsize=9)
    ax.set_xticks([1,5,8,12,16,20,24,27], ['n0001','n0005','n0008','n0012','n0016','n0020','n0024','n0027'])
    ax.set_xlabel('Candidate order')
    ax.set_ylabel('Held-out total loss ↓')
    ax.set_xlim(0,28)
    ax.set_ylim(min(ys)-.028,max(max(ys),base['loss'])+.035)
    ax.set_title('Overnight search · 23 measured 500-step screens',loc='left',pad=35)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.11),frameon=False,ncol=3,fontsize=9)
    style(ax)
    save(fig,out,'overnight_screens','RTX 3090 · seed 42 · 60 held-out samples · 4 unmeasured/interrupted attempts omitted · current GPU-processor study only')

    fig, axes = plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.10,right=.97,bottom=.20,top=.83,wspace=.38)
    ordered=sorted(promotions,key=lambda r:r['loss'],reverse=True)
    ax=axes[0]
    for i,r in enumerate(ordered):
        color=BLUE if r['id']=='n0008' else ORANGE if r['id']=='baseline' else GRAY
        ax.scatter(r['loss'],i,s=70,color=color,zorder=3)
        ax.annotate(f"{r['loss']:.4f}",(r['loss'],i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9,color=color)
    ax.set_yticks(range(len(ordered)),[r['id'] for r in ordered]); ax.invert_yaxis()
    ax.set_xlim(.268,.331); ax.set_xlabel('Held-out total loss ↓'); ax.set_title('A  ·  1,500-step comparison',loc='left'); style(ax)
    ax=axes[1]
    ids=[r['id'] for r in promotions if r['id']!='baseline']
    screenmap={r['id']:r for r in screens}
    ranked_ids=sorted(ids,key=lambda ident: next(r['loss'] for r in promotions if r['id']==ident),reverse=True)
    endpoint_labels={ident: 8.8+1.15*i for i,ident in enumerate(ranked_ids)}
    for ident in ids:
        r=next(r for r in promotions if r['id']==ident)
        values=[100*(base['loss']-screenmap[ident]['loss'])/base['loss'],100*(pbase['loss']-r['loss'])/pbase['loss']]
        color=BLUE if ident=='n0008' else GRAY
        ax.plot([0,1],values,color=color,alpha=1 if ident=='n0008' else .45,linewidth=2 if ident=='n0008' else 1,marker='o',markersize=5)
        ax.plot([1.03,1.14],[values[1],endpoint_labels[ident]],color=color,linewidth=.6,alpha=.6)
        ax.text(1.16,endpoint_labels[ident],ident,va='center',fontsize=8,color=color)
    ax.axhline(0,color=ORANGE,linestyle='--',linewidth=1)
    ax.set_xticks([0,1],['500-step screen','1,500-step promotion']);ax.set_xlim(-.15,1.42)
    ax.set_ylabel('Loss reduction vs matched baseline · % ↑');ax.set_title('B  ·  Ranking changes with budget',loc='left');style(ax)
    fig.suptitle('n0008 achieves the lowest promoted loss · 13.6% below baseline',x=.10,ha='left',fontsize=13,fontweight='bold')
    save(fig,out,'overnight_promotions','Independent fixed-budget probes · baselines: 500 steps = 0.4074; 1,500 steps = 0.3197 · single seed; no uncertainty estimate')

    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.80,wspace=.29)
    for ax,metric,title,label in zip(axes,['peak_gib','wall_minutes'],['A  ·  Training memory (zoomed scale)','B  ·  Total probe duration'],['Peak allocated training VRAM · GiB ↓','Setup + training + validation · min ↓']):
        frontier=[r for r in promotions if not any(s[metric]<=r[metric] and s['loss']<=r['loss'] and (s[metric]<r[metric] or s['loss']<r['loss']) for s in promotions)]
        frontier=sorted(frontier,key=lambda r:r[metric])
        ax.plot([r[metric] for r in frontier],[r['loss'] for r in frontier],color=BLUE,linestyle='--',linewidth=1,alpha=.65)
        for r in promotions:
            ident=r['id'];color=BLUE if ident=='n0008' else ORANGE if ident=='baseline' else GRAY
            ax.scatter(r[metric],r['loss'],s=140 if ident=='n0008' else 45,marker='*' if ident=='n0008' else 'o',color=color,zorder=3)
            offsets={'n0001':(-42,7),'n0005':(5,8),'n0006':(-43,-13),'n0007':(5,8),'n0008':(5,-14),'n0010':(5,4),'baseline':(5,-3)}
            ax.annotate(ident,(r[metric],r['loss']),xytext=offsets[ident],textcoords='offset points',fontsize=8,color=color)
        ax.set_xlabel(label);ax.set_ylabel('Held-out total loss ↓');ax.set_title(title,loc='left');ax.set_ylim(.268,.325);style(ax)
        if metric=='peak_gib':ax.set_xlim(7.486,7.530)
        else:ax.set_xlim(19.60,21.35)
    fig.suptitle('Measured promotion trade-offs · 1,500 steps',x=.09,ha='left',fontsize=13,fontweight='bold')
    fig.text(.09,.12,'n0008: 7.491 GiB vs baseline 7.493 GiB — essentially unchanged memory.',fontsize=9,color=GRAY)
    save(fig,out,'overnight_pareto','Dashed line: non-dominated measurements · total probe duration is not inference latency · one measurement per configuration')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--journal',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'studies/dream-rsi/figures');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);snapshot=args.output/'overnight_metrics.json'
    if args.journal:
        data=extract(args.journal);snapshot.write_text(json.dumps(data,indent=2)+'\n')
    else:data=json.loads(snapshot.read_text())
    plot(data,args.output)
    print(f"Saved 3 PNG/PDF figures and measured data to {args.output}")
if __name__=='__main__':main()
