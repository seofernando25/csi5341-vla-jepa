const $ = id => document.getElementById(id);
let state = null, view = 'chart', selectedStudy = '.rsi', fetching = false, refreshTimer = null, openAttempt = null;
let selectedCurrentRun = null;
const finite = x => typeof x === 'number' && Number.isFinite(x);
const esc = x => String(x ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function choices(id, items) {
  const old = $(id).value;
  $(id).replaceChildren(...items.map(([value, text]) => new Option(text, value)));
  if (items.some(x => x[0] === old)) $(id).value = old;
}
function detail(row) {
  openAttempt = state?.attempts.some(r=>r.id===row.id) ? row.id : null;
  const proposal = row.proposal || row.draft_proposal;
  $('detail-title').textContent = row.id || row.variant || 'Measurement';
  let overview = proposal?.structural_change || row.status || '';
  if(row.summary?.episodes) overview = `${row.summary.successes} / ${row.summary.episodes} successful episodes`;
  if(row.summary?.pipeline) overview = `${row.summary.pipeline.median_ms.toFixed(1)} ms pipeline latency`;
  if(finite(row.loss)) overview += ` · loss ${row.loss.toFixed(4)}${row.samples ? ' · '+row.samples+' samples' : ''}`;
  if(row.results?.success) overview = `${(100*row.results.success.rate).toFixed(1)}% success · ${row.results.success.episodes} episodes · ${row.results.timing.pipeline.median_ms.toFixed(1)} ms`;
  $('detail-summary').textContent = overview.replace(/^ · /,'');
  const fields = [['Hypothesis','hypothesis'],['Proposed change','structural_change'],['Expected effect','expected_effect'],['Limitations noted at proposal time','limitations'],['Compared with parent','relation_to_parent'],['Proposal worker checks · before GPU evaluation','implementation_checks']];
  const prose = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  $('proposal-plan').innerHTML = proposal ? `<p class="plan-status">${row.proposal ? 'Recorded proposal' : 'Worker draft · awaiting acceptance'}</p>` + fields.filter(([,key])=>proposal[key]).map(([label,key])=>`<h3>${label}</h3><p>${esc(prose(proposal[key]))}</p>`).join('') : '';
  $('worker-history').innerHTML = (row.proposal_history || []).map(h=>`<details><summary>${esc(h.retry)} · authored notes</summary>${h.messages.map(m=>`<p>${esc(m)}</p>`).join('')}${h.proposal ? `<pre>${esc(JSON.stringify(h.proposal,null,2))}</pre>` : ''}<small>${esc(h.source)}</small></details>`).join('');
  $('timeline').innerHTML = (row.timeline || []).map(e => `<li><time>${time(e.at || e.ended_at || e.started_at)}</time><span>${esc(e.kind.replaceAll('_',' '))}${e.reason ? ' · '+esc(e.reason) : ''}${e.proposal?.structural_change ? ' · '+esc(e.proposal.structural_change) : ''}</span></li>`).join('');
  $('detail-body').textContent = JSON.stringify(row, null, 2);
  if(!$('detail').open)$('detail').showModal();
}
const time = x => x ? new Date(x).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
const colors = {B16:'#526172',Q8:'#ba6c27',Q4:'#1965d2',S500:'#9061bc'};
function graphSetup() {
  const d = $('dataset').value;
  choices('metric', d === 'final' ? [['latency','Pipeline latency · ms'],['memory','Inference VRAM · GiB']] :
    d === 'benchmarks' ? [['memory','Inference VRAM · GiB']] :
    d === 'learning' ? [['steps','Optimizer steps']] : [['memory','Training VRAM · GiB'],['seconds','Probe wall time · s']]);
  $('gpu-label').hidden = d === 'learning';
  $('protocol-label').hidden = d !== 'benchmarks';
  const available = d === 'final' ? state.evaluation.final : d === 'benchmarks' ? state.evaluation.benchmarks : state.points.filter(p => p.phase === d);
  const gpus = [...new Set(available.map(p => p.gpu))];
  choices('gpu', gpus.length ? gpus.map(g => [g,g]) : [['','No measured GPU yet']]);
  const protocols = [...new Set(state.evaluation.benchmarks.map(p => `${p.predictions}×${p.repetitions}`))];
  choices('protocol', protocols.sort().reverse().map(p => [p,p+' predictions × repetitions']));
}
function plot() {
  const d = $('dataset').value, metric = $('metric').value;
  let points = d === 'final' ? state.evaluation.final : d === 'benchmarks' ? state.evaluation.benchmarks : d === 'learning' ? state.evaluation.learning : state.points.filter(p => p.phase === d);
  if (d !== 'learning') points = points.filter(p => p.gpu === $('gpu').value);
  if (d === 'benchmarks') points = points.filter(p => `${p.predictions}×${p.repetitions}` === $('protocol').value);
  const ykey = d === 'final' ? 'success' : d === 'benchmarks' ? 'latency' : 'loss';
  points = points.filter(p => finite(p[metric]) && finite(p[ykey]));
  const higher = d === 'final';
  $('plot-note').textContent = d === 'final' ? '500 LIBERO episodes / model · higher success, lower cost · bars: 95% Wilson intervals' :
    d === 'benchmarks' ? 'All recorded inference benchmarks in the selected timing protocol · lower left is better' :
    d === 'learning' ? 'Earlier SmolVLM adaptation · 200 held-out samples / checkpoint · separate from Dream-RSI' :
    `${d === 'screen' ? state.config.screen_steps : state.config.promotion_steps}-step search protocol · lower left is better · loss ≠ LIBERO success`;
  $('legend').textContent = d === 'learning' ? `${points.length} validation checkpoints` : `● Frontier　○ Other runs　· ${points.length} measurements`;
  if (!points.length) {
    $('plot').innerHTML = `<div class="empty">${d === 'screen' || d === 'promotion' ? 'This fresh search has no completed measurements yet.<br>Earlier experiments are in the other result views.' : 'No completed measurements for this selection.'}</div>`;
    return;
  }
  const ycost = p => higher ? -p[ykey] : p[ykey];
  const frontier = points.filter(p => !points.some(q => q[metric] <= p[metric] && ycost(q) <= ycost(p) && (q[metric] < p[metric] || ycost(q) < ycost(p))));
  let xmin=Math.min(...points.map(p=>p[metric])), xmax=Math.max(...points.map(p=>p[metric]));
  let ymin=Math.min(...points.map(p=>p[ykey])), ymax=Math.max(...points.map(p=>p[ykey]));
  if (higher) { ymin=0; ymax=100; } else { const dy=ymax-ymin || Math.max(ymax*.1,.01); ymin=Math.max(0,ymin-dy*.13); ymax+=dy*.18; }
  const dx=xmax-xmin || Math.max(xmax*.1,1); xmin=Math.max(0,xmin-dx*.12); xmax+=dx*.15;
  const X=x=>82+(x-xmin)/(xmax-xmin)*800, Y=y=>330-(y-ymin)/(ymax-ymin)*290;
  const fmt=x=>metric === 'memory' ? x.toFixed(2) : x.toFixed(0);
  let svg='<svg viewBox="0 0 960 395" role="img" aria-label="Recorded experiment measurements">';
  for(let i=0;i<5;i++) {
    const x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;
    svg+=`<path d="M82 ${Y(y)}H882" stroke="#e8edf1"/><text x="66" y="${Y(y)+4}" text-anchor="end" fill="#65717d" font-size="12">${ykey==='loss'?y.toFixed(3):y.toFixed(0)}</text><text x="${X(x)}" y="355" text-anchor="middle" fill="#65717d" font-size="12">${fmt(x)}</text>`;
  }
  const ylabel = higher ? 'LIBERO success · % ↑' : ykey === 'latency' ? 'Pipeline latency · ms ↓' : 'Held-out loss ↓';
  svg+=`<text x="22" y="185" transform="rotate(-90 22 185)" text-anchor="middle" fill="#65717d" font-size="12">${ylabel}</text><text x="480" y="385" text-anchor="middle" fill="#65717d" font-size="12">${esc($('metric').selectedOptions[0].text)}${d==='learning'?'':' ↓'}</text>`;
  const joined = (d === 'learning' ? points : frontier).slice().sort((a,b)=>a[metric]-b[metric]);
  if(joined.length>1) svg+=`<polyline points="${joined.map(p=>`${X(p[metric])},${Y(p[ykey])}`).join(' ')}" fill="none" stroke="#a9bbce" stroke-width="1.5" ${d==='learning'?'':'stroke-dasharray="4 4"'}/>`;
  const labels = [];
  points.forEach((p,i)=>{
    const nondominated = frontier.includes(p), color = colors[p.label] || (p.baseline?'#ba6c27':'#1965d2');
    if (higher && p.ci95?.length===2) svg+=`<path d="M${X(p[metric])} ${Y(p.ci95[0])}V${Y(p.ci95[1])}" stroke="${color}"/>`;
    svg+=`<g data-point="${i}" tabindex="0" role="button" aria-label="Details: ${esc(p.id)}"><circle cx="${X(p[metric])}" cy="${Y(p[ykey])}" r="${d==='learning'?4:6}" fill="${d==='learning'||nondominated?color:'white'}" stroke="${color}" stroke-width="2"><title>${esc(p.id)} · ${p[ykey].toFixed(3)} · ${p[metric].toFixed(2)}</title></circle>`;
    if(d!=='learning') {
      const text=p.label || p.id, width=text.length*6.5;
      let lx=X(p[metric])+10, ly=Y(p[ykey])-12;
      for(const offset of [-12,22,-32,42,62]) {
        ly=Math.min(325,Math.max(18,Y(p[ykey])+offset));
        if(!labels.some(b=>Math.abs(b.y-ly)<17 && lx<b.x+b.width+5 && lx+width>b.x-5))break;
      }
      labels.push({x:lx,y:ly,width});
      svg+=`<text x="${lx}" y="${ly}" fill="${color}" font-size="12">${esc(text)}</text>`;
    }
    svg+='</g>';
  });
  $('plot').innerHTML=svg+'</svg>';
  for (const g of $('plot').querySelectorAll('[data-point]')) {
    const open = () => { const p=points[Number(g.dataset.point)]; detail(p.detail || state.attempts.find(r=>r.id===p.id.split(' · ')[0]) || p); };
    g.onclick=open; g.onkeydown=e=>{if(e.key==='Enter')open();};
  }
}
function currentRun() {
  const rollout=state.evaluation.runs.slice().reverse().find(r=>r.experiment==='rollout'&&r.status==='running');
  const options=[];
  if(rollout) options.push({id:'rollout:'+rollout.id,label:rollout.variant+' · LIBERO',running:true,rollout});
  if(state.confirmation?.progress) options.push({id:'confirmation',label:state.confirmation.progress.label+(state.confirmation.progress.cloud?' · cloud':' · confirmation'),running:state.confirmation.status==='running',progress:state.confirmation.progress});
  for(const job of state.local_training_jobs||[]) options.push({...job,running:job.status==='running'});
  if(state.progress) options.push({id:'search',label:'Search training',running:state.service==='active',progress:state.progress});
  if(!options.some(o=>o.id===selectedCurrentRun)) selectedCurrentRun=(options.find(o=>o.running)||options[0])?.id;
  choices('current-run',options.map(o=>[o.id,o.label]));$('current-run').value=selectedCurrentRun||'';
  const chosen=options.find(o=>o.id===selectedCurrentRun);
  const selectedRollout=chosen?.rollout;
  if(selectedRollout){
    const rollout=selectedRollout;
    const p=rollout.progress;
    $('current-title').textContent=`${rollout.variant} · LIBERO ${rollout.phase} · running`;
    $('current-stats').textContent=`${p.completed} / ${p.total} episodes · ${p.successes} successes so far`;
    $('current-note').textContent=`Latest completed task: ${p.last_task||'initializing'} · partial results · updates every 3s`;
    let svg='<svg viewBox="0 0 960 395" role="img" aria-label="Cumulative LIBERO success rate by completed episode">';
    const X=x=>82+800*x/Math.max(p.total,1),Y=y=>330-2.9*y;
    for(let i=0;i<=4;i++){const y=i*25,x=p.total*i/4;svg+=`<path d="M82 ${Y(y)}H882" stroke="#e8edf1"/><text x="66" y="${Y(y)+4}" text-anchor="end" fill="#65717d">${y}%</text><text x="${X(x)}" y="355" text-anchor="middle" fill="#65717d">${Math.round(x)}</text>`;}
    svg+=`<polyline points="${p.curve.map(v=>`${X(v.episode)},${Y(v.success_rate)}`).join(' ')}" fill="none" stroke="#1965d2" stroke-width="2"/>`;
    svg+='<text x="480" y="385" text-anchor="middle" fill="#65717d">Completed episode</text><text x="22" y="185" transform="rotate(-90 22 185)" text-anchor="middle" fill="#65717d">Cumulative success · %</text>';
    $('current-plot').innerHTML=svg+'</svg>';return;
  }

  const p=chosen?.progress, points=p?.curve || [];
  $('current-title').textContent=p ? `${p.label} · ${p.phase} · ${p.completed?'completed':chosen.running?'running':chosen.status||'last observed run'}` : 'Waiting for a training run';
  const last=points.at(-1);
  $('current-stats').textContent=p ? `${p.step??'—'} / ${p.total??'—'} ${p.unit||'optimizer updates'}${last ? ` · loss ${last.loss.toFixed(3)} at update ${last.step}` : ''}` : '';
  $('current-note').textContent=(p?.cloud ? 'Cloud logs sync every minute · held-out results appear after evaluation. ' : 'Logged training loss · updates every 3s · held-out results appear after evaluation. ')+(p ? `Log updated ${new Date(p.updated_at*1000).toLocaleTimeString()}` : '');
  if(p?.averaged_microbatches) $('current-note').textContent=`Mean loss over ${p.averaged_microbatches} microbatches per update · ${p.native_microstep}/${p.native_total_microsteps} native microsteps · updated ${new Date(p.updated_at*1000).toLocaleTimeString()}`;
  if(!points.length){$('current-plot').innerHTML='<div class="empty">Waiting for the first logged training loss.</div>';return;}
  const keys=[['loss','Total','#1965d2'],['action_loss','Action','#ba6c27'],['wm_loss','World model','#357a57']];
  const values=points.flatMap(p=>keys.map(([k])=>p[k])).filter(finite);
  const low=Math.max(0,Math.min(...values)*.9), high=Math.max(...values)*1.08 || 1;
  const maxStep=Math.max(p.total||0,...points.map(p=>p.step),1);
  const X=x=>82+x/maxStep*800,Y=y=>330-(y-low)/(high-low)*290;
  let svg='<svg viewBox="0 0 960 395" role="img" aria-label="Training loss by optimizer step">';
  for(let i=0;i<5;i++){const y=low+(high-low)*i/4,x=maxStep*i/4;svg+=`<path d="M82 ${Y(y)}H882" stroke="#e8edf1"/><text x="66" y="${Y(y)+4}" text-anchor="end" fill="#65717d" font-size="12">${y.toFixed(3)}</text><text x="${X(x)}" y="355" text-anchor="middle" fill="#65717d" font-size="12">${Math.round(x)}</text>`;}
  for(const [key,label,color] of keys){const valid=points.filter(p=>finite(p[key]));svg+=`<polyline points="${valid.map(p=>`${X(p.step)},${Y(p[key])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;for(const p of valid)svg+=`<circle cx="${X(p.step)}" cy="${Y(p[key])}" r="2.5" fill="${color}"><title>${label} · step ${p.step}: ${p[key].toFixed(3)}</title></circle>`;}
  svg+='<text x="22" y="185" transform="rotate(-90 22 185)" text-anchor="middle" fill="#65717d" font-size="12">Training loss</text><text x="480" y="385" text-anchor="middle" fill="#65717d" font-size="12">Optimizer step</text>';
  keys.forEach(([,label,color],i)=>{svg+=`<text x="${100+i*130}" y="20" fill="${color}" font-size="12">● ${label}</text>`;});
  $('current-plot').innerHTML=svg+'</svg>';
}
function rows(target, records, cells) {
  $(target).replaceChildren();
  records.forEach(r => {
    const tr=document.createElement('tr');tr.tabIndex=0;tr.classList.toggle('active',!!r.active);
    tr.innerHTML=cells(r); tr.onclick=()=>detail(r);tr.onkeydown=e=>{if(e.key==='Enter')detail(r);};$(target).append(tr);
  });
}
function history() {
  const batch=$('batch').value,q=$('search').value.toLowerCase();
  const candidates=state.attempts.filter(r=>(batch==='all'||r.batch===batch)&&JSON.stringify(r).toLowerCase().includes(q)).sort((a,b)=>Number(!!b.active)-Number(!!a.active)||b.id.localeCompare(a.id));
  rows('rows',candidates,r=>`<td><strong>${esc(r.id)}${r.active?' ●':''}</strong></td><td>${esc(r.batch)}<small>← ${esc(r.parent)}</small></td><td>${esc(r.proposal?.structural_change||r.draft_proposal?.structural_change||r.summary||'Agent constructing candidate')}<small>${esc(r.proposal?.mechanism_family||r.draft_proposal?.mechanism_family||'')}</small></td><td><span class="badge">${esc(r.status)}</span></td><td>${finite(r.loss)?r.loss.toFixed(4):'—'}</td>`);
  $('empty').textContent=candidates.length?'':state.attempts.length?'No matches.':'No proposal batches yet — the fresh search is measuring its baseline.';
  const rq=$('run-search').value.toLowerCase();
  rows('run-rows',state.evaluation.runs.filter(r=>JSON.stringify(r).toLowerCase().includes(rq)).slice().reverse(),r=>{
    const s=r.summary; let result=s.task_macro_success !== undefined ? `${(100*s.task_macro_success).toFixed(1)}% · ${s.successes}/${s.episodes}` : s.pipeline?.median_ms ? `${s.pipeline.median_ms.toFixed(1)} ms` : s.median_ms ? `${s.median_ms.toFixed(1)} ms` : '—';
    if(r.progress && r.status==='running') result=`${r.progress.completed}/${r.progress.total} episodes · ${r.progress.successes} successes so far`;
    return `<td><strong>${esc(r.variant)}</strong><small>${esc(r.id)}</small></td><td>${esc(r.experiment)}</td><td>${esc(r.phase)}</td><td>${esc(r.status)}</td><td>${esc(result)}</td>`;
  });
}
function render() {
  choices('study',state.studies.map(s=>[s.id,s.label]));$('study').value=state.study;
  const p=state.confirmation?.status==='running' ? state.confirmation.progress : state.progress;
  $('stage').textContent=state.confirmation?.status==='running' ? `${state.confirmation.arm} · ${state.confirmation.stage}` : state.service==='active'?state.stage:`Search ${state.service}`;
  $('progress-text').textContent=p?.total?`${p.step ?? '—'} / ${p.total} · latest log`:'';
  $('progress').value=p?.total?100*(p.step||0)/p.total:0;
  const liveRollout=state.evaluation.runs.slice().reverse().find(r=>r.experiment==='rollout'&&r.status==='running');
  if(liveRollout){const q=liveRollout.progress;$('stage').textContent=`${liveRollout.variant} · LIBERO ${liveRollout.phase}`;$('progress-text').textContent=`${q.completed}/${q.total} episodes · ${q.successes} successes`;$('progress').value=100*q.completed/Math.max(q.total,1);}
  $('totals').textContent=`${state.counts.batches} batches · ${state.counts.measured} search measurements · ${state.disk_free_gib.toFixed(0)} GiB free`;
  $('run-count').textContent=state.evaluation.runs.length; $('proposal-count').textContent=state.attempts.length;
  choices('batch',[['all','All batches'],...state.batches.map(b=>[b.id,`${b.id} · cycle ${b.cycle}`])]);
  $('activity').innerHTML=state.events.slice().reverse().map(e=>`<li><time>${time(e.at)}</time><span>${esc(e.kind.replaceAll('_',' '))}${e.attempt?' · '+esc(e.attempt):''}${e.batch?' · '+esc(e.batch):''}${e.reason?' · '+esc(e.reason):''}</span></li>`).join('');
  $('updated').textContent='Updated '+new Date(state.updated_at).toLocaleTimeString();
  graphSetup(); plot(); history(); currentRun();
  if($('detail').open && openAttempt){const row=state.attempts.find(r=>r.id===openAttempt);if(row)detail(row);}
}
async function refresh() {
  if(fetching)return;clearTimeout(refreshTimer);fetching=true;
  try { const response=await fetch('/api/state?study='+encodeURIComponent(selectedStudy),{cache:'no-store'});if(!response.ok)throw Error();state=await response.json();render();$('connection').textContent='● Live · '+(state.confirmation?.status==='running'?'confirmation running':state.service); }
  catch { $('connection').textContent='Reconnecting…'; }
  finally {fetching=false;refreshTimer=setTimeout(refresh,3000);}
}
for(const button of document.querySelectorAll('nav button')) button.onclick=()=>{
  view=button.dataset.view;for(const b of document.querySelectorAll('nav button')){if(b===button)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');}
  for(const v of ['chart','current','runs','proposals','activity'])$(v+'-view').hidden=v!==view;
};
$('study').onchange=()=>{$('detail').close();openAttempt=null;selectedStudy=$('study').value;refresh();};
$('current-run').onchange=()=>{selectedCurrentRun=$('current-run').value;if(state)currentRun();};
$('dataset').onchange=()=>{if(state){graphSetup();plot();}};
for(const id of ['metric','gpu','protocol'])$(id).onchange=()=>state&&plot();
$('batch').onchange=()=>state&&history();for(const id of ['search','run-search'])$(id).oninput=()=>state&&history();
$('close').onclick=()=>$('detail').close();refresh();
