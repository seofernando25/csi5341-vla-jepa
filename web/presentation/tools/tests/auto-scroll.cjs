const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function harness() {
  let frame, now=0;
  const nodes = {};
  for (const id of ['script-reader','speaking-notes','scroll-play','scroll-speed','scroll-status','speed-value','scroll-top'])
    nodes[id] = {value:'14',textContent:'',scrollTop:0,offsetHeight:1550,events:{},setAttribute(){},addEventListener(name,fn){this.events[name]=fn;}};
  const context = vm.createContext({document:{hidden:false,getElementById:id=>nodes[id]},window:{},localStorage:{getItem:()=>null,setItem(){}},performance:{now:()=>now},getComputedStyle:()=>({lineHeight:'55'}),requestAnimationFrame:fn=>{frame=fn;}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../../src/sync/auto-scroll.js'),'utf8'),context);
  return {nodes,api:context.window.prompterScroll,step(n){now=n;frame(n);},run(from,to,hz){for(let t=from;t<=to;t+=1000/hz){now=t;frame(t);}}};
}
const a=harness();
a.api.setConnected(true);a.api.chapterChanged();a.step(1000);assert.equal(a.nodes['script-reader'].scrollTop,0,'reading lead-in');
a.run(1500,3500,60);assert.ok(a.nodes['script-reader'].scrollTop>27 && a.nodes['script-reader'].scrollTop<31);
const at=a.nodes['script-reader'].scrollTop;
a.nodes['scroll-play'].onclick();a.run(3500,4500,60);assert.equal(a.nodes['script-reader'].scrollTop,at,'pause holds');
a.nodes['scroll-play'].onclick();a.run(4500,5500,60);assert.ok(a.nodes['script-reader'].scrollTop>at);
a.api.setConnected(false);const offline=a.nodes['script-reader'].scrollTop;a.run(5500,6500,60);assert.equal(a.nodes['script-reader'].scrollTop,offline,'offline holds');
a.api.setConnected(true);a.nodes['script-reader'].events.wheel();a.run(6500,7500,60);assert.equal(a.nodes['script-reader'].scrollTop,offline,'manual reading pauses auto');
a.nodes['scroll-play'].onclick();a.api.chapterChanged();assert.equal(a.nodes['script-reader'].scrollTop,0,'new chapter resets');
a.nodes['speaking-notes'].offsetHeight=100;a.run(9000,15000,60);assert.equal(a.nodes['script-reader'].scrollTop,45,'last line reaches top and holds');assert.equal(a.nodes['scroll-play'].textContent,'Restart scroll');
a.nodes['scroll-play'].onclick();assert.equal(a.nodes['script-reader'].scrollTop,0);
for(const hz of [30,60,120]) {const b=harness();b.api.setConnected(true);b.api.chapterChanged();b.run(1500,3500,hz);assert.ok(Math.abs(b.nodes['script-reader'].scrollTop-28)<2,`${hz}fps uses elapsed time`);}
console.log('Auto scroll: lead-in, pace, pause/resume, manual override, offline hold, chapter reset, final-line hold and 30/60/120 Hz verified.');
