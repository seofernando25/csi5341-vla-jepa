const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.join(__dirname,'../../src');
let stream;
const nodes={};
const buttons=['prev','next','animate','chapter'].map(id=>({id,hidden:false,disabled:false}));
const document={querySelector:()=>({}),querySelectorAll:()=>buttons,getElementById:id=>nodes[id]??=( {textContent:'',append(){}}),createElement:()=>({style:{},setAttribute(){}})};
const context=vm.createContext({document,window:{},location:{search:''},URLSearchParams,performance:{now:()=>1000},Date,setInterval(){},EventSource:class {constructor(){stream=this;}}});
for(const file of ['timing/clock.js','timing/cues.js']) vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
vm.runInContext(`let index=0;const presenter=new CueClock();function cueSpecs(){return SCENE_CUES[index]}function select(n){index=n;presenter.load(cueSpecs(),1000)}function syncPresenterMedia(){}function syncPresenterFrame(){}select(0);`,context);
vm.runInContext(fs.readFileSync(path.join(root,'sync/viewer.js'),'utf8'),context);
assert.equal(context.window.deckReadOnly,true);assert.ok(buttons.every(b=>b.disabled));
const specs=vm.runInContext('SCENE_CUES',context);
let checks=0;
for(const [scene,cues] of Object.entries(specs)) for(let cue=0;cue<cues.length;cue++){
  const spec=cues[cue];
  const state={scene:Number(scene),cue,running:true,direction:1,seconds:spec.start,progress:0,remaining:spec.duration};
  stream.onmessage({data:JSON.stringify({state,controllerOnline:true,ageMs:spec.duration/2})});
  const result=vm.runInContext('({scene:index,cue:presenter.cursor,seconds:presenter.seconds,remaining:presenter.run.duration,progress:presenter.progress})',context);
  assert.equal(result.scene,Number(scene));assert.equal(result.cue,cue);assert.ok(Math.abs(result.seconds-(spec.start+spec.end)/2)<1e-7);assert.equal(result.remaining,spec.duration/2);assert.equal(result.progress,0.5);checks++;
  if(cue>0){
    state.direction=-1;state.seconds=spec.end;
    stream.onmessage({data:JSON.stringify({state,controllerOnline:true,ageMs:0})});
    vm.runInContext(`presenter.tick(${1000+spec.duration});`,context);
    const previous=vm.runInContext('({cue:presenter.cursor,seconds:presenter.seconds})',context);
    assert.equal(previous.cue,cue-1);assert.equal(previous.seconds,cues[cue-1].end);checks++;
  }
}
stream.onerror();const held={scene:3,cue:4,running:false,seconds:specs[3][4].end,progress:1,remaining:0,direction:0};
stream.onmessage({data:JSON.stringify({state:held,controllerOnline:true,ageMs:0})});assert.equal(vm.runInContext('presenter.cursor',context),4);
stream.onerror();stream.onmessage({data:JSON.stringify({state:held,controllerOnline:true,ageMs:0})});assert.equal(vm.runInContext('presenter.seconds',context),held.seconds);
vm.runInContext(fs.readFileSync(path.join(root,'beat-controller.js'),'utf8').replace('const presenter = new CueClock();',''),context);
vm.runInContext('advanceBeat(1);advanceBeat(-1);replayBeat();',context); // No audio/clock/navigation allowed for viewers.
assert.equal(vm.runInContext('presenter.cursor',context),4);
console.log(`${checks} viewer animation snapshots checked across all chapters, including reverse completion, age compensation, reconnect and read-only guards.`);
