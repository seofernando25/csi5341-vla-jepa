const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'../..');const context=vm.createContext({});vm.runInContext(fs.readFileSync(path.join(root,'src/beat-controller.js'),'utf8'),context);
const cues=vm.runInContext('PRESENTATION_BEATS',context);
// Independently authored semantic change times. Each text replacement needs its own click.
const changes={2:[6.1],3:[9,36,55],9:[10,20,34],10:[4,9.75,15.5,27],11:[24]};
for(const [scene,times] of Object.entries(changes)){const stops=cues[scene];assert(stops[0]<times[0],`Scene ${scene}: initial cue crosses a replacement`);let before=0;for(const stop of stops){assert(times.filter(t=>t>before&&t<=stop).length<=1,`Scene ${scene}: multiple replacements per click`);before=stop;}assert(stops.at(-1)>=times.at(-1)+.48,`Scene ${scene}: last replacement does not finish`);}
for(const [scene,stops] of Object.entries(cues))assert(stops.every((t,j)=>t>0&&(!j||t>stops[j-1])),`Scene ${scene}: unordered cues`);
const player=fs.readFileSync(path.join(root,'src/player.js'),'utf8');assert(player.includes('presenter.manual?sceneSeconds()'),'Dataset explanation must use the held scene clock');
console.log('All 18 scene cue lists checked; 12 text replacements require separate clicks.');
