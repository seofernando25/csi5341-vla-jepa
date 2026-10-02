/* Shared objects persist across the method chapters. Positions, not slides, change. */
const motion = {from: null, began: 0, duration: 900, exportTime: null, dataFrom: null};
const mix=(a,b,t)=>a+(b-a)*t;
const beat=(seconds,start,length=.65)=>ease((seconds-start)/length);
const poseFor=i=>({
 4:{now:[70,185,285,160],future:[605,185,285,160],model:[435,256],z:[480,168],pred:[350,405],target:[730,405]},
 5:{now:[70,190,255,143],future:[759,370,123,69],model:[450,264],z:[730,264],pred:[480,407],target:[800,407]},
 6:{now:[65,198,175,99],future:[742,194,145,82],model:[475,285],z:[475,174],pred:[710,365],target:[810,365]},
 7:{now:[65,194,160,90],future:[65,337,160,90],model:[335,281],z:[335,167],pred:[570,297],target:[785,297]},
 8:{now:[65,183,180,102],future:[710,359,180,102],model:[485,279],z:[485,176],pred:[760,279],target:[760,394]}
}[i]||null);
function beginTransform(next){motion.dataFrom=charts[index]&&charts[next]?charts[index].values:null;if(index>=4&&index<=8&&next>=4&&next<=8){motion.from=poseFor(index);motion.began=performance.now();}else motion.from=null;}
function methodPose(){let a=poseFor(index);if(!motion.from)return a;let t=ease((motion.exportTime ?? (performance.now()-motion.began))/motion.duration),r={};for(let key in a)r[key]=a[key].map((v,j)=>mix(motion.from[key][j],v,t));if(t===1)motion.from=null;return r;}
function photo(c,im,x,y,w,h){c.save();c.beginPath();c.roundRect(x,y,w,h,10);c.clip();c.fillStyle='#eef2f4';c.fillRect(x,y,w,h);if(im.complete&&im.naturalWidth){let k=Math.min(w/im.naturalWidth,h/im.naturalHeight);c.drawImage(im,x+(w-im.naturalWidth*k)/2,y+(h-im.naturalHeight*k)/2,im.naturalWidth*k,im.naturalHeight*k);}c.restore();}
function pill(c,text,x,y,color=blue){c.font='600 12px "Plus Jakarta Sans"';let w=c.measureText(text).width+24;c.fillStyle=color+'12';c.beginPath();c.roundRect(x-w/2,y-14,w,26,13);c.fill();label(c,text,x,y+3,12,color,true,'center');}
function featureRibbon(c,x,y,w=125,color=blue,amount=1){const vals=[.3,.79,.52,.91,.24,.63,.4,.83,.32,.58,.92,.45];for(let j=0;j<vals.length;j++){let h=8+vals[j]*28*amount;c.fillStyle=color;c.globalAlpha=.3+vals[j]*.7;c.beginPath();c.roundRect(x-w/2+j*w/12,y-h/2,w/12-3,h,2);c.fill();}c.globalAlpha=1;}
function block(c,x,y,labelText,sub,color=ink){c.fillStyle='#f1f6f8';c.beginPath();c.roundRect(x-70,y-40,140,80,14);c.fill();label(c,labelText,x,y-3,20,color,true,'center');label(c,sub,x,y+21,11,muted,false,'center');}
function link(c,x,y,x2,y2,color=blue,t=1,bend=0){if(t<=0)return;const pts=[[x,y],[x+(x2-x)*.36,y+bend],[x+(x2-x)*.65,y2+bend],[x2,y2]];curve(c,pts,color,2,t);if(t>.98){let tail=bez(.98,pts);segment(c,...tail,x2,y2,color,1,true);}}
function methodMotion(c){const s=sceneSeconds(),a=methodPose(),now=a.now,future=a.future;
const title={4:'Predict a future state — in feature space',5:'The tokens describe an intended transition',6:'The predictor outputs the next state’s features',7:'The target stays fixed. The prediction learns.',8:'The policy never receives its future target'}[index];
headerC(c,'Method · '+({4:'From video to supervision',5:'Policy representation',6:'World prediction',7:'Predictive alignment',8:'The information boundary'}[index]),title,'Paper §3.2 · SSV2 validation example #174198 · Features are schematic; not measured robot coordinates');
photo(c,humanExamples[0],...now);photo(c,humanExamples[2],...future);
if(index===4){
 label(c,'Now',now[0],now[1]-18,17,ink,true);label(c,'Later',future[0],future[1]-18,17,ink,true);
 pill(c,'“Put the jar into the box”',480,175);
 link(c,365,258,595,258,blue,beat(s,7,.9),-30);
 const predOn=beat(s,19,.45),targetOn=beat(s,29,.45);
 frameToFeatures(c,humanExamples[0],now,[350,405,170],beat(s,17,1.6));
 frameToFeatures(c,humanExamples[2],future,[730,405,170],beat(s,27,1.6),orange);
 c.save();c.globalAlpha=predOn;featureRibbon(c,350,405,170);label(c,'Predicted future features',350,449,18,blue,true,'center');c.restore();
 c.save();c.globalAlpha=targetOn;featureRibbon(c,730,405,170,orange);label(c,'Encoded future target',730,449,18,orange,true,'center');c.restore();
 link(c,212,349,350,386,blue,beat(s,19,.6));link(c,748,349,730,385,orange,beat(s,29,.6));
 link(c,450,405,623,405,gray,beat(s,35,.65));emerge(c,s,35,()=>pill(c,'Match',538,392,ink));
 label(c,'Like anticipating “jar inside box” — without drawing the next frame.',52,489,18,ink);
 }else if(index===5){
 label(c,'Current observation',now[0],now[1]-16,16,muted,true);pill(c,'Put the jar into the box',198,384);
 block(c,...a.model,'VLM','Qwen3-VL-2B');link(c,335,261,380,264,blue,beat(s,3));link(c,285,384,380,284,blue,beat(s,3),25);
 emerge(c,s,8,()=>featureRibbon(c,...a.z,170));link(c,523,264,628,264,blue,beat(s,7));label(c,'Latent action tokens',730,322,20,blue,true,'center');
 label(c,'An intended transition',730,356,16,muted,false,'center');label(c,'Supervision only',820,461,12,orange,false,'center');
 label(c,'These tokens condition both prediction and control.',52,468,23,ink,true);label(c,'They are neither pixels nor executable motor commands.',52,493,16,muted);
 }else if(index===6){
 label(c,'Encoded state history',65,164,17,ink,true);featureRibbon(c,155,330,170,gray);pill(c,'Frozen V-JEPA 2 encoder',155,373,ink);
 block(c,...a.model,'Predictor','causal state history + tokens');featureRibbon(c,...a.z,130);label(c,'Latent action tokens',475,140,17,blue,true,'center');
 link(c,475,192,475,245,blue,beat(s,.2));link(c,255,330,403,290,gray,beat(s,.2));link(c,548,285,710,345,blue,beat(s,9),15);
 label(c,'Future target only',814,301,12,orange,false,'center');emerge(c,s,10,()=>featureRibbon(c,...a.pred,155));label(c,'Predicted next-state embedding',710,414,19,blue,true,'center');
 label(c,'The output describes the next state; the tokens describe the transition.',52,476,18,ink);label(c,'No image decoder is needed for this supervision objective.',52,499,16,muted);
 }else if(index===7){
 block(c,...a.model,'Predictor','trainable');featureRibbon(c,...a.z,105);link(c,335,187,335,241,blue,beat(s,.1));link(c,230,250,263,272,gray,beat(s,.1));curve(c,[[230,384],[430,477],[815,472],[785,426]],orange,2,beat(s,.2));if(s>1)segment(c,785,430,785,426,orange,1,true);link(c,405,281,540,281,blue,beat(s,.35));
 label(c,'Observed future',65,451,13,orange,true);label(c,'Predicted',570,227,17,blue,true,'center');label(c,'Target · frozen',785,227,17,orange,true,'center');
 const u=beat(s,5,12),target=[.3,.79,.52,.91,.24,.63,.4,.83],initial=[.85,.2,.81,.24,.75,.2,.9,.19];
 for(let j=0;j<8;j++){let yy=253+j*22,v=target[j]+(initial[j]-target[j])*Math.pow(1-u,2)+.11*(1-u)*Math.sin(u*26+j);segment(c,550,yy,662,yy,'#e6edf1',2);segment(c,550,yy,550+112*v,yy,blue,7);segment(c,740,yy,852,yy,'#e6edf1',2);segment(c,740,yy,740+112*target[j],yy,orange,7);}
 const gap=1-u;c.save();c.globalAlpha=1-beat(s,17,.8);pill(c,'Feature mismatch',700,452,ink);c.restore();label(c,'Training makes noisy corrections, not a literal motion between states.',52,495,16,muted);
 }else{
 label(c,'Initial observation',65,165,16,ink,true);block(c,...a.model,'Predictor','causal history');featureRibbon(c,...a.z,130);label(c,'Policy → latent tokens',485,142,16,blue,true,'center');link(c,245,234,413,268,blue,beat(s,.2));link(c,485,195,485,238,blue,beat(s,.2));link(c,558,279,686,279,blue,beat(s,.8));featureRibbon(c,...a.pred,140);label(c,'Prediction',760,322,18,blue,true,'center');
 c.save();c.strokeStyle='#dbe4ec';c.setLineDash([3,7]);c.beginPath();c.moveTo(52,347);c.lineTo(908,347);c.stroke();c.restore();label(c,'Training target only',65,395,21,orange,true);label(c,'Future frames stay below this boundary.',65,432,17,muted);link(c,702,396,662,396,orange,beat(s,1.5));pill(c,'Compare only',610,396,orange);label(c,'At deployment: observation + instruction → tokens → action head.',52,491,18,ink,true);
 }
}
function relatedMotion(c){headerC(c,'Related work · Compare the target and the pipeline','Feature prediction alone is not the whole distinction','LAPA §3 · UniVLA §III · VLA-JEPA §3 · These are pipeline differences, not proof of failure');
const rows=[['LAPA','Pixels','Frame pair → discrete code → policy → robot adaptation',gray],['UniVLA','DINO features','Frame pair + language → discrete code → policy → action decoder',blue],['VLA-JEPA','V-JEPA 2 features','Current observation → continuous tokens → future-state prediction',orange]];
rows.forEach(([name,target,path,color],i)=>{let t=beat(sceneSeconds(),.12+i*.08,.65),yy=185+i*95;c.save();c.translate(18*(1-t),0);label(c,name,52,yy+17,24,color,true);pill(c,target,290,yy+12,color);label(c,path,370,yy+17,13,ink);segment(c,52,yy+53,908,yy+53,'#e8eef2',1);c.restore();});
label(c,'UniVLA already uses feature reconstruction.',52,490,19,ink,true);label(c,'VLA-JEPA directly trains the policy tokens through future-state alignment.',52,463,19,blue,true);
}
