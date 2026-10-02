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
function beginTransform(next){motion.oldHeader=motion.header||'';motion.dataFrom=charts[index]&&charts[next]?charts[index].values:null;if(index>=4&&index<=8&&next>=4&&next<=8){motion.from=poseFor(index);motion.began=performance.now();}else motion.from=null;}
function methodPose(){let a=poseFor(index);if(!motion.from)return a;let t=ease((motion.exportTime ?? (performance.now()-motion.began))/motion.duration),r={};for(let key in a)r[key]=a[key].map((v,j)=>mix(motion.from[key][j],v,t));if(t===1)motion.from=null;return r;}
function photo(c,im,x,y,w,h){c.save();c.beginPath();c.roundRect(x,y,w,h,10);c.clip();c.fillStyle='#eef2f4';c.fillRect(x,y,w,h);if(im.complete&&im.naturalWidth){let k=Math.min(w/im.naturalWidth,h/im.naturalHeight);c.drawImage(im,x+(w-im.naturalWidth*k)/2,y+(h-im.naturalHeight*k)/2,im.naturalWidth*k,im.naturalHeight*k);}c.restore();}
function pill(c,text,x,y,color=blue){c.font='600 12px "Plus Jakarta Sans"';let w=c.measureText(text).width+24;c.fillStyle=color+'12';c.beginPath();c.roundRect(x-w/2,y-14,w,26,13);c.fill();label(c,text,x,y+3,12,color,true,'center');}
function featureRibbon(c,x,y,w=125,color=blue,amount=1){const vals=[.3,.79,.52,.91,.24,.63,.4,.83,.32,.58,.92,.45];for(let j=0;j<vals.length;j++){let h=8+vals[j]*28*amount;c.fillStyle=color;c.globalAlpha=.3+vals[j]*.7;c.beginPath();c.roundRect(x-w/2+j*w/12,y-h/2,w/12-3,h,2);c.fill();}c.globalAlpha=1;}
function block(c,x,y,labelText,sub,color=ink){c.fillStyle='#f1f6f8';c.beginPath();c.roundRect(x-70,y-40,140,80,14);c.fill();c.font='700 20px "Plus Jakarta Sans"';const titleSize=Math.min(20,120/c.measureText(labelText).width*20);label(c,labelText,x,y-3,titleSize,color,true,'center');c.font='500 10px "Plus Jakarta Sans"';const lines=[];for(const part of sub.split('|')){let row='';for(const word of part.split(' ')){const next=row?row+' '+word:word;if(row&&c.measureText(next).width>120){lines.push(row);row=word;}else row=next;}if(row)lines.push(row);}lines.forEach((line,j)=>label(c,line,x,y+15+j*13,10,muted,false,'center'));}
function link(c,x,y,x2,y2,color=blue,t=1,bend=0){const dx=x2-x,dy=y2-y;const pts=Math.abs(dx)>Math.abs(dy)?[[x,y],[x+dx*.38,y+bend],[x2-dx*.38,y2+bend],[x2,y2]]:[[x,y],[x+bend,y+dy*.38],[x2+bend,y2-dy*.38],[x2,y2]];curve(c,pts,color,2,t,true);}
function methodMotion(c){const s=sceneSeconds(),a=methodPose(),now=a.now,future=a.future;
const title={4:'Predict a future state — in feature space',5:'The tokens describe an intended transition',6:'The predictor outputs the next state’s features',7:'The target stays fixed. The prediction learns.',8:'The policy never receives its future target'}[index];
headerC(c,'Method · '+({4:'From video to supervision',5:'Policy representation',6:'World prediction',7:'Predictive alignment',8:'The information boundary'}[index]),title,'Paper §3.2');
photo(c,humanExamples[0],...now);photo(c,humanExamples[2],...future);
if(index===4){
 label(c,'Now',now[0],now[1]-18,17,ink,true);label(c,'Later',future[0],future[1]-18,17,ink,true);
 typedPill(c,'“Put the jar into the box”',480,175,blue,clamp((s-.5)/.7));
 link(c,370,260,590,260,gray,beat(s,7,.45),-25);emerge(c,s,7,()=>pill(c,'Observed video transition',480,229,muted));
 const predOn=beat(s,19,.45),targetOn=beat(s,29,.45);
 frameToFeatures(c,humanExamples[0],now,[350,405,170],beat(s,17,1.6));
 frameToFeatures(c,humanExamples[2],future,[730,405,170],beat(s,27,1.6),orange);
 c.save();c.globalAlpha=predOn;featureRibbon(c,350,405,170);label(c,'Predicted future features',350,449,18,blue,true,'center');c.restore();
 c.save();c.globalAlpha=targetOn;featureRibbon(c,730,405,170,orange);label(c,'Encoded future target',730,449,18,orange,true,'center');c.restore();
 curve(c,[[212,354],[212,385],[235,405],[260,405]],blue,2,beat(s,19,.45),true);link(c,748,354,730,378,orange,beat(s,29,.45));
 emerge(c,s,35,()=>{segment(c,450,405,462,405,gray,1.4);segment(c,612,405,623,405,gray,1.4);pill(c,'Compare',538,405,ink);});
 label(c,'Like anticipating “jar inside box” — without drawing the next frame.',52,489,18,ink);
 }else if(index===5){
 label(c,'Current observation',now[0],now[1]-16,16,muted,true);pill(c,'Put the jar into the box',198,384);
 block(c,...a.model,'VLM','Qwen3-VL-2B');link(c,335,261,380,264,blue,beat(s,3));curve(c,[[285,384],[334,384],[328,284],[380,284]],blue,2,beat(s,3),true);
 emerge(c,s,8,()=>featureRibbon(c,...a.z,170));link(c,523,264,628,264,blue,beat(s,7));stampText(c,'Latent action tokens',730,322,20,blue,true,clamp((s-7)/TYPE_MOTION.stampSeconds),'center');
 label(c,'An intended transition',730,356,16,muted,false,'center');label(c,'Supervision only',820,461,12,orange,false,'center');
 label(c,'These tokens condition both prediction and control.',52,468,23,ink,true);
 }else if(index===6){
 label(c,'Encoded state history',65,164,17,ink,true);featureRibbon(c,155,330,170,gray);pill(c,'Frozen V-JEPA 2 encoder',155,373,ink);
 block(c,...a.model,'Predictor','causal state history + tokens');featureRibbon(c,...a.z,130);label(c,'Latent action tokens',475,140,17,blue,true,'center');
 link(c,475,192,475,245,blue,beat(s,.2));link(c,255,330,403,290,gray,beat(s,.2));curve(c,[[548,285],[597,285],[587,365],[622,365]],blue,2,beat(s,9),true);
 label(c,'Future target only',814,301,12,orange,false,'center');emerge(c,s,10,()=>featureRibbon(c,...a.pred,155));stampText(c,'Predicted next-state embedding',710,414,19,blue,true,clamp((s-10)/TYPE_MOTION.stampSeconds),'center');
 label(c,'Predict features. No image decoder.',52,489,20,ink,true);
 }else if(index===7){
 // Two independent lanes: each encoder precedes its own feature vector.
 block(c,335,238,'Predictor','trainable');featureRibbon(c,335,156,105);
 link(c,335,178,335,195,blue,beat(s,.1));link(c,230,239,260,239,gray,beat(s,.1));
 block(c,335,382,'Video encoder','frozen',orange);link(c,230,382,260,382,orange,beat(s,.2));
 curve(c,[[408,238],[452,238],[453,253],[500,253]],blue,2,beat(s,.35),true);
 curve(c,[[408,382],[452,382],[453,397],[500,397]],orange,2,beat(s,.35),true);
 label(c,'Now',65,180,12,muted,true);label(c,'Observed future',65,451,12,orange,true);
 label(c,'Predicted features',505,213,16,blue,true);label(c,'Target features · frozen',505,357,16,orange,true);
 const u=beat(s,5,12),target=[.3,.79,.52,.91,.24,.63,.4,.83],initial=[.85,.2,.81,.24,.75,.2,.9,.19];
 for(let j=0;j<8;j++){const x=507+j*45,v=target[j]+(initial[j]-target[j])*Math.pow(1-u,2)+.11*(1-u)*Math.sin(u*26+j);c.fillStyle='#edf3f6';c.beginPath();c.roundRect(x,226,24,69,3);c.fill();c.roundRect(x,370,24,69,3);c.fill();c.fillStyle=blue;c.beginPath();c.roundRect(x,295-69*clamp(v),24,69*clamp(v),3);c.fill();c.fillStyle=orange;c.beginPath();c.roundRect(x,439-69*target[j],24,69*target[j],3);c.fill();}
 c.save();c.globalAlpha=1-beat(s,17,.8);pill(c,'Compare features',691,329,ink);c.restore();
 label(c,'Only the prediction is updated.',52,489,19,ink,true);

 }else{
 label(c,'Initial observation',65,165,16,ink,true);block(c,...a.model,'Predictor','causal history');featureRibbon(c,...a.z,130);label(c,'Policy → latent tokens',485,142,16,blue,true,'center');link(c,245,234,413,268,blue,beat(s,.2));link(c,485,195,485,238,blue,beat(s,.2));link(c,558,279,686,279,blue,beat(s,.8));featureRibbon(c,...a.pred,140);label(c,'Prediction',760,322,18,blue,true,'center');
 c.save();c.strokeStyle='#dbe4ec';c.setLineDash([3,7]);c.beginPath();c.moveTo(52,347);c.lineTo(908,347);c.stroke();c.restore();label(c,'Training target only',65,395,21,orange,true);link(c,702,396,662,396,orange,beat(s,1.5));pill(c,'Compare only',610,396,orange);label(c,'At deployment: observation + instruction → tokens → action head.',52,491,18,ink,true);
 }
}
function relatedMotion(c){const sec=sceneSeconds();headerC(c,'Related work · Why learn from video?','Same ambition. Different supervision.','OpenVLA · LAPA §3 · UniVLA §III · VLA-JEPA §3');
const rows=[['Robot-labelled VLA','Recorded controls','Image + instruction → executable actions',ink,0],['LAPA','Future pixels','Learn video codes, then adapt them to control',muted,16],['UniVLA','DINO features','Feature reconstruction already existed',blue,31],['VLA-JEPA','Future state features','Train policy tokens through predictive alignment',orange,45]];
rows.forEach(([name,target,detail,color,start],j)=>{const y=176+j*77;emerge(c,sec,start,()=>{stampText(c,name,52,y,21,color,true,clamp((sec-start)/TYPE_MOTION.stampSeconds));pill(c,target,369,y-4,color);kineticLine(c,detail,505,y,14,ink,false,1);segment(c,52,y+29,908,y+29,'#e5edf2',1);},8);});
const steps=[['Robot demonstrations are costly.',0],['Video provides changes, without motor-command labels.',9],['An embedding target alone does not define the pipeline.',36],['The key question: what does the policy see, and predict?',55]];let j=steps.findLastIndex(x=>sec>=x[1]);const [text,start]=steps[Math.max(0,j)];reelCardText(c,text,52,493,20,blue,true,clamp((sec-start)/TYPE_MOTION.reelSeconds),j>0?steps[j-1][0]:'');}
