/* One action-space stage for training and inference. All coordinates are schematic. */
function flowAxes(c){c.fillStyle='#f6f9fb';c.beginPath();c.roundRect(52,149,584,313,18);c.fill();c.save();c.strokeStyle='#e4ecf2';c.lineWidth=.6;for(let x=85;x<630;x+=36){c.beginPath();c.moveTo(x,164);c.lineTo(x,444);c.stroke();}for(let y=166;y<450;y+=36){c.beginPath();c.moveTo(67,y);c.lineTo(621,y);c.stroke();}c.restore();label(c,'Schematic projection of an action chunk',70,178,11,muted);}
function flowMotion(c){let sec=sceneSeconds(),training=index===9;headerC(c,'Method · Flow matching · '+(training?'Learning the field':'Generating control'),training?'Teach the field with noise–demonstration pairs':'Now follow the field from a fresh noise sample','Paper Eqs. 7–8 · Schematic 2D projection · Mixing time is not physical robot time');flowAxes(c);
const pts=[[133,385],[295,420],[351,180],[569,231]],u=beat(sec,1,18);
if(training){const noise=[133,385],demo=[569,231],tau=.48+Math.sin(sec*1.7)*.1*(1-beat(sec,4.3,.6)),a=[mix(noise[0],demo[0],tau),mix(noise[1],demo[1],tau)];
 c.save();c.setLineDash([3,7]);segment(c,...noise,...demo,'#aab9c8',1.5);c.restore();
 for(let j=0;j<40;j++){let angle=j*2.399,r=7*Math.sqrt(j);dotC(c,noise[0]+Math.cos(angle)*r,noise[1]+Math.sin(angle)*r,1.8,'#a6b5c4');}
 dotC(c,...noise,6,gray);dotC(c,...demo,9,blue);dotC(c,...a,8,orange);label(c,'Noise',104,438,14,muted,true);label(c,'Demonstrated action',443,213,14,blue,true);label(c,'Mixed sample',a[0]-10,a[1]+36,15,orange,true,'center');
 const match=beat(sec,10,6),dx=93,dy=-33;segment(c,a[0],a[1]-7,a[0]+dx,a[1]-7+dy,orange,2.4,true);segment(c,a[0],a[1]+9,a[0]+mix(45,dx,match),a[1]+9+mix(40,dy,match),blue,2.4,true);
 pill(c,'Target velocity',440,295,orange);pill(c,'Predicted velocity',460,344,blue);
 }else{
 for(let j=0;j<38;j++){const angle=j*2.399,r=6*Math.sqrt(j),dx=Math.cos(angle)*r,dy=Math.sin(angle)*r,q=[[133+dx,385+dy],[295+dx*.7,420+dy*.7],[351+dx*.35,180+dy*.35],[569+dx*.2,231+dy*.2]];c.save();c.globalAlpha=.12;curve(c,q,blue,.8,1);c.restore();c.save();c.globalAlpha=.55;curve(c,q,blue,1,u);c.restore();dotC(c,...bez(u,q),1.8,blue);}
 curve(c,pts,blue,3,u);dotC(c,...bez(u,pts),7,blue);let tail=bez(Math.max(0,u-.025),pts);if(u>.02)segment(c,...tail,...bez(u,pts),blue,1,true);
 for(let k=1;k<=4;k++){const pt=bez(k/4,pts);c.save();c.globalAlpha=beat(u,k/4-.055,.055);dotC(c,...pt,3.5,blue);pill(c,String(k),pt[0],pt[1]+28,blue);c.restore();}
 label(c,'Fresh noise',89,437,15,muted,true);label(c,'Action chunk',465,209,16,blue,true);
 }
 featureRibbon(c,786,173,150);label(c,'Latent tokens',786,211,18,blue,true,'center');link(c,786,224,786,258,blue,beat(sec,.15));block(c,786,304,'Velocity field','conditioned on tokens + time');
 label(c,training?'Sample a pair':'Start from fresh noise',683,395,17,ink,true);label(c,training?'Choose a mixing time':'Re-evaluate the field',683,425,17,ink,true);label(c,training?'Match the velocity':'After each small update',683,455,17,ink,true);
 label(c,training?'Robot demonstrations teach which directions lead to valid control.':'The action head generates controls; it does not decode a predicted image.',52,492,18,ink,true);
}
