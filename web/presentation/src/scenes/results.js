/* Paired measurements move together. No delayed second comparator. */
function benchmarkMotion(c){const a=charts[index],sec=sceneSeconds(),t=beat(sec,.08,.9),positive=index!==15,colour=positive?blue:orange;
headerC(c,index<14?'Evidence · Benchmarks':'Evidence · Human-video ablation',a.title,a.foot);
const x=270,w=510;label(c,'Success rate',52,169,14,muted);for(let n=0;n<=100;n+=25){const xx=x+n*w/100;segment(c,xx,188,xx,397,'#e6edf2',1);label(c,String(n),xx,421,12,muted,false,'center');}
a.labels.forEach((name,j)=>{const y=225+j*110,v=mix(motion.dataFrom?.[j]??0,a.values[j],t);c.fillStyle=j?colour:gray;c.beginPath();c.roundRect(x,y,v*w/100,42,6);c.fill();label(c,name,52,y+27,19,ink,true);label(c,v.toFixed(1)+'%',Math.min(908,x+v*w/100+16),y+28,23,j?colour:muted,true);});
const u=beat(sec,1,.4);c.save();c.translate(0,10*(1-u));c.globalAlpha=u;label(c,a.claim,52,477,27,colour,true);c.restore();label(c,index===12?'A near tie on standard tasks does not establish a meaningful advantage.':index===13?'The stronger claim is resilience under changed conditions.':index===14?'Compare the same method, with and without the extra video data.':'The benefit depends on the benchmark; human video is not uniformly helpful.',52,500,15,muted);
}
