/* Deterministic motion typography. Authored beats, not audio callbacks, drive it.
   Headings retain their existing treatment; these accents teach a changed idea. */
const TYPE_MOTION={reelSeconds:.48,stampSeconds:.46,wordDelay:.027};
function typeMotionProgress(t){return motion.exportTime===null&&matchMedia('(prefers-reduced-motion: reduce)').matches?1:clamp(t);}
function typeSpring(t){const u=clamp(t);const v=u-1;return 1+2.15*v*v*v+1.15*v*v;}
function stampText(c,text,x,y,size,color=ink,bold=true,t=1,align='left'){
 t=typeMotionProgress(t);if(t<=0)return;if(t>=1){label(c,text,x,y,size,color,bold,align);return;}
 c.save();const u=typeSpring(t),pin=align==='center'?0:align==='right'?-1:1;
 c.font=`${bold?700:500} ${size}px "Plus Jakarta Sans"`;const w=c.measureText(text).width;
 const anchor=x+pin*w/2;c.translate(anchor,y-size*.36);
 c.scale(1+.09*(1-u),Math.max(.035,u));c.translate(-anchor,-y+size*.36);
 label(c,text,x,y+7*(1-u),size,color,bold,align);c.restore();
}
function typewriterText(c,text,x,y,size,color=ink,bold=false,t=1,align='left'){
 t=typeMotionProgress(t);if(t>=1){label(c,text,x,y,size,color,bold,align);return;}
 c.save();c.font=`${bold?700:500} ${size}px "Plus Jakarta Sans"`;const full=c.measureText(text).width;
 const left=x-(align==='center'?full/2:align==='right'?full:0),n=Math.floor(text.length*t);
 const shown=text.slice(0,n);label(c,shown,left,y,size,color,bold);
 if(t>0){const xx=left+c.measureText(shown).width+3;segment(c,xx,y-size*.8,xx,y+2,color,1.5);}
 c.restore();
}
function reelCardText(c,text,x,y,size,color=ink,bold=true,t=1,oldText=''){
 t=typeMotionProgress(t);if(t>=1){label(c,text,x,y,size,color,bold);return;}
 c.save();c.font=`${bold?700:500} ${size}px "Plus Jakarta Sans"`;
 const old=oldText.split(' '),fresh=text.split(' ');let common=0;
 while(common<old.length&&common<fresh.length&&old[common]===fresh[common])common++;
 const prefix=fresh.slice(0,common).join(' '),prefixWidth=prefix?c.measureText(prefix+' ').width:0;
 if(prefix)label(c,prefix,x,y,size,color,bold);
 const left=x+prefixWidth,w=Math.min(908-left,Math.max(c.measureText(text.slice(prefix.length).trimStart()).width,c.measureText(old.slice(common).join(' ')).width)+14);
 c.beginPath();c.rect(left-2,y-size-7,w+4,size+17);c.clip();
 // A compressed card opens behind the rolling words and disappears on settling.
 const pulse=Math.sin(Math.PI*t),height=(size+13)*(.28+.72*typeSpring(t));
 c.save();c.globalAlpha*=pulse;c.fillStyle=color==='white'?'#ffffff10':color+'0b';c.beginPath();c.roundRect(left-3,y-size*.4-height/2,w,height,7);c.fill();c.restore();
 const draw=(words,enter)=>{let xx=left;words.forEach((word,j)=>{
  const delay=Math.min(.22,j*TYPE_MOTION.wordDelay),u=clamp((t-delay)/(1-delay));
  const spring=typeSpring(u),offset=enter?(1-spring)*(size+14):-Math.pow(u,.75)*(size+14);
  const sy=enter?Math.max(.08,1-.72*Math.pow(1-u,2)):Math.max(.08,1-.55*u);
  const ww=c.measureText(word+' ').width;c.save();c.translate(xx,y+offset-size*.32);c.scale(1,sy);
  label(c,word,0,size*.32,size,color,bold);c.restore();xx+=ww;
 });};
 if(oldText)draw(old.slice(common),false);draw(fresh.slice(common),true);c.restore();
}

function typedPill(c,text,x,y,color,t){c.save();c.font='600 12px "Plus Jakarta Sans"';const w=c.measureText(text).width+24;c.fillStyle=color+'12';c.beginPath();c.roundRect(x-w/2,y-14,w,26,13);c.fill();typewriterText(c,text,x,y+3,12,color,true,t,'center');c.restore();}
