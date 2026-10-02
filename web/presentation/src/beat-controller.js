/* Presenter controls use authored cue boundaries, not narration callbacks. */
const PRESENTATION_BEATS={0:[2.1,8,10,18],2:[6.1,10],3:[9.5,16.7,31.7,36.6,45.7,56,62],4:[1.4,7.7,20,30,36,40],5:[4,8.8,19],6:[1.1,11,19],7:[1.1,5,17.8,21],8:[1.2,2.4,33],9:[1,11,21,35,46,55],10:[2.8,4.8,10.5,16.3,27.8,34],11:[1.5,24.7,43],12:[2,10],13:[2,17],14:[2,16],15:[2,14],16:[2,24],17:[2,23],18:[2,5.8,15]};
const presenter={manual:true,stop:0,speed:1};
function cueTimes(i=index){return PRESENTATION_BEATS[i]||[motionLength(i)];}
function playTo(seconds){const span=Math.max(0,seconds-sceneSeconds());presenter.manual=true;presenter.stop=clamp(seconds/motionLength());presenter.speed=Math.max(1,span/1.6);animation=span>.001;lastPaint=performance.now();document.getElementById('animate').textContent='Replay beat';}
function resetPresenter(){presenter.manual=true;presenter.stop=0;p=0;animation=false;playTo(cueTimes()[0]);}
function advanceBeat(direction=1){audio.pause();continuous=false;const time=animation&&presenter.manual?presenter.stop*motionLength():sceneSeconds(),cues=cueTimes();if(direction>0){const next=cues.find(t=>t>time+.03);if(next!==undefined)playTo(next);else moveScene(1);}else{const at=cues.findIndex(t=>t>=time-.03),prior=Math.max(0,at-1);if(time<=cues[0]+.03)moveScene(-1);else{p=(prior?cues[prior-1]:0)/motionLength();playTo(cues[prior]);}}render();}
function replayBeat(){const time=sceneSeconds(),cues=cueTimes(),at=Math.max(0,cues.findIndex(t=>t>=time-.03));p=(at?cues[at-1]:0)/motionLength();playTo(cues[at]);}
