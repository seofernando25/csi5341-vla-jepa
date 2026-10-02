"""Assemble locally rendered scenes using the same content and narration assets."""
from pathlib import Path
import subprocess,json,shutil
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'build';R=BUILD/'rendered';BUILD.mkdir(exist_ok=True)
data=json.loads((ROOT/'data.json').read_text());main=[(i,s) for i,s in enumerate(data) if not s.get('hidden') and not s.get('backup')]
total=sum(s['budget'] for _,s in main)
raw_movie=R/'main.h264'
# Each authoring stream starts on a keyframe and includes its Annex-B headers.
# Joining raw frames avoids MP4 chapter time-base rounding and an extra encode.
with raw_movie.open('wb') as out:
 for i,s in main:
  raw=R/f'video-{i+1:02}.h264'
  if not raw.exists():raise SystemExit('Render all chapters in the browser authoring view first.')
  with raw.open('rb') as source:shutil.copyfileobj(source,out)
for i,s in main:
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(ROOT/'assets'/s['audioAsset']),'-af',f'apad,atrim=duration={s["budget"]}','-ar','24000','-ac','1',str(R/f'audio-{i+1:02}.wav')],check=True)
audio_concat=R/'audio-concat.txt';audio_concat.write_text(''.join(f"file 'audio-{i+1:02}.wav'\n" for i,_ in main))
narration=BUILD/'narration.wav'
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(audio_concat),str(narration)],check=True)
# Assign every frame one global 60 fps timestamp and encode narration once.
subprocess.run(['ffmpeg','-v','error','-y','-r','60','-i',str(raw_movie),'-i',str(narration),'-map','0:v','-map','1:a','-vf','setpts=N/(60*TB)','-r','60','-c:v','libx264','-preset','veryfast','-crf','18','-threads','3','-c:a','aac','-b:a','160k','-t',str(total),'-movflags','+faststart',str(BUILD/'VLA-JEPA-presentation.mp4')],check=True)
print('Saved build/VLA-JEPA-presentation.mp4')
