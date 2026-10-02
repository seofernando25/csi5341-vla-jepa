"""Assemble locally rendered scenes using the same content and narration assets."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'build';R=BUILD/'rendered';BUILD.mkdir(exist_ok=True)
data=json.loads((ROOT/'data.json').read_text());main=[(i,s) for i,s in enumerate(data) if not s.get('hidden') and not s.get('backup')]
for i,s in main:
 stem=f'video-{i+1:02}';raw=R/(stem+'.h264')
 if not raw.exists():raise SystemExit('Render all chapters in the browser authoring view first.')
 subprocess.run(['ffmpeg','-v','error','-y','-r','60','-i',str(raw),'-i',str(ROOT/'assets'/s['audioAsset']),'-t',str(s['budget']),'-vf','setpts=N/(60*TB)','-r','60','-c:v','libx264','-preset','veryfast','-crf','18','-c:a','aac','-b:a','160k',str(R/(stem+'.mp4'))],check=True)
concat=R/'concat.txt';concat.write_text(''.join(f"file 'video-{i+1:02}.mp4'\n" for i,_ in main))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(BUILD/'VLA-JEPA-presentation.mp4')],check=True)
print('Saved build/VLA-JEPA-presentation.mp4')
