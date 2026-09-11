"""Create a timed slide presentation with an explicitly synthetic English voice."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess

from persona_memory_ranker.io import read_json, write_json

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--milestone',choices=['M1','M2'],required=True);args=parser.parse_args()
milestone=args.milestone
ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
if not ffmpeg or not ffprobe: raise RuntimeError('FFmpeg and ffprobe must be installed')
data=read_json(root/f'artifacts/presentations/{milestone}-content.json')
build=root/f'artifacts/presentations/{milestone}'
durations=[]
for i in range(1,len(data['slides'])+1):
    audio=build/f'audio/slide-{i:02d}.wav'
    duration=float(subprocess.check_output([ffprobe,'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(audio)],text=True))
    durations.append(duration)
pause=1.5
target=data['duration_seconds']
tempo=sum(durations)/(target-pause*len(durations))
if not .65 <= tempo <= 1.5: raise ValueError(f'Narration needs substantive revision; required tempo {tempo:.3f}')
clips=[]
timeline=[]
elapsed=0
for i,raw_duration in enumerate(durations,1):
    duration=raw_duration/tempo+pause
    output=build/f'clip-{i:02d}.mp4'
    subprocess.run([ffmpeg,'-y','-loglevel','error','-loop','1','-i',str(build/f'slide-{i:02d}.png'),'-i',str(build/f'audio/slide-{i:02d}.wav'),
                    '-af',f'atempo={tempo:.9f},apad=pad_dur={pause}','-t',f'{duration:.9f}','-r','12','-c:v','libx264','-preset','fast','-tune','stillimage',
                    '-threads','2','-crf','24','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(output)],check=True)
    clips.append(output)
    timeline.append({'slide':i,'title':data['slides'][i-1]['title'],'start_seconds':elapsed,'duration_seconds':duration})
    elapsed+=duration
    print(f'Encoded {milestone} slide {i}/{len(durations)}',flush=True)
listing=build/'concat.txt'
listing.write_text(''.join("file '"+p.as_posix().replace("'","'\\''")+"'\n" for p in clips),encoding='utf-8')
out=root/f'deliverables/video/{milestone}-presentation.mp4';out.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-t',str(target),'-c','copy','-movflags','+faststart',str(out)],check=True)
actual=float(subprocess.check_output([ffprobe,'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(out)],text=True))
if abs(actual-target)>.25: raise ValueError(f'Video duration mismatch: {actual} vs {target}')
write_json(root/f'reports/{milestone}_video.json',{'path':out.relative_to(root).as_posix(),'duration_seconds':actual,'target_seconds':target,'voice':'Microsoft David Desktop (synthetic)','tempo_adjustment':tempo,'slides':timeline})
print(json.dumps({'video':str(out),'duration_seconds':actual,'synthetic_narration':True}))
