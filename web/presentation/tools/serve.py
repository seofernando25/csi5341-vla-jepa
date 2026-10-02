"""Serve the source presentation and accept explicit local authoring exports."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build'/'rendered';OUT.mkdir(parents=True,exist_ok=True)
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def do_POST(self):
  name=self.path.rsplit('/',1)[-1]
  if not self.path.startswith('/export/') or not re.fullmatch(r'(video-\d\d\.h264|slide-\d\d\.png|audit-\d\d-\d+\.png)',name):self.send_error(400);return
  size=int(self.headers.get('Content-Length','0'))
  if size<=0 or size>500_000_000:self.send_error(413);return
  (OUT/name).write_bytes(self.rfile.read(size));self.send_response(200);self.end_headers();self.wfile.write(b'ok')
 def log_message(self,*a):pass
if __name__=='__main__':
 print('Preview: http://localhost:8765/   Authoring: http://localhost:8765/?authoring=1',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
