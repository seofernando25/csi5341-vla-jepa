#!/usr/bin/env python3
"""In-memory, single-controller LAN relay. Python standard library only."""
import argparse
import hmac
import json
import math
import secrets
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
# Use the same chapter/cue definitions as the browser; no separately maintained map.
import re
CUES = {}
for scene, body in re.findall(r'(\d+):\s*\[(.*?)\]', (ROOT / 'src/timing/cues.js').read_text(), re.S):
    CUES[int(scene)] = len(re.findall(r'\bcue\(', body))


class Relay(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, token=None):
        super().__init__(address, Handler)
        self.token = token or secrets.token_urlsafe(24)
        self.condition = threading.Condition()
        self.settings = {'scrollSpeed': 10}
        self.settings_revision = 0
        self.state = None
        self.revision = 0
        self.sequence = -1
        self.controller = None
        self.last_seen = 0

    def snapshot(self):
        # Call under condition lock.
        return {'state': self.state, 'revision': self.revision,
                'settings': dict(self.settings), 'settingsRevision': self.settings_revision,
                'controllerOnline': time.monotonic() - self.last_seen < 7,
                'ageMs': max(0, (time.monotonic() - self.last_seen) * 1000) if self.state else 0}


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass  # Normal when a browser closes/reconnects.

    def log_message(self, *args):
        pass  # Controller pairing key must not appear in request logs.

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def reply(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/relay/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            try:
                while True:
                    with self.server.condition:
                        payload = self.server.snapshot()
                    self.wfile.write(('data: ' + json.dumps(payload) + '\n\n').encode())
                    self.wfile.flush()
                    with self.server.condition:
                        self.server.condition.wait(timeout=2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return
        if path in ('/', '/index.html'):
            metadata = '<meta name="presenter-relay" content="local">'
            if parse_qs(urlsplit(self.path).query).get('relay') == ['control']:
                metadata += f'<meta name="presenter-controller-key" content="{self.server.token}">'
            body = (ROOT / 'index.html').read_text().replace('<meta charset="UTF-8">', '<meta charset="UTF-8">' + metadata).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == '/relay/state':
            with self.server.condition:
                self.reply(200, self.server.snapshot())
            return
        if '..' in path or '%' in path or not (
            path in ('/', '/index.html', '/presenter.html', '/presenter.css', '/styles.css', '/data.json')
            or path.startswith(('/src/', '/assets/'))
        ):
            self.reply(404, {'error': 'Not a presentation resource'})
            return
        super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path == '/relay/settings':
            # Any LAN screen may change reading pace, never slide navigation.
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 128:
                    raise ValueError('Invalid body length')
                data = json.loads(self.rfile.read(length))
                value = data['scrollSpeed']
                if set(data) != {'scrollSpeed'} or type(value) is not int or not 4 <= value <= 48:
                    raise ValueError('Invalid scroll speed')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.close_connection = True
                self.reply(400, {'error': 'Scroll speed must be an integer from 4 to 48 px/s'})
                return
            with self.server.condition:
                if value != self.server.settings['scrollSpeed']:
                    self.server.settings['scrollSpeed'] = value
                    self.server.settings_revision += 1
                payload = self.server.snapshot()
                self.server.condition.notify_all()
            self.reply(200, payload)
            return
        if urlsplit(self.path).path != '/relay/state':
            self.reply(404, {'error': 'Unknown endpoint'})
            return
        if not hmac.compare_digest(self.headers.get('X-Presenter-Key', ''), self.server.token):
            self.close_connection = True
            self.reply(403, {'error': 'Only the paired slide computer can publish'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 2048:
                raise ValueError('Invalid body length')
            data = json.loads(self.rfile.read(length))
            sequence = data['sequence']
            if type(sequence) is not int or sequence < 0:
                raise ValueError('Invalid sequence')
            state = data['state']
            scene, cue = state['scene'], state['cue']
            if type(scene) is not int or type(cue) is not int or scene not in CUES or not 0 <= cue < CUES[scene]:
                raise ValueError('Invalid chapter or cue')
            if type(state['running']) is not bool or not isinstance(data['client'], str) or not 1 <= len(data['client']) <= 100:
                raise ValueError('Invalid client or motion state')
            clean = {k: state[k] for k in ('scene', 'cue', 'running')}
            # Optional animation snapshot; old chapter/cue clients remain compatible.
            if 'seconds' in state:
                for name, lo, hi in [('seconds', 0, 600), ('progress', 0, 1), ('remaining', 0, 15000), ('direction', -1, 1)]:
                    value = state[name]
                    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                        raise ValueError('Invalid animation snapshot')
                    clean[name] = value
                if clean['direction'] not in (-1, 0, 1):
                    raise ValueError('Invalid direction')
            state = clean
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.close_connection = True
            self.reply(400, {'error': 'Invalid presentation state'})
            return
        with self.server.condition:
            if self.server.controller != data['client'] and time.monotonic() - self.server.last_seen < 7:
                self.reply(409, {'error': 'Another slide controller is connected'})
                return
            if self.server.controller == data['client'] and sequence <= self.server.sequence:
                self.reply(200, self.server.snapshot())
                return  # A timed-out request can arrive late after Wi-Fi recovers.
            self.server.sequence = sequence
            if state != self.server.state:
                self.server.revision += 1
            self.server.state = state
            self.server.controller = data['client']
            self.server.last_seen = time.monotonic()
            payload = self.server.snapshot()
            self.server.condition.notify_all()
        self.reply(200, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8766, type=int)
    parser.add_argument('--key', help=argparse.SUPPRESS)
    args = parser.parse_args()
    server = Relay((args.host, args.port), args.key)
    address = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        address.connect(('192.0.2.1', 80))  # Routing lookup only; sends no data.
        lan = address.getsockname()[0]
    except OSError:
        lan = '<slide-computer-LAN-IP>'
    finally:
        address.close()
    print(f'Slides (this computer): http://localhost:{args.port}/?relay=control', flush=True)
    print(f'Notes (second computer): http://{lan}:{args.port}/presenter.html', flush=True)
    local_name = socket.gethostname()
    if local_name.endswith('.local'):
        print(f'Notes (mDNS alternative): http://{local_name}:{args.port}/presenter.html', flush=True)
    print('Keep this terminal running. Ctrl+C stops the relay. State is not saved to disk.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
