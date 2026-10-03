"""Real HTTP/SSE relay regressions; no third-party dependencies."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location('relay', Path(__file__).parents[1] / 'relay.py')
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)

class RelayTests(unittest.TestCase):
    def setUp(self):
        self.server = relay.Relay(('127.0.0.1', 0), 'test-key')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.sequence = 0
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def publish(self, scene=3, cue=0, running=False, client='slide-computer', key='test-key', sequence=None):
        self.sequence += 1
        body = json.dumps(dict(client=client, sequence=self.sequence if sequence is None else sequence, state=dict(scene=scene, cue=cue, running=running))).encode()
        return urlopen(Request(self.url+'/relay/state', data=body, headers={'X-Presenter-Key':key}), timeout=3)

    def event(self, connection):
        while True:
            line = connection.readline().decode()
            if line.startswith('data: '):
                return json.loads(line[6:])

    def test_initial_forward_back_replay_and_reconnect_snapshot(self):
        # Viewer before slides: no fabricated current chapter.
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            self.assertIsNone(self.event(viewer)['state'])
        with self.publish(): pass
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            self.assertEqual(self.event(viewer)['state']['cue'], 0)
            with self.publish(cue=1, running=True): pass
            self.assertEqual(self.event(viewer)['state'], {'scene':3,'cue':1,'running':True})
            with self.publish(cue=1): pass
            self.assertFalse(self.event(viewer)['state']['running'])
            with self.publish(cue=0): pass
            self.assertEqual(self.event(viewer)['state']['cue'], 0)
        # Notes was disconnected during further navigation: latest snapshot, no replay queue.
        with self.publish(scene=4, cue=4): pass
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            snapshot = self.event(viewer)
            self.assertEqual(snapshot['state'], {'scene':4,'cue':4,'running':False})
            self.assertTrue(snapshot['controllerOnline'])
        with self.publish(scene=3, cue=6): pass
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            self.assertEqual(self.event(viewer)['state']['scene'], 3)

    def test_delayed_updates_cannot_rewind_reconnected_controller(self):
        with self.publish(scene=4, cue=3, sequence=100): pass
        with self.publish(scene=3, cue=0, sequence=99): pass
        with urlopen(self.url+'/relay/state') as response:
            self.assertEqual(json.load(response)['state'], {'scene':4,'cue':3,'running':False})
        with self.publish(scene=4, cue=4, sequence=101): pass
        with urlopen(self.url+'/relay/state') as response:
            self.assertEqual(json.load(response)['state']['cue'], 4)

    def test_read_only_validation_lease_and_offline(self):
        for kwargs, status in [({'key':'wrong'},403), ({'scene':1},400), ({'cue':99},400), ({'scene':True},400)]:
            with self.assertRaises(HTTPError) as e: self.publish(**kwargs)
            self.assertEqual(e.exception.code, status)
            e.exception.close()
        with self.publish(): pass
        with self.assertRaises(HTTPError) as e: self.publish(client='second-controller')
        self.assertEqual(e.exception.code,409)
        e.exception.close()
        # Simulate controller lease expiry without making test wait seven seconds.
        with self.server.condition: self.server.last_seen -= 10
        with urlopen(self.url+'/relay/state') as response:
            state = json.load(response)
            self.assertFalse(state['controllerOnline'])
            self.assertIsNotNone(state['state'])
        with self.publish(scene=7, client='second-controller'): pass
        with urlopen(self.url+'/relay/state') as response:
            self.assertTrue(json.load(response)['controllerOnline'])
        for path in ['/tools/serve.py','/../AGENTS.md','/build/rendered','/%2e%2e/AGENTS.md']:
            with self.assertRaises(HTTPError) as e: urlopen(self.url+path)
            self.assertEqual(e.exception.code,404)
            e.exception.close()

if __name__ == '__main__': unittest.main()
