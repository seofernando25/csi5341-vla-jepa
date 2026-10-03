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
        with self.publish(scene=9, cue=4): pass
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            snapshot = self.event(viewer)
            self.assertEqual(snapshot['state'], {'scene':9,'cue':4,'running':False})
            self.assertTrue(snapshot['controllerOnline'])
        with self.publish(scene=3, cue=3): pass
        with urlopen(self.url+'/relay/events', timeout=4) as viewer:
            self.assertEqual(self.event(viewer)['state']['scene'], 3)

    def test_delayed_updates_cannot_rewind_reconnected_controller(self):
        with self.publish(scene=9, cue=3, sequence=100): pass
        with self.publish(scene=3, cue=0, sequence=99): pass
        with urlopen(self.url+'/relay/state') as response:
            self.assertEqual(json.load(response)['state'], {'scene':9,'cue':3,'running':False})
        with self.publish(scene=9, cue=4, sequence=101): pass
        with urlopen(self.url+'/relay/state') as response:
            self.assertEqual(json.load(response)['state']['cue'], 4)

    def test_controller_route_pairing_and_viewer_animation_snapshot(self):
        with urlopen(self.url+'/') as response:
            page = response.read().decode()
            self.assertIn('name="presenter-relay"', page)
            self.assertNotIn('presenter-controller-key',page)
            self.assertNotIn('test-key',page)
        with urlopen(self.url+'/?relay=control') as response:
            self.assertIn('name="presenter-controller-key" content="test-key"',response.read().decode())
        state = dict(scene=3,cue=1,running=True,direction=-1,seconds=9.6,progress=0.25,remaining=600)
        body = json.dumps(dict(client='slides',sequence=1,state=state)).encode()
        with urlopen(Request(self.url+'/relay/state',data=body,headers={'X-Presenter-Key':'test-key'})) as response:
            snapshot=json.load(response)
            self.assertEqual(snapshot['state'],state)
            self.assertGreaterEqual(snapshot['ageMs'],0)
        state['remaining']=float('nan')
        body=json.dumps(dict(client='slides',sequence=2,state=state)).encode()
        with self.assertRaises(HTTPError) as e:
            urlopen(Request(self.url+'/relay/state',data=body,headers={'X-Presenter-Key':'test-key'}))
        self.assertEqual(e.exception.code,400)
        e.exception.close()

    def test_shared_speed_broadcast_reconnect_and_navigation_isolation(self):
        def speed(value):
            body = json.dumps(value).encode()
            return urlopen(Request(self.url+'/relay/settings', data=body), timeout=3)
        with self.publish(scene=9, cue=2): pass
        original = self.server.snapshot()
        with urlopen(self.url+'/relay/events', timeout=4) as a, urlopen(self.url+'/relay/events', timeout=4) as b:
            self.assertEqual(self.event(a)['settings']['scrollSpeed'], 10)
            self.event(b)
            with speed({'scrollSpeed': 30}) as r:
                self.assertEqual(json.load(r)['settingsRevision'], 1)
            self.assertEqual(self.event(a)['settings']['scrollSpeed'], 30)
            self.assertEqual(self.event(b)['settings']['scrollSpeed'], 30)
            with speed({'scrollSpeed': 8}): pass
            self.assertEqual(self.event(a)['settings']['scrollSpeed'], 8)
            self.assertEqual(self.event(b)['settings']['scrollSpeed'], 8)
        with urlopen(self.url+'/relay/events', timeout=4) as late:
            snapshot = self.event(late)
            self.assertEqual(snapshot['settings'], {'scrollSpeed': 8})
            self.assertEqual(snapshot['settingsRevision'], 2)
            self.assertEqual(snapshot['state'], original['state'])
            self.assertEqual(snapshot['revision'], original['revision'])
            self.assertTrue(snapshot['controllerOnline'])
        for invalid in [{'scrollSpeed': True}, {'scrollSpeed': 49}, {'scrollSpeed': 3}, {'scrollSpeed': 8.5}, {'scrollSpeed': 8, 'state': {}}, []]:
            with self.assertRaises(HTTPError) as e: speed(invalid)
            self.assertEqual(e.exception.code, 400)
            e.exception.close()
        self.assertEqual(self.server.settings['scrollSpeed'], 8)

    def test_read_only_validation_lease_and_offline(self):
        for kwargs, status in [({'key':'wrong'},403), ({'scene':1},400), ({'scene':4},400), ({'scene':16},400), ({'cue':99},400), ({'scene':True},400)]:
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
