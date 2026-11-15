"""Bounded offline suite with progress, thread dumps, and machine-readable results.

Default: real Windows asyncio, unchanged tests, no audio/models/cloud/browser.
--socket-timeout is diagnostic ONLY: bound stdlib socket creation in restricted
environments. Such errors remain errors; this option cannot turn them into passes.
"""
import argparse
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parent
MODULES = ('test_hey_stars', 'test_local_commands', 'test_youtube_commands',
           'test_m02_reliability')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def worker(names, directory, socket_timeout=None):
    faulthandler.dump_traceback_later(15, repeat=True)
    if socket_timeout is not None:
        import socket
        socket.setdefaulttimeout(socket_timeout)
    (ROOT / '.tmp').mkdir(exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    state = {'discovered': suite.countTestCases(), 'started': 0,
             'active_test': None, 'outcomes': [], 'complete': False,
             'socket_timeout_diagnostic': socket_timeout}
    progress = directory / 'progress.json'
    write_json(progress, state)

    class TrackingResult(unittest.TextTestResult):
        def startTest(self, test):
            state['started'] += 1
            state['active_test'] = test.id()
            write_json(progress, state)
            super().startTest(test)

        def stopTest(self, test):
            failed = any(t is test or getattr(t, 'test_case', None) is test
                         for t, _ in self.failures)
            errored = any(t is test or getattr(t, 'test_case', None) is test
                          for t, _ in self.errors)
            skipped = next((reason for t, reason in self.skipped if t is test), None)
            outcome = ('error' if errored else 'failed' if failed else
                       'skipped' if skipped is not None else 'passed')
            state['outcomes'].append({'test': test.id(), 'outcome': outcome,
                                      'skip_reason': skipped})
            state['active_test'] = None
            write_json(progress, state)
            super().stopTest(test)

    try:
        result = unittest.TextTestRunner(verbosity=2, resultclass=TrackingResult).run(suite)
        state['complete'] = True
        state['remaining_threads'] = [t.name for t in threading.enumerate()
                                      if t is not threading.main_thread()]
        write_json(progress, state)
        return 0 if result.wasSuccessful() else 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def run_bounded(names, timeout, directory, socket_timeout=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
               '--output', str(directory)]
    if socket_timeout is not None:
        command += ['--socket-timeout', str(socket_timeout)]
    command += list(names)
    started = time.monotonic()
    timed_out = False
    with (directory / 'tests.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    progress = directory / 'progress.json'
    state = json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {}
    outcomes = state.get('outcomes', [])
    counts = {name: sum(o['outcome'] == name for o in outcomes)
              for name in ('passed', 'failed', 'error', 'skipped')}
    summary = dict(state, **counts, timed_out=int(timed_out),
                   not_run=max(0, state.get('discovered', 0)-state.get('started', 0)),
                   process_exit_code=process.returncode,
                   elapsed_seconds=time.monotonic()-started, timeout_seconds=timeout)
    write_json(directory / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modules', nargs='*', default=list(MODULES))
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--socket-timeout', type=float)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0 or (args.socket_timeout is not None and args.socket_timeout <= 0):
        parser.error('Timeouts must be positive.')
    directory = args.output or ROOT / 'verification' / 'M02-R' / (
        'tests_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args.modules, directory, args.socket_timeout)
    summary = run_bounded(args.modules, args.timeout, directory, args.socket_timeout)
    print(json.dumps({k: v for k, v in summary.items() if k != 'outcomes'}, indent=2))
    print('Detailed evidence:', directory)
    return 0 if summary.get('complete') and not summary['timed_out'] and summary['process_exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
