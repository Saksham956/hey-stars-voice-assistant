"""Summarize an opt-in local STARS test log. Never calls Gemini or loads models.

Supply --expected-trials to include missed wakes/no-response attempts in the
failure count. Unknown manual Windows/audible outcomes remain unverified.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def log_records(text):
    """Rejoin PowerShell 5 native-stderr wrapping without merging real records."""
    timestamp = re.compile(r'\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d+\s+(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s')
    record = None
    for line in text.splitlines():
        if timestamp.search(line):
            if record is not None:
                yield record
            record = line.strip()
        elif record is None:
            # Plain test fixtures and logs without timestamps remain line based.
            yield line
        elif line.strip() and not re.match(r'^\s*(?:At line:|\+|~|CategoryInfo\s*:|FullyQualifiedErrorId\s*:)', line):
            record += ' ' + line.strip()
    if record is not None:
        yield record


def parse_log(text):
    turns = {}
    def turn(generation):
        return turns.setdefault(int(generation), {'generation': int(generation)})
    active = None
    for line in log_records(text):
        wake = re.search(r'Listening for command \(generation (\d+)\)', line)
        if wake:
            active = int(wake[1]); turn(active)
        if 'YOU (local transcript):' in line and active is not None:
            turn(active)['transcript'] = line.split('YOU (local transcript):', 1)[1].strip()
        timing = re.search(r'Local timing generation=(\d+) recognition_wait_seconds=([\d.]+) match_seconds=([\d.]+) execution_seconds=([\d.]+) outcome=(\w+) intent=(\w+)', line)
        if timing:
            row = turn(timing[1])
            for key, value in zip(('recognition_wait_seconds','match_seconds','execution_seconds'), timing.group(2,3,4)):
                row[key] = float(value)
            row.update(outcome=timing[5], intent=timing[6])
        for pattern, field in (
            (r'Local action generation=(\d+) end-of-speech-to-result=([\d.]+)s', 'result_seconds'),
            (r'TTFA generation=(\d+) end-of-speech-to-first-PCM=([\d.]+)s', 'first_pcm_seconds')):
            match = re.search(pattern, line)
            if match: turn(match[1])[field] = float(match[2])
    return list(turns.values())


def summarize(rows, expected_trials=None):
    # Stop is deliberately silent. Unmatched/rejected replies are NOT successful
    # command latency samples even if they produced a fast spoken response.
    handled = [r for r in rows if r.get('outcome') == 'handled']
    success = [r for r in handled if r.get('intent') == 'stop' or 'first_pcm_seconds' in r]
    samples = sorted(r['first_pcm_seconds'] for r in success if 'first_pcm_seconds' in r and r.get('intent') != 'stop')
    observed = len(rows)
    expected = max(observed, expected_trials if expected_trials is not None else observed)
    result = {'observed_activations': observed, 'expected_trials': expected,
              'unobserved_attempts': expected-observed, 'handled_count':len(handled),
              'completed_with_pcm_or_silent_stop':len(success),
              'failure_or_missing_response_count':expected-len(success),
              'first_pcm_samples': len(samples),
              'manual_action_and_audibility_verification': 'pending; logs alone cannot verify Windows effects or audible playback',
              'recognition_boundary': 'final result wait after capture for streaming mode; full decode wait in batch mode',
              'first_pcm_boundary': 'last VAD speech frame callback arrival to output callback supplying first PCM; includes endpointing/backlog, excludes speech duration, cold startup and hardware playback latency',
              'percentile_method': 'nearest rank: sorted[ceil(0.95*n)-1]'}
    if samples:
        result.update(median_first_pcm_seconds=statistics.median(samples),
                      p95_first_pcm_seconds=samples[math.ceil(.95*len(samples))-1],
                      min_first_pcm_seconds=min(samples), max_first_pcm_seconds=max(samples))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--expected-trials', type=int)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'verification'/'M02-R'/'live_analysis')
    args = parser.parse_args()
    if args.expected_trials is not None and args.expected_trials < 0:
        parser.error('expected-trials must be nonnegative')
    raw = args.log.read_bytes()
    # Windows PowerShell 5 Tee-Object uses UTF-16; PowerShell 7 uses UTF-8.
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    rows = parse_log(raw.decode(encoding, errors='replace'))
    result = summarize(rows, args.expected_trials)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fields = ['generation','transcript','intent','outcome','recognition_wait_seconds',
              'match_seconds','execution_seconds','result_seconds','first_pcm_seconds']
    with (args.output/'trials.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
