"""Private review leads for recurring unknown voices and conflicting matches."""
from collections import defaultdict
import hashlib
import json


REPEAT_SIMILARITY = 0.82
WITHIN_RECORDING_SIMILARITY = 0.72
LONG_SAMPLE_SECONDS = 8
MAX_VISIBLE_REVIEWS = 6


def similarity(left, right):
    if len(left) != len(right):
        return -1
    return sum(a * b for a, b in zip(left, right))


def candidate_signals(evidence):
    """Keep at most two examples per voice, without inferring a confirmed name."""
    by_speaker = defaultdict(list)
    for window in evidence:
        by_speaker[window['speaker']].append(window)
    signals = []
    for speaker, samples in by_speaker.items():
        samples.sort(key=lambda row: row['end'] - row['start'], reverse=True)
        named = {}
        for sample in samples:
            match = sample.get('alias')
            if match:
                named.setdefault(match, sample)
        if len(named) > 1:
            signals.append({'speaker': speaker, 'reason': 'conflicting_matches',
                            'examples': list(named.values())[:2]})
            continue
        if named:
            known = [sample for sample in samples if sample.get('alias')]
            different = next((sample for sample in samples if not sample.get('alias')
                              and max(similarity(sample['embedding'], anchor['embedding'])
                                      for anchor in known) < .50), None)
            if different:
                signals.append({'speaker': speaker, 'reason': 'possibly_mixed_voice',
                                'examples': [known[0], different]})
            continue
        first = samples[0]
        second = next((sample for sample in samples[1:]
                       if min(first['end'], sample['end']) <= max(first['start'], sample['start'])
                       and similarity(first['embedding'], sample['embedding']) >= WITHIN_RECORDING_SIMILARITY), None)
        signals.append({'speaker': speaker, 'reason': 'unknown_voice',
                        'examples': [first, second] if second else [first]})
    return signals


def approved_intervals(profile):
    """Previously reviewed source intervals keep their review disposition."""
    known = defaultdict(list)
    for file in (profile / 'speaker-review').glob('*.json'):
        if not file.name.startswith('auday-speaker-review-'):
            continue
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                continue
            for row in record.get('decisions', []):
                if row.get('choice') == '两段是不同人':
                    continue
                key = row['assetID'], row['audioSHA256']
                known[key].extend((w['start'], w['end']) for w in row['windows'])
        except (ValueError, OSError, TypeError, KeyError):
            continue
    return known


def review_summary(profile, assets):
    """Merge similar leads across recordings, then expose only playable excerpts."""
    approved = approved_intervals(profile)
    groups = []
    for audio, asset, signals in assets:
        key = (asset['id'], asset['sha256'])
        for signal in signals:
            examples = [{**example, 'assetID': asset['id'], 'audioSHA256': asset['sha256'],
                         'audioPath': str(audio), 'sourceStart': asset.get('start')}
                        for example in signal['examples']
                        if not any(min(right, example['end']) - max(left, example['start']) >=
                                   .8 * (example['end'] - example['start'])
                                   for left, right in approved.get(key, []))]
            if not examples:
                continue
            target = next((g for g in groups if signal['reason'] == g['reason'] == 'unknown_voice'
                           and all(x['assetID'] != y['assetID'] or
                                   min(x['end'], y['end']) <= max(x['start'], y['start'])
                                   for x in examples for y in g['examples'])
                           and min(similarity(x['embedding'], y['embedding'])
                                   for x in examples for y in g['examples']) >= REPEAT_SIMILARITY), None)
            if target:
                target['assets'].add(asset['id'])
                if target['examples'][0]['assetID'] == target['examples'][-1]['assetID']:
                    target['examples'] = [target['examples'][0], examples[0]]
                continue
            sample = examples[0]
            identity = json.dumps([asset['id'], asset['sha256'], sample['start'], sample['end']])
            groups.append({'id': hashlib.sha256(identity.encode()).hexdigest()[:20],
                           'reason': signal['reason'], 'assets': {asset['id']}, 'examples': examples})
    pending = [g for g in groups if g['reason'] != 'unknown_voice' or len(g['examples']) > 1
               or g['examples'][0]['end'] - g['examples'][0]['start'] >= LONG_SAMPLE_SECONDS]
    pending.sort(key=lambda g: (g['reason'] != 'unknown_voice', len(g['assets']),
                                str(g['examples'][0].get('sourceStart') or '')), reverse=True)
    visible = [{'id': g['id'], 'reason': g['reason'], 'recordings': len(g['assets']),
                'clips': [{k: v for k, v in sample.items() if k != 'embedding'} for sample in g['examples']]}
               for g in pending[:MAX_VISIBLE_REVIEWS]]
    return {'pending': len(pending), 'waitingForClearerAudio': len(groups) - len(pending), 'items': visible}
