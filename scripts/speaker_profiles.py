"""Optional, private cross-recording evidence for explicitly enabled voices."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

POLICY = 'cross-day-v1'


def recording_time(value, zone):
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is not None:
            return stamp.timestamp(), stamp.astimezone(zone).date()
    except (AttributeError, ValueError, OverflowError, OSError):
        pass
    return None


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


class CrossDayProfile:
    def __init__(self, document, model_sha256, enabled_aliases):
        self.identity = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        enabled = document.get('enabledAliases')
        offset = document.get('utcOffsetMinutes')
        if (document.get('schemaVersion') != 1 or document.get('modelSHA256') != model_sha256
                or not isinstance(enabled, list) or not enabled or not set(enabled) <= set(enabled_aliases)
                or type(offset) is not int or not -840 <= offset <= 840):
            raise ValueError('invalid_cross_day_speaker_profile')
        self.enabled = set(enabled)
        self.zone = timezone(timedelta(minutes=offset))
        self.dimension = None
        self.people = self.references(document.get('people'))
        self.distractors = self.samples(document.get('distractors'))
        if not self.enabled <= self.people.keys() or len(self.people) < 2 or not self.distractors:
            raise ValueError('incomplete_cross_day_speaker_profile')

    def references(self, people):
        if not isinstance(people, dict) or not all(isinstance(alias, str) and alias for alias in people):
            raise ValueError('invalid_cross_day_speaker_references')
        return {alias: self.samples(refs) for alias, refs in people.items()}

    def samples(self, rows):
        if not isinstance(rows, list) or not rows:
            raise ValueError('missing_cross_day_speaker_references')
        result = []
        for row in rows:
            asset = row.get('assetID') if isinstance(row, dict) else None
            stamp = recording_time(row.get('recordedAt'), self.zone) if asset else None
            vector = row.get('embedding') if stamp is not None else None
            if (not isinstance(asset, str) or not isinstance(vector, list) or not vector
                    or not all(type(x) in (int, float) and math.isfinite(x) for x in vector)
                    or abs(dot(vector, vector) - 1) > .01
                    or (self.dimension is not None and len(vector) != self.dimension)):
                raise ValueError('invalid_cross_day_speaker_reference')
            self.dimension = len(vector)
            result.append((asset, *stamp, vector))
        return result

    def suggest(self, vector, asset_id, recorded_at):
        stamp = recording_time(recorded_at, self.zone)
        if not stamp or not asset_id or len(vector) != self.dimension:
            return None
        time, _ = stamp

        def distinct_scores(samples):
            scores = defaultdict(lambda: -1.0)
            for asset, ts, day, ref in samples:
                if asset != asset_id and abs(ts - time) > 3600:
                    scores[day] = max(scores[day], dot(vector, ref))
            return sorted(scores.values(), reverse=True)[:2]

        ranks = []
        for alias, samples in self.people.items():
            scores = distinct_scores(samples)
            if scores:
                ranks.append((scores[1] if len(scores) == 2 else -1.0, scores[0], alias))
        if len(ranks) < 2:
            return None
        ranks.sort(reverse=True)
        score, best, alias = ranks[0]
        margin = score - ranks[1][0]
        other = max(rank[1] for rank in ranks[1:])
        playback = next(iter(distinct_scores(self.distractors)), -1.0)
        if (alias not in self.enabled or score < .72 or best < .745 or margin < .15
                or score < other + .04 or score < playback + .04):
            return None
        return {'alias': alias, 'status': 'automatic_candidate', 'similarity': round(min(score, 1), 6),
                'margin': round(min(margin, 2), 6), 'policy': POLICY, 'profileSHA256': self.identity}


def load(profile, references):
    path = Path(profile) / 'speaker-matching.json'
    if not path.exists():
        return None
    if path.is_symlink() or references is None:
        raise ValueError('invalid_cross_day_speaker_profile_source')
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError('invalid_cross_day_speaker_profile')
    return CrossDayProfile(document, references['modelSHA256'], references['people'])
