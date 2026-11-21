"""Optional local speaker suggestions using user-enrolled examples."""
import json
import math
from pathlib import Path
import re
from statistics import mean

from refine_audio import digest

MIN_SECONDS = 4
MIN_SIMILARITY = 0.72
MIN_MARGIN = 0.15


def read_references(profile):
    """Read the private registry. It is never part of an export."""
    path = Path(profile) / 'speaker-references.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('schemaVersion') != 1
            or not re.fullmatch(r'[a-f0-9]{64}', value.get('modelSHA256', ''))
            or not isinstance(value.get('people'), dict)):
        raise ValueError('invalid_speaker_reference_registry')
    for alias, references in value['people'].items():
        if (not re.fullmatch(r'[A-Za-z0-9_.-]{1,24}', alias) or not isinstance(references, list)
                or not references):
            raise ValueError('invalid_speaker_reference')
        for item in references:
            if (not isinstance(item, dict) or not isinstance(item.get('assetID'), str)
                    or not isinstance(item.get('speakerCluster'), str)
                    or not re.fullmatch(r'[a-f0-9]{64}', item.get('audioSHA256', ''))
                    or not all(type(item.get(key)) in (int, float) and math.isfinite(item[key])
                               for key in ('start', 'end')) or not 0 <= item['start'] < item['end']):
                raise ValueError('invalid_speaker_reference_source')
            vector = item.get('embedding')
            if (not isinstance(vector, list) or not vector or not all(
                    type(x) in (int, float) and math.isfinite(x) for x in vector)
                    or abs(sum(x*x for x in vector) - 1) > 0.01):
                raise ValueError('invalid_speaker_reference_vector')
    return value


def suggest(vector, people, exclude_asset=None):
    scores = []
    for alias, refs in people.items():
        used = [r['embedding'] for r in refs if r['assetID'] != exclude_asset]
        if not used:
            continue
        if any(len(ref) != len(vector) for ref in used):
            raise ValueError('speaker_reference_dimension_mismatch')
        score = mean(sum(x*y for x, y in zip(vector, ref)) for ref in used)
        scores.append((score, alias))
    scores.sort(reverse=True)
    if len(scores) < 2:
        return None
    score, alias = scores[0]
    margin = score - scores[1][0]
    if score < MIN_SIMILARITY or margin < MIN_MARGIN:
        return None
    return {'alias': alias, 'status': 'automatic_candidate',
            'similarity': round(score, 4), 'margin': round(margin, 4)}


class SpeakerMatcher:
    def __init__(self, model, references, cross_day=None, anonymous_only=False):
        import sherpa_onnx
        if digest(model) != references['modelSHA256']:
            raise ValueError('speaker_reference_model_changed')
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2, provider='cpu')
        if not config.validate():
            raise ValueError('invalid_speaker_embedding_model')
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.people = references['people']
        self.display_names = references.get('displayNames', {})
        self.cross_day = cross_day
        self.anonymous_only = anonymous_only

    def tag(self, segments, turns, samples, asset_id=None, recorded_at=None):
        import numpy as np
        from speaker_review import candidate_signals
        evidence = []
        named = {}
        for row in segments:
            start, end, speaker = row['start'], row['end'], row['speaker']
            if speaker == 'unknown' or end - start < MIN_SECONDS:
                continue
            if any(t['speaker'] != speaker and min(end, t['end']) - max(start, t['start']) > .12 for t in turns):
                continue
            stop = min(end, start + 12)
            stream = self.extractor.create_stream()
            stream.accept_waveform(16000, samples[int(start*16000):int(stop*16000)])
            stream.input_finished()
            if not self.extractor.is_ready(stream):
                continue
            vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0:
                continue
            embedding = (vector/norm).tolist()
            if self.cross_day:
                candidate = self.cross_day.suggest(embedding, asset_id, recorded_at)
            elif self.anonymous_only:
                candidate = None
            else:
                candidate = suggest(embedding, self.people, exclude_asset=asset_id)
            evidence.append({'start': start, 'end': stop, 'speaker': speaker, 'embedding': embedding,
                             'alias': candidate['alias'] if candidate else None})
            if candidate:
                name = self.display_names.get(candidate['alias']) if isinstance(self.display_names, dict) else None
                if isinstance(name, str) and 0 < len(name) <= 80 and name.isprintable():
                    candidate['displayName'] = name
                row['speakerMatch'] = candidate
                named.setdefault(speaker, set()).add(candidate['alias'])
        if self.cross_day:
            for row in segments:
                if len(named.get(row['speaker'], ())) > 1:
                    row.pop('speakerMatch', None)
        return candidate_signals(evidence)


def visible_speaker(row):
    match = row.get('speakerMatch')
    if isinstance(match, dict) and match.get('status') == 'automatic_candidate':
        alias = match.get('alias', '')
        if re.fullmatch(r'[A-Za-z0-9_.-]{1,24}', alias):
            name = match.get('displayName', alias)
            if isinstance(name, str) and 0 < len(name) <= 80 and name.isprintable():
                return f'{name} (automatic match, {row["speaker"]})'
            return f'{alias} (automatic match, {row["speaker"]})'
    return row['speaker']
