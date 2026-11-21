"""Source-bound historical speaker labels in the local Auday database."""
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from qwen_audio import digest
from speaker_references import MIN_MARGIN, MIN_SECONDS, MIN_SIMILARITY, read_references


SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_speaker_labels (
    asset_id TEXT NOT NULL, audio_sha256 TEXT NOT NULL, transcript_sha256 TEXT NOT NULL,
    recorded_at TEXT NOT NULL, start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
    speaker_cluster TEXT NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL
        CHECK(status IN ('user_confirmed','automatic_match')),
    segment_index INTEGER, similarity REAL, margin REAL, transcript_text TEXT, evidence TEXT NOT NULL,
    PRIMARY KEY(asset_id,audio_sha256,transcript_sha256,start_seconds,end_seconds,status,speaker_cluster)
)
"""


def current_transcripts(profile, connection):
    """Select the newest complete derivative for each current phone asset."""
    assets = {key: json.loads(body) for key, body in connection.execute('SELECT id,body FROM assets')}
    newest = {}
    for path in sorted((profile / 'derivatives/qwen3-asr').glob('*/*/result.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('completed') is not True:
            continue
        asset = item['source']['asset']
        current = assets.get(asset['id'])
        if current is None or (current.get('sha256') and current['sha256'] != item['source']['audioSHA256']):
            continue
        if digest(path.with_name('transcript.md')) != item['transcriptSHA256']:
            raise ValueError('invalid_local_transcript')
        prior = newest.get(asset['id'])
        if prior is None or item['createdAt'] > prior[1]['createdAt']:
            newest[asset['id']] = asset, item
    return newest


def saved_labels(connection):
    labels = defaultdict(list)
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audio_speaker_labels'"
    ).fetchone() is None:
        return labels
    query = (
        'SELECT asset_id,audio_sha256,transcript_sha256,start_seconds,end_seconds,'
        'speaker_cluster,label,status,segment_index FROM audio_speaker_labels'
    )
    for aid, media, transcript, start, end, cluster, name, status, segment in connection.execute(query):
        labels[(aid, media)].append({'transcript': transcript, 'start': start, 'end': end,
                                     'cluster': cluster, 'name': name, 'status': status, 'segment': segment})
    return labels


def source_rows(bundle, newest, registry):
    reference_id = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
    evidence = bundle.get('sourceScanSHA256', '')
    if (bundle.get('schemaVersion') != 1 or bundle.get('referenceSnapshot') != reference_id
            or not re.fullmatch(r'[a-f0-9]{64}', evidence)):
        raise ValueError('historical_speaker_labels_snapshot_changed')
    names = set(registry['people']) | set(registry.get('displayNames', {}).values())
    rows = []
    visited = set()
    for record in bundle['recordings']:
        aid = record['assetID']
        if aid in visited or aid not in newest:
            raise ValueError('invalid_historical_speaker_label_source')
        visited.add(aid)
        asset, item = newest[aid]
        media, transcript = record['audioSHA256'], record['transcriptSHA256']
        if (item['source']['audioSHA256'] != media or item['transcriptSHA256'] != transcript
                or asset['start'] != record['start']):
            raise ValueError('historical_speaker_label_source_changed')
        prefix = aid, media, transcript, asset['start']
        for status, entries in [('user_confirmed', record['confirmed']), ('automatic_match', record['automatic'])]:
            for entry in entries:
                start, end, cluster, name = (entry[k] for k in ('start', 'end', 'speakerCluster', 'label'))
                if (entry['status'] != status or not isinstance(name, str) or not name.isprintable()
                        or len(name) > 120 or not isinstance(cluster, str)
                        or not (0 <= start < end <= item['duration'] + .1)):
                    raise ValueError('invalid_historical_speaker_label')
                values = (None, None, None, None)
                if status == 'user_confirmed':
                    if entry['assetID'] != aid or entry['audioSHA256'] != media:
                        raise ValueError('invalid_historical_speaker_label_source')
                else:
                    index, similarity, margin = (entry[k] for k in ('segmentIndex', 'similarity', 'margin'))
                    if (type(index) is not int or not 0 <= index < len(item['segments'])
                            or name not in names or end - start < MIN_SECONDS
                            or not math.isfinite(similarity) or not MIN_SIMILARITY <= similarity <= 1
                            or not math.isfinite(margin) or not MIN_MARGIN <= margin <= 2):
                        raise ValueError('invalid_automatic_historical_speaker_label')
                    row = item['segments'][index]
                    if (row['start'], row['end'], row['speaker']) != (start, end, cluster):
                        raise ValueError('invalid_historical_speaker_label_interval')
                    values = index, similarity, margin, row['text']
                rows.append((*prefix, start, end, cluster, name, status, *values, evidence))
    return rows


def apply_bundle(profile, path):
    profile = Path(profile).resolve()
    bundle = json.loads(path.read_text(encoding='utf-8'))
    registry = read_references(profile)
    if not registry:
        raise ValueError('speaker_references_unavailable')
    with closing(sqlite3.connect((profile / 'library.sqlite').as_uri() + '?mode=rw', uri=True)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        rows = source_rows(bundle, current_transcripts(profile, db), registry)
        db.execute(SCHEMA)
        db.execute('CREATE INDEX IF NOT EXISTS audio_speaker_labels_by_name ON audio_speaker_labels(label,recorded_at)')
        db.executemany(
            "DELETE FROM audio_speaker_labels WHERE asset_id=? AND audio_sha256=? "
            "AND transcript_sha256=? AND status='automatic_match'",
            sorted({row[:3] for row in rows}),
        )
        db.executemany(
            """
            INSERT INTO audio_speaker_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT DO UPDATE SET label=excluded.label, segment_index=excluded.segment_index,
            similarity=excluded.similarity, margin=excluded.margin,
            transcript_text=excluded.transcript_text, evidence=excluded.evidence
            """, rows,
        )
        counts = dict(db.execute(
            'SELECT status,count(*) FROM audio_speaker_labels WHERE evidence=? GROUP BY status',
            (bundle['sourceScanSHA256'],),
        ))
        if sum(counts.values()) != len(rows):
            raise ValueError('historical_speaker_label_count_mismatch')
    return {'recordings': len({r[0] for r in rows}),
            'automaticMatches': counts.get('automatic_match', 0),
            'confirmedIntervals': counts.get('user_confirmed', 0)}
