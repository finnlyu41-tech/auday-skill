import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import context_mirror
import speaker_labels
from speaker_profiles import CrossDayProfile
from speaker_references import suggest
from speaker_review import candidate_signals, review_summary


class SpeakerSuggestions(unittest.TestCase):
    def setUp(self):
        self.references = {'A': [{'assetID': 'sample-a', 'embedding': [1.0, 0.0, 0.0]}],
                           'B': [{'assetID': 'sample-b', 'embedding': [0.0, 1.0, 0.0]}]}

    def test_accepts_clear_match_and_marks_it_as_candidate(self):
        match = suggest([1.0, 0.0, 0.0], self.references)
        self.assertEqual((match['alias'], match['status']), ('A', 'automatic_candidate'))

    def test_rejects_ambiguous_weak_and_self_reference_matches(self):
        self.assertIsNone(suggest([0.73, 0.68, 0.0], self.references))
        self.assertIsNone(suggest([0.0, 0.0, 1.0], self.references))
        self.assertIsNone(suggest([1.0, 0.0, 0.0], self.references, exclude_asset='sample-a'))


class ValidatedSyncLabels(unittest.TestCase):
    @staticmethod
    def profile():
        def sample(asset, day, vector):
            return {'assetID': asset, 'recordedAt': f'2026-01-{day:02}T12:00:00Z', 'embedding': vector}

        model = 'a' * 64
        return CrossDayProfile({'schemaVersion': 1, 'modelSHA256': model, 'utcOffsetMinutes': 480,
            'enabledAliases': ['A'], 'people': {
                'A': [sample('a', 1, [1, 0, 0]), sample('b', 2, [1, 0, 0])],
                'B': [sample('c', 1, [0, 1, 0]), sample('d', 2, [0, 1, 0])]},
            'distractors': [sample('media', 1, [0, 0, 1])]}, model, ['A', 'B'])

    def test_new_source_needs_two_dates_and_an_enabled_name(self):
        profile = self.profile()
        date = '2026-01-03T12:00:00Z'
        match = profile.suggest([1, 0, 0], 'new', date)
        self.assertEqual((match['alias'], match['profileSHA256']), ('A', profile.identity))
        self.assertIsNone(profile.suggest([1, 0, 0], 'a', date))
        self.assertIsNone(profile.suggest([1, 0, 0], 'new', '2026-01-02T12:30:00Z'))
        self.assertIsNone(profile.suggest([0, 1, 0], 'new', date))
        self.assertIsNone(profile.suggest([0, 0, 1], 'new', date))

    def test_completed_derivative_retries_without_replacing_manual_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            policy = self.profile()
            date, media = '2026-01-03T12:00:00Z', 'e' * 64
            match = policy.suggest([1, 0, 0], 'new', date)
            asset = {'id': 'new', 'start': date, 'sha256': media}
            run = profile / 'derivatives/qwen3-asr/key/run'
            run.mkdir(parents=True)
            transcript = run / 'transcript.md'
            transcript.write_text('local derivative')
            path = run / 'result.json'
            item = {'completed': True, 'source': {'asset': asset, 'audioSHA256': media},
                    'createdAt': date, 'duration': 16,
                    'transcriptSHA256': hashlib.sha256(transcript.read_bytes()).hexdigest(),
                    'segments': [{'start': i, 'end': i+6, 'speaker': 's1', 'text': 'sample',
                                  'speakerMatch': match} for i in (0, 8)]}
            path.write_text(json.dumps(item))
            registry = {'people': {'A': [], 'B': []}}
            with sqlite3.connect(profile / 'library.sqlite') as db:
                db.execute('CREATE TABLE assets(id TEXT, body TEXT)')
                db.execute('INSERT INTO assets VALUES (?,?)', ('new', json.dumps(asset)))
                db.execute(speaker_labels.SCHEMA)
                db.execute('INSERT INTO audio_speaker_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           ('new', media, item['transcriptSHA256'], date, 1, 4, 's1', 'B', 'user_confirmed',
                            None, None, None, None, 'evidence'))

            def phone_export(_, stage, language):
                stage.mkdir()
                (stage / 'manifest.json').touch()
                return {'recordCount': 1, 'syncedAt': date}

            with patch.object(context_mirror, 'export_library', side_effect=phone_export):
                before, after = profile / 'before', profile / 'after'
                before.mkdir()
                after.mkdir()
                self.assertEqual(context_mirror.document_set(profile, before)['archiveAutomaticMatches'], 0)
                self.assertEqual(speaker_labels.save_sync_matches(profile, [(path, item)], registry, policy),
                                 {'inserted': 1, 'existingOrOverlapping': 1})
                self.assertEqual(speaker_labels.save_sync_matches(profile, [(path, item)], registry, policy),
                                 {'inserted': 0, 'existingOrOverlapping': 2})
                report = context_mirror.document_set(profile, after)
                self.assertEqual((report['archiveAutomaticMatches'], report['archiveConfirmedIntervals']), (1, 1))


class TextMirror(unittest.TestCase):
    def test_saved_destination_updates_and_keeps_user_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            profile = root / 'local-profile'
            profile.mkdir()
            destination = root / 'synced'
            text = 'first'

            def render(_, staging):
                (staging / 'index.md').write_text(text)
                return {'phoneRecords': 1}

            with patch.object(context_mirror, 'document_set', side_effect=render):
                self.assertIsNone(context_mirror.refresh_saved(profile))
                self.assertEqual(context_mirror.enable(profile, destination)['updated'], 1)
                text = 'second'
                self.assertEqual(context_mirror.refresh_saved(profile)['updated'], 1)
                self.assertEqual((destination / 'index.md').read_text(), 'second')
                self.assertEqual(context_mirror.refresh_saved(profile)['updated'], 0)
                (destination / 'index.md').write_text('user edit')
                with self.assertRaisesRegex(ValueError, 'context_file_was_edited'):
                    context_mirror.refresh_saved(profile)
                self.assertEqual((destination / 'index.md').read_text(), 'user edit')

    def test_publication_can_resume_after_interrupted_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            profile = root / 'local-profile'
            profile.mkdir()
            destination = root / 'synced'

            def render(_, staging):
                for name in ('day.md', 'index.md'):
                    (staging / name).write_text(name)
                return {'phoneRecords': 1}

            write = context_mirror.copy_atomic

            def write_with_failure(path, data):
                if path.name == 'index.md':
                    raise OSError('synthetic interrupt')
                write(path, data)

            with patch.object(context_mirror, 'document_set', side_effect=render):
                with patch.object(context_mirror, 'copy_atomic', side_effect=write_with_failure):
                    with self.assertRaises(OSError):
                        context_mirror.enable(profile, destination)
                self.assertTrue((destination / 'day.md').is_file())
                self.assertFalse((profile / 'context-mirror.json').exists())
                result = context_mirror.enable(profile, destination)
                self.assertEqual(result['updated'], 1)
                self.assertEqual((destination / 'index.md').read_text(), 'index.md')
                state = json.loads((profile / 'context-mirror-state.json').read_text())
                self.assertNotIn('pending', state)


class SpeakerReview(unittest.TestCase):
    def sample(self, start, embedding, alias=None, speaker='s1'):
        return {'speaker': speaker, 'start': start, 'end': start+5, 'embedding': embedding, 'alias': alias}

    def test_recurring_unidentified_voice_is_one_card_with_two_exact_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary)
            first = candidate_signals([self.sample(0, [1,0])])
            unrelated = candidate_signals([self.sample(0, [0,1])])
            second = candidate_signals([self.sample(10, [1,0])])
            data = [(profile / f'{label}.caf', {'id': label, 'sha256': label*64}, signal)
                    for label, signal in [('a', first), ('b', unrelated), ('c', second)]]
            report = review_summary(profile, data)
            self.assertEqual(report['pending'], 1)
            self.assertEqual(report['waitingForClearerAudio'], 1)
            item = report['items'][0]
            self.assertEqual(item['recordings'], 2)
            self.assertEqual([clip['assetID'] for clip in item['clips']], ['a', 'c'])
            self.assertTrue(all('embedding' not in clip for clip in item['clips']))

    def test_conflict_shows_distinct_evidence_and_known_voice_needs_no_duplicate_review(self):
        self.assertEqual(candidate_signals([self.sample(0,[1,0],'A'),self.sample(6,[1,0])]), [])
        signals = candidate_signals([self.sample(0,[1,0],'A'),self.sample(6,[0,1],'B'),
                                     self.sample(12,[0,1],'B')])
        self.assertEqual(signals[0]['reason'], 'conflicting_matches')
        self.assertEqual([r['alias'] for r in signals[0]['examples']], ['A','B'])
        mixed = candidate_signals([self.sample(0,[1,0],'A'),self.sample(6,[0,1])])
        self.assertEqual(mixed[0]['reason'], 'possibly_mixed_voice')


class HistoricalSpeakerLabels(unittest.TestCase):
    def test_apply_is_idempotent_and_mirror_checks_exact_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary)
            date = '2026-01-01T12:00:00+00:00'
            media = 'a' * 64
            asset = {'id': 'synthetic', 'sha256': media, 'start': date}
            registry = {'people': {'A': []}}
            run = profile / 'derivatives/qwen3-asr/key/run'
            run.mkdir(parents=True)
            text = run / 'transcript.md'
            text.write_text('original derivative')
            checksum = hashlib.sha256(text.read_bytes()).hexdigest()
            item = {
                'completed': True, 'transcriptSHA256': checksum, 'createdAt': date, 'duration': 30,
                'source': {'asset': asset, 'audioSHA256': media},
                'segments': [{'start': 0, 'end': 6, 'speaker': 's1', 'text': 'first'},
                             {'start': 8, 'end': 14, 'speaker': 's1', 'text': 'second'}],
            }
            (run / 'result.json').write_text(json.dumps(item))
            db_path = profile / 'library.sqlite'
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE assets(id TEXT, body TEXT)')
                db.execute('INSERT INTO assets VALUES (?,?)', (asset['id'], json.dumps(asset)))
            bundle_path = profile / 'verified-labels.json'
            bundle = {
                'schemaVersion': 1, 'sourceScanSHA256': 'b' * 64,
                'referenceSnapshot': hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest(),
                'recordings': [{'assetID': 'synthetic', 'audioSHA256': media, 'transcriptSHA256': checksum,
                                'start': date,
                                'confirmed': [{'assetID': 'synthetic', 'audioSHA256': media, 'start': 1, 'end': 5,
                                               'speakerCluster': 's1', 'status': 'user_confirmed', 'label': 'B'}],
                                'automatic': [{'start': 8, 'end': 14, 'speakerCluster': 's1',
                                               'status': 'automatic_match', 'label': 'A', 'segmentIndex': 1,
                                               'similarity': .85, 'margin': .2}]}],
            }
            bundle_path.write_text(json.dumps(bundle))
            expected = {'recordings': 1, 'automaticMatches': 1, 'confirmedIntervals': 1}
            with patch.object(speaker_labels, 'read_references', return_value=registry):
                self.assertEqual(speaker_labels.apply_bundle(profile, bundle_path), expected)
                self.assertEqual(speaker_labels.apply_bundle(profile, bundle_path), expected)
                bundle['recordings'][0]['automatic'][0]['segmentIndex'] = 0
                bundle_path.write_text(json.dumps(bundle))
                with self.assertRaisesRegex(ValueError, 'invalid_historical_speaker_label_interval'):
                    speaker_labels.apply_bundle(profile, bundle_path)
            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM audio_speaker_labels').fetchone(), (2,))
                self.assertEqual(json.loads(db.execute('SELECT body FROM assets').fetchone()[0]), asset)
            self.assertEqual(text.read_text(), 'original derivative')

            def phone_export(_, path, language):
                path.mkdir()
                (path / 'manifest.json').touch()
                return {'recordCount': 1, 'syncedAt': date}

            with patch.object(context_mirror, 'export_library', side_effect=phone_export):
                stage = profile / 'mirror'
                stage.mkdir()
                result = context_mirror.document_set(profile, stage)
                self.assertEqual(result['archiveAutomaticMatches'], 1)
                self.assertEqual(result['archiveConfirmedIntervals'], 1)
                output = (stage / '本机重转录/2026-01-01.md').read_text()
                self.assertIn('B（已确认片段）', output)
                self.assertIn('A（自动匹配）', output)
