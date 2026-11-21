import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audio_workflow as workflow
import qwen_audio as qwen


class PreparedAudioTests(unittest.TestCase):
    def test_qwen_cleans_wav_only_after_saving_completed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'original.wav'
            source.write_bytes(b'original audio')
            analyzer = qwen.Analyzer({
                'modelRevision': 'synthetic', 'modelFiles': [], 'runtimeVersion': 'synthetic',
                'ffmpeg': 'synthetic', 'diarize': False,
            })

            def decode(command, **kwargs):
                Path(command[-1]).write_bytes(b'decoded audio')

            soundfile = SimpleNamespace(read=lambda *args, **kwargs: ([0.0] * 16000, 16000))
            with patch.dict(sys.modules, {'numpy': SimpleNamespace(), 'soundfile': soundfile}), \
                    patch.object(qwen.subprocess, 'run', side_effect=decode), \
                    patch.object(analyzer, '_turns', return_value=([], 0)):
                success = root / 'success'
                result = analyzer.analyze(source, success)
                self.assertTrue(qwen.completed_result(
                    success / 'result.json', result['pipelineIdentity'], qwen.digest(source)))
                self.assertFalse((success / 'prepared.wav').exists())

                failed = root / 'failed'
                with patch.object(qwen, 'atomic_json', side_effect=OSError('synthetic write failure')):
                    with self.assertRaises(OSError):
                        analyzer.analyze(source, failed)
                self.assertFalse((failed / 'result.json').exists())
                self.assertTrue((failed / 'prepared.wav').is_file())
                self.assertEqual(source.read_bytes(), b'original audio')


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.profile = self.home / 'default'
        self.profile.mkdir()
        self.app = Mock(root=self.profile, profile='default')
        self.app.root = self.profile
        self.app.profile = 'default'
        self.app.metadata.return_value = {'access': 'text-and-audio-read-only'}
        self.app.audio_batch.return_value = {'complete': True}

    def tearDown(self):
        self.tmp.cleanup()

    def enable(self):
        config = self.home / 'verified-config.json'
        config.write_text(json.dumps({'verifiedAt': 'verified', 'diarize': True}))
        with patch('qwen_audio_setup.runtime_config', return_value=config):
            return workflow.configure(self.home, enabled=True, languages=['Chinese', 'Cantonese', 'English'])

    def test_default_sync_does_not_import_audio_or_install_tools(self):
        self.assertIsNone(workflow.after_sync(self.app))
        self.app.audio_batch.assert_not_called()

    def test_enable_requires_verified_speaker_pipeline(self):
        with patch('qwen_audio_setup.runtime_config', return_value=self.home / 'missing'):
            with self.assertRaisesRegex(ValueError, 'prepare_and_verify'):
                workflow.configure(self.home, enabled=True)
        self.assertFalse((self.profile / 'audio-workflow.json').exists())

    def test_saved_workflow_runs_one_batch_and_local_processing(self):
        self.enable()
        with patch.object(workflow, 'reprocess_local', return_value={'complete': True}) as process:
            result = workflow.after_sync(self.app)
        self.app.audio_batch.assert_called_once_with()
        process.assert_called_once_with(self.home, 'default')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['originalPhoneRecordsPreserved'])
        self.assertTrue(workflow.preferences(self.home)['enabled'])

    def test_text_only_invocation_keeps_saved_preference(self):
        self.enable()
        self.assertEqual(workflow.after_sync(self.app, skip=True)['status'], 'skipped_for_this_sync')
        self.assertTrue(workflow.preferences(self.home)['enabled'])
        self.app.audio_batch.assert_not_called()

    def test_phone_scope_is_never_expanded(self):
        self.enable()
        self.app.metadata.return_value = {'access': 'text-only-read-only'}
        result = workflow.after_sync(self.app)
        self.assertEqual(result['status'], 'needs_phone_audio_scope')
        self.assertTrue(result['textSyncPreserved'])
        self.app.audio_batch.assert_not_called()

    def test_partial_import_retains_processed_work_without_retrying_phone(self):
        self.enable()
        self.app.audio_batch.return_value = {'complete': False, 'remaining': 2}
        with patch.object(workflow, 'reprocess_local', return_value={'complete': True}):
            result = workflow.after_sync(self.app)
        self.assertEqual(result['status'], 'partial')
        self.app.audio_batch.assert_called_once_with()

    def test_prepared_runtime_cannot_silently_disable_saved_speakers(self):
        self.enable()
        config = self.home / 'changed-runtime.json'
        config.write_text(json.dumps({'verifiedAt': 'verified', 'diarize': False}))
        with patch('qwen_audio_setup.runtime_config', return_value=config), patch.object(workflow.subprocess, 'run') as run:
            result = workflow.reprocess_local(self.home)
        self.assertEqual(result['status'], 'needs_verified_diarization_setup')
        run.assert_not_called()

    def test_invalid_profile_cannot_write_outside_home(self):
        with self.assertRaisesRegex(ValueError, 'invalid_profile'):
            workflow.configure(self.home, '../other', enabled=False)


class WindowTests(unittest.TestCase):
    def test_long_turns_preserve_full_interval_with_bounded_windows(self):
        rows = qwen.speech_windows([{'start': 1.0, 'end': 76.0, 'speaker': 'speaker_01'}], 80)
        self.assertEqual([(r['start'], r['end']) for r in rows], [(1, 31), (31, 61), (61, 76)])

    def test_different_and_overlapping_speakers_are_not_merged(self):
        rows = qwen.speech_windows([
            {'start': 0, 'end': 3, 'speaker': 'speaker_01'},
            {'start': 2, 'end': 5, 'speaker': 'speaker_02'}], 5)
        self.assertEqual(len(rows), 2)
        attached = qwen.attach_speakers([{'start': 2, 'end': 3}], rows)
        self.assertEqual(attached[0]['speaker'], 'unknown')

    def test_invalid_duration_is_not_sent_to_model(self):
        self.assertEqual(qwen.speech_windows([{'start': 2, 'end': 1, 'speaker': 'x'}], 5), [])


class ResumeTests(unittest.TestCase):
    def test_completed_outputs_reused_but_changed_source_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'source.m4a'
            audio.write_bytes(b'synthetic-audio')
            asset = {'id': 'synthetic', 'sha256': qwen.digest(audio)}
            fake = Mock(identity='pipeline-v1')
            def analyze(source, output, provenance):
                output.mkdir(parents=True)
                (output / 'transcript.md').write_text('Synthetic words')
                qwen.atomic_json(output / 'result.json', {
                    'completed': True, 'pipelineIdentity': fake.identity,
                    'source': {'audioSHA256': qwen.digest(source)},
                    'transcriptSHA256': qwen.digest(output / 'transcript.md')})
            fake.analyze.side_effect = analyze
            with patch.object(qwen, 'Analyzer', return_value=fake), patch.object(qwen, 'local_audio', return_value=[(audio, asset)]):
                first = qwen.process_batch(root, {})
                second = qwen.process_batch(root, {})
                self.assertEqual(first['completed'], 1)
                self.assertEqual(second['reused'], 1)
                audio.write_bytes(b'changed-synthetic-audio')
                asset['sha256'] = qwen.digest(audio)
                third = qwen.process_batch(root, {})
                self.assertEqual(third['completed'], 1)
            self.assertEqual(audio.read_bytes(), b'changed-synthetic-audio')
            self.assertEqual(fake.analyze.call_count, 2)


    def test_duplicate_suffixes_select_verified_canonical_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'audio'
            folder.mkdir()
            canonical = folder / 'same.caf'
            canonical.write_bytes(b'synthetic')
            (folder / 'same.bin').write_bytes(b'synthetic')
            qwen.atomic_json(folder / 'same.asset.json',
                             {'id': 'sample', 'mime': 'audio/x-caf', 'sha256': qwen.digest(canonical)})
            inputs = list(qwen.local_audio(root))
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0][0], canonical)
            canonical.write_bytes(b'corrupted')
            inputs = list(qwen.local_audio(root))
            self.assertEqual(inputs[0][0].suffix, '.bin')
            (folder / 'same.bin').unlink()
            with self.assertRaisesRegex(ValueError, 'no_matching_media'):
                list(qwen.local_audio(root))

    def test_transport_metadata_does_not_force_retranscription_but_timeline_does(self):
        asset = {'id': 'sample', 'timeline': [{'fileStart': 0, 'fileEnd': 3}], 'preparedAt': 'before'}
        first = qwen.analysis_key(asset, 'hash', 'pipeline')
        asset['preparedAt'] = 'after'
        self.assertEqual(first, qwen.analysis_key(asset, 'hash', 'pipeline'))
        asset['timeline'][0]['fileEnd'] = 4
        self.assertNotEqual(first, qwen.analysis_key(asset, 'hash', 'pipeline'))

    def test_malformed_or_damaged_outputs_are_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / 'result.json'
            transcript = root / 'transcript.md'
            transcript.write_text('Original synthetic transcript')
            good = {'completed': True, 'pipelineIdentity': 'pipeline',
                    'source': {'audioSHA256': 'source'},
                    'transcriptSHA256': qwen.digest(transcript)}
            qwen.atomic_json(result, good)
            self.assertTrue(qwen.completed_result(result, 'pipeline', 'source'))
            transcript.write_text('')
            self.assertFalse(qwen.completed_result(result, 'pipeline', 'source'))
            result.write_text('[]')
            self.assertFalse(qwen.completed_result(result, 'pipeline', 'source'))
            result.write_text('{"source": []}')
            self.assertFalse(qwen.completed_result(result, 'pipeline', 'source'))


if __name__ == '__main__':
    unittest.main()
