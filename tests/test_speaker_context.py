import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import context_mirror
from speaker_references import suggest


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
