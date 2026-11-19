"""Explicit per-profile audio import and re-transcription preferences."""
import json
import os
from pathlib import Path
import re
import subprocess


from qwen_audio import atomic_json

HERE = Path(__file__).resolve().parent


def profile_root(home, profile):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', profile):
        raise ValueError('invalid_profile_name')
    return Path(home).resolve() / profile


def preferences(home, profile='default'):
    path = profile_root(home, profile) / 'audio-workflow.json'
    if not path.exists():
        return {'schemaVersion': 1, 'enabled': False}
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schemaVersion') != 1 or type(value.get('enabled')) is not bool:
        raise ValueError('invalid_audio_workflow_preferences')
    if value['enabled'] and (value.get('engine') != 'qwen3-mlx'
                             or value.get('scope') != 'all-audio'
                             or value.get('diarize') is not True):
        raise ValueError('unsupported_audio_workflow_preferences')
    return value


def configure(home, profile='default', *, enabled, languages=None):
    # This saves an explicitly requested workflow, never phone credentials.
    root = profile_root(home, profile)
    if enabled:
        from qwen_audio_setup import runtime_config
        path = runtime_config(home)
        config = json.loads(path.read_text()) if path.is_file() else {}
        if not config.get('verifiedAt') or not config.get('diarize'):
            raise ValueError('prepare_and_verify_qwen_with_diarization_first')
    try:
        previous = preferences(home, profile)
    except (OSError, ValueError):
        previous = {}
    value = {'schemaVersion': 1, 'enabled': enabled, 'engine': 'qwen3-mlx',
             'scope': 'all-audio', 'diarize': True, 'languagePolicy': 'automatic',
             'preserveOriginals': True, 'reuseUnchangedDerivatives': True}
    if languages is None:
        languages = previous.get('preferredLanguages')
    if languages is not None:
        if not languages or any(not isinstance(x, str) or len(x) > 40 for x in languages):
            raise ValueError('invalid_preferred_languages')
        value['preferredLanguages'] = list(dict.fromkeys(languages))
    atomic_json(root / 'audio-workflow.json', value)
    return value


def reprocess_local(home, profile='default'):
    from qwen_audio_setup import runtime_config
    config_path = runtime_config(home)
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if not config.get('verifiedAt'):
        return {'status': 'needs_qwen_setup', 'complete': False}
    selected = preferences(home, profile)
    if selected.get('enabled') and selected.get('diarize') and config.get('diarize') is not True:
        return {'status': 'needs_verified_diarization_setup', 'complete': False}
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', PYANNOTE_METRICS_ENABLED='0')
    # Only this bundled runner is executed; neither recordings nor phone data
    # supply commands. The selected interpreter is a locally verified setting.
    result = subprocess.run([config['python'], str(HERE / 'qwen_audio.py'),
                             '--config', str(config_path), '--profile',
                             str(profile_root(home, profile))],
                            env=env, stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout)


def after_sync(app, *, skip=False):
    home, profile = app.root.parent, app.profile
    if skip:
        return {'status': 'skipped_for_this_sync'}
    try:
        selected = preferences(home, profile)
    except (OSError, ValueError):
        return {'status': 'invalid_audio_workflow_configuration', 'complete': False,
                'textSyncPreserved': True}
    if not selected['enabled']:
        return None
    if app.metadata().get('access') != 'text-and-audio-read-only':
        return {'status': 'needs_phone_audio_scope', 'complete': False,
                'textSyncPreserved': True}
    try:
        imported = app.audio_batch()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        imported = {'complete': False, 'errorType': type(error).__name__}
    try:
        processed = reprocess_local(home, profile)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        processed = {'complete': False, 'errorType': type(error).__name__}
    return {'status': 'completed' if imported.get('complete') and processed.get('complete') else 'partial',
            'imports': imported, 'transcription': processed,
            'originalPhoneRecordsPreserved': True}
