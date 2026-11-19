"""Prepare one pinned Apple Silicon Qwen3 runtime and verify a public sample."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone

from platform_utils import support_root, venv_python
from progress import report
from setup_runtime import download_asset, ensure_uv, managed_env
from audio_setup import _command, _ensure_source, _probe_python
from qwen_audio import atomic_json, digest

HERE = Path(__file__).resolve().parent
ASSETS = json.loads((HERE / 'qwen-assets.json').read_text())


def runtime_config(root):
    return support_root(root) / 'runtime' / 'qwen-audio' / 'config.json'


def status(root):
    path = runtime_config(root)
    if not path.is_file():
        return {'configured': False, 'verifiedBySetup': False}
    try:
        value = json.loads(path.read_text())
        files = value.get('modelFiles', []) + value.get('speakerFiles', [])
        return {'configured': True, 'verifiedBySetup': bool(value.get('verifiedAt')),
                'filesPresent': bool(files) and all(Path(x['path']).is_file() for x in files),
                'verifiedAt': value.get('verifiedAt'), 'diarization': value.get('diarize', False),
                'engine': 'qwen3-mlx'}
    except (OSError, ValueError, TypeError, KeyError):
        return {'configured': True, 'verifiedBySetup': False, 'error': 'invalid_config'}


def prepare_models(root, diarize=True):
    root = support_root(root)
    model = root / 'models' / 'qwen3-asr-1.7b-8bit'
    model.mkdir(parents=True, exist_ok=True, mode=0o700)
    verified = []
    for asset in ASSETS['files']:
        name = asset['rfilename']
        if Path(name).name != name:
            raise ValueError('qwen_asset_path_must_be_a_filename')
        path = model / name
        url = ('https://huggingface.co/' + ASSETS['repository'] + '/resolve/'
               + ASSETS['revision'] + '/' + name)
        if asset.get('lfs'):
            download_asset(url, path, asset['lfs']['sha256'], asset['size'],
                           label='Downloading Qwen3-ASR model')
        else:
            if path.exists():
                content = path.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read(20 * 1024 * 1024)
            blob = hashlib.sha1(('blob ' + str(len(content)) + '\0').encode() + content).hexdigest()
            if blob != asset['blobId'] or len(content) != asset['size']:
                raise ValueError('qwen_configuration_integrity_failed')
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_bytes(content)
                temporary.replace(path)
        verified.append({'path': str(path), 'sha256': digest(path)})
    speakers, speaker_files = {}, []
    if diarize:
        manifest = json.loads((HERE / 'audio-assets.json').read_text())
        for name, asset in manifest['diarization'].items():
            path = download_asset(asset['url'], root / 'models' / 'audio' / asset['filename'],
                                  asset['sha256'], asset['size'], label='Downloading speaker model')
            speakers[name + 'Model'] = str(path)
            speaker_files.append({'path': str(path), 'sha256': asset['sha256']})
    return model, verified, speakers, speaker_files


def setup_qwen(root, diarize=True):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('qwen_mlx_requires_apple_silicon_macos')
    os.umask(0o077)
    root = support_root(root)
    runtime = root / 'runtime' / 'qwen-audio'
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = managed_env(root)
    env.update({'UV_PROJECT_ENVIRONMENT': str(runtime / 'venv'),
                'HF_HUB_DISABLE_TELEMETRY': '1', 'PYANNOTE_METRICS_ENABLED': '0'})
    uv = ensure_uv(root)
    command = [uv, 'sync', '--project', HERE / 'qwen-runtime', '--frozen', '--no-config',
               '--python', '3.12.15', '--managed-python', '--no-build']
    if diarize:
        command += ['--extra', 'diarization']
    _command(command, runtime / 'setup.log', env, 'Preparing isolated Qwen3 runtime')
    python = venv_python(runtime / 'venv')
    ffmpeg = _probe_python(python,
        "from pathlib import Path; import imageio_ffmpeg; "
        "print(next((Path(imageio_ffmpeg.__file__).parent/'binaries').glob('ffmpeg-macos*')))", env)
    model, files, speakers, speaker_files = prepare_models(root, diarize)
    config = {'schemaVersion': 1, 'python': str(python), 'ffmpeg': ffmpeg,
              'modelDirectory': str(model), 'modelRevision': ASSETS['revision'],
              'modelFiles': files, 'speakerFiles': speaker_files,
              'runtimeVersion': 'mlx-audio==0.6.6', 'diarize': diarize, **speakers}
    # Verify with a public recording; this is not a private-recording quality claim.
    source = _ensure_source(runtime)
    pending = runtime / 'pending-config.json'
    atomic_json(pending, config)
    output = runtime / ('verification-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    _command([python, HERE / 'qwen_audio.py', '--config', pending,
              '--audio', source / 'samples' / 'jfk.wav', '--output', output],
             runtime / 'verification.log', env, 'Verifying local ASR and speaker pipeline', timeout=600)
    result = json.loads((output / 'result.json').read_text())
    words = ' '.join(x['text'] for x in result['segments']).lower()
    if 'country' not in words or not result['completed']:
        raise ValueError('qwen_public_sample_verification_failed')
    if diarize and result['diarization']['speakerCount'] < 1:
        raise ValueError('speaker_sample_verification_failed')
    config['verifiedAt'] = datetime.now(timezone.utc).isoformat()
    config['verification'] = {'publicSampleOnly': True, 'nonemptyTranscript': True,
                              'diarizationCompleted': diarize,
                              'representativePrivateAudioQuality': 'not_yet_reviewed'}
    atomic_json(runtime / 'config.json', config)
    pending.unlink(missing_ok=True)
    return {'ok': True, 'engine': 'qwen3-mlx', 'verifiedAt': config['verifiedAt'],
            'diarizationReady': diarize, 'config': str(runtime / 'config.json')}
