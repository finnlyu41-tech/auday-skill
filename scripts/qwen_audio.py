"""Local Qwen3 ASR with optional anonymous diarization and separate derivatives."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from progress import report
from platform_utils import exclusive_file_lock, secure_directory
from refine_audio import attach_speakers, source_spans

PIPELINE_VERSION = 1


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.qwen-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def speech_windows(turns, duration, limit=30.0):
    """Keep speaker boundaries; ambiguous overlap is resolved by attribution."""
    result = []
    for row in sorted(turns, key=lambda x: (x['start'], x['end'])):
        left, right = max(0.0, row['start']), min(duration, row['end'])
        if not math.isfinite(left + right) or right - left < 0.3:
            continue
        # Merge only adjacent, same-speaker, non-overlapping windows.
        if (result and result[-1]['speaker'] == row['speaker']
                and 0 <= left - result[-1]['end'] <= 0.35
                and right - result[-1]['start'] <= limit):
            result[-1]['end'] = right
            continue
        while right - left > limit:
            result.append({'start': left, 'end': left + limit, 'speaker': row['speaker']})
            left += limit
        if right > left:
            result.append({'start': left, 'end': right, 'speaker': row['speaker']})
    return result


class Analyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.speaker_engine = None
        self.identity = hashlib.sha256(json.dumps({
            'pipelineVersion': PIPELINE_VERSION,
            'modelRevision': config['modelRevision'],
            'modelFiles': [x['sha256'] for x in config['modelFiles']],
            'speakerFiles': [x['sha256'] for x in config.get('speakerFiles', [])],
            'diarize': config.get('diarize', True),
            'runtimeVersion': config['runtimeVersion'],
            'windowSeconds': 30,
        }, sort_keys=True).encode()).hexdigest()

    def verify(self):
        for item in self.config['modelFiles'] + self.config.get('speakerFiles', []):
            if digest(item['path']) != item['sha256']:
                raise ValueError('qwen_model_integrity_failed')
        # Only audited local assets are permitted; no custom tokenizer code.
        tokenizer = json.loads((Path(self.config['modelDirectory']) / 'tokenizer_config.json').read_text())
        if tokenizer.get('auto_map'):
            raise ValueError('custom_tokenizer_code_not_allowed')

    def _turns(self, samples):
        if not self.config.get('diarize', True):
            return [{'start': 0.0, 'end': len(samples) / 16000, 'speaker': 'unknown'}], 0
        import sherpa_onnx
        if self.speaker_engine is None:
            settings = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=self.config['segmentationModel']), num_threads=4, provider='cpu'),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=self.config['embeddingModel'], num_threads=4, provider='cpu'),
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
                min_duration_on=0.3, min_duration_off=0.5)
            if not settings.validate():
                raise ValueError('invalid_qwen_diarization_config')
            self.speaker_engine = sherpa_onnx.OfflineSpeakerDiarization(settings)
        labels, turns = {}, []
        for row in self.speaker_engine.process(samples).sort_by_start_time():
            label = labels.setdefault(row.speaker, f'speaker_{len(labels) + 1:02d}')
            turns.append({'start': float(row.start), 'end': float(row.end), 'speaker': label})
        return turns, len(labels)

    def analyze(self, audio, output, asset=None):
        import numpy as np
        import soundfile as sf
        audio, output = Path(audio), Path(output)
        if audio.is_symlink() or not audio.is_file():
            raise ValueError('regular_audio_file_required')
        audio = audio.resolve()
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        source_hash = digest(audio)
        asset = asset or {}
        if asset.get('sha256') and source_hash != asset['sha256']:
            raise ValueError('asset_source_hash_mismatch')
        report('Preparing local audio for Qwen3-ASR', force=True)
        started = time.monotonic()
        wav = output / 'prepared.wav'
        with (output / 'decoder.log').open('wb') as log:
            subprocess.run([self.config['ffmpeg'], '-nostdin', '-hide_banner', '-loglevel', 'error',
                            '-i', str(audio), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                            str(wav)], stdout=log, stderr=log, check=True, timeout=600)
        samples, rate = sf.read(wav, dtype='float32')
        duration = len(samples) / rate
        if rate != 16000 or not 0 < duration <= 7200:
            raise ValueError('audio_must_be_positive_and_at_most_two_hours')
        report('Identifying anonymous speaker turns', force=True)
        turns, speaker_count = self._turns(samples)
        windows = speech_windows(turns, duration)
        if windows and self.model is None:
            from mlx_audio.stt import load
            with (output / 'engine.log').open('a') as engine_log, contextlib.redirect_stdout(engine_log), contextlib.redirect_stderr(engine_log):
                self.model = load(self.config['modelDirectory'])
        segments = []
        report('Transcribing speaker turns locally', 0, len(windows), 'turns', force=True)
        for index, window in enumerate(windows):
            left, right = window['start'], window['end']
            clip = np.asarray(samples[int(left * rate):int(right * rate)], dtype=np.float32)
            with (output / 'engine.log').open('a') as engine_log, contextlib.redirect_stdout(engine_log), contextlib.redirect_stderr(engine_log):
                result = self.model.generate(clip, language=None, temperature=0.0, max_tokens=1024,
                                             chunk_duration=30.0, min_chunk_duration=0.1, verbose=False)
            text = result.text.strip()
            if text:
                segments.append({'start': left, 'end': right, 'text': text,
                                 'language': result.language,
                                 'sourceSpans': source_spans(left, right, asset.get('timeline', [])),
                                 'speaker': 'unknown'})
            report('Transcribing speaker turns locally', index + 1, len(windows), 'turns')
        if self.config.get('diarize', True):
            attach_speakers(segments, turns)
        if digest(audio) != source_hash:
            raise ValueError('source_audio_changed_during_analysis')
        result = {
            'schemaVersion': 1, 'completed': True, 'pipelineIdentity': self.identity,
            'reviewStatus': 'machine-output-not-human-verified',
            'speakerCountMeaning': 'automatic voice clusters, not verified people',
            'createdAt': datetime.now(timezone.utc).isoformat(),
            'source': {'audioSHA256': source_hash, 'asset': asset},
            'engine': {'name': 'Qwen3-ASR-1.7B-8bit', 'revision': self.config['modelRevision'],
                       'runtime': self.config['runtimeVersion'], 'languagePolicy': 'auto-per-turn'},
            'timingPrecision': 'speaker-turn boundaries, not word alignment',
            'duration': duration, 'segments': segments,
            'diarization': {'status': 'completed' if self.config.get('diarize', True) else 'not_requested',
                            'speakerCount': speaker_count, 'identityScope': 'anonymous-within-this-asset',
                            'turns': turns},
            'elapsedSeconds': round(time.monotonic() - started, 3),
        }
        lines = ['# Local re-transcription', '', 'Machine output; no human accuracy review has been performed.',
                 'Speaker labels are anonymous clusters within this recording; voices may be split or merged.', '']
        for row in segments:
            lines.append(f"[{row['start']:.2f}–{row['end']:.2f}] {row['speaker']}: {row['text']}")
        (output / 'transcript.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        result['transcriptSHA256'] = digest(output / 'transcript.md')
        atomic_json(output / 'result.json', result)
        report('Local re-transcription complete', force=True)
        return result


def local_audio(profile):
    folder = Path(profile) / 'audio'
    suffixes = {'audio/mp4': '.m4a', 'audio/wav': '.wav', 'audio/x-wav': '.wav',
                'audio/x-caf': '.caf', 'audio/aac': '.aac'}
    for metadata in sorted(folder.glob('*.asset.json')):
        asset = json.loads(metadata.read_text(encoding='utf-8'))
        if not isinstance(asset, dict) or not isinstance(asset.get('sha256'), str):
            raise ValueError('invalid_local_audio_provenance')
        stem = metadata.name[:-len('.asset.json')]
        expected_suffix = suffixes.get(asset.get('mime'), '.bin')
        candidates = [p for p in folder.glob(stem + '.*')
                      if p.suffix in ('.m4a', '.wav', '.caf', '.aac', '.bin') and not p.is_symlink()]
        candidates.sort(key=lambda p: (p.suffix != expected_suffix, p.name))
        valid = next((p for p in candidates if digest(p) == asset['sha256']), None)
        if valid is None:
            raise ValueError('cached_audio_provenance_has_no_matching_media')
        yield valid, asset


def completed_result(path, identity, source_hash):
    try:
        saved = json.loads(Path(path).read_text())
        transcript = Path(path).parent / 'transcript.md'
        source = saved.get('source') if isinstance(saved, dict) else None
        return bool(isinstance(source, dict) and saved.get('completed') is True
                    and saved.get('pipelineIdentity') == identity
                    and source.get('audioSHA256') == source_hash
                    and transcript.is_file()
                    and digest(transcript) == saved.get('transcriptSHA256'))
    except (OSError, ValueError, TypeError):
        return False


def analysis_key(asset, source_hash, pipeline_identity):
    # Snapshot/transport/preparation timestamps must not force identical audio
    # through ASR again. Timeline changes do require a new derivative.
    source = {key: asset.get(key) for key in ('id', 'timeline', 'start', 'timeZone')}
    source['audioSHA256'] = source_hash
    return hashlib.sha256((json.dumps(source, sort_keys=True) + pipeline_identity).encode()).hexdigest()


def _process_batch(profile, config, limit=None):
    profile = Path(profile).resolve()
    analyzer = Analyzer(config)
    analyzer.verify()
    inputs = list(local_audio(profile))
    if limit is not None:
        inputs = inputs[:limit]
    completed = reused = 0
    failed = []
    for index, (audio, asset) in enumerate(inputs):
        key = analysis_key(asset, digest(audio), analyzer.identity)
        parent = profile / 'derivatives' / 'qwen3-asr' / key
        prior = sorted(parent.glob('*/result.json')) if parent.exists() else []
        valid = any(completed_result(previous, analyzer.identity, digest(audio)) for previous in prior)
        if valid:
            reused += 1
            report('Re-transcribing local audio', index + 1, len(inputs), 'assets')
            continue
        output = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + os.urandom(4).hex())
        try:
            analyzer.analyze(audio, output, asset)
            completed += 1
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            failed.append({'assetKey': key, 'errorType': type(error).__name__})
        report('Re-transcribing local audio', index + 1, len(inputs), 'assets', force=True)
    return {'completed': completed, 'reused': reused, 'failed': failed,
            'localAssets': len(inputs), 'complete': not failed,
            'outputDirectory': str(profile / 'derivatives' / 'qwen3-asr')}


def process_batch(profile, config, limit=None):
    folder = Path(profile).resolve() / 'derivatives' / 'qwen3-asr'
    secure_directory(folder)
    with exclusive_file_lock(folder / '.batch.lock', blocking=False):
        return _process_batch(profile, config, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'PYANNOTE_METRICS_ENABLED': '0'})
    os.umask(0o077)
    config = json.loads(args.config.read_text())
    if args.profile:
        result = process_batch(args.profile, config, args.limit)
    elif args.audio and args.output:
        analyzer = Analyzer(config)
        analyzer.verify()
        result = analyzer.analyze(args.audio, args.output)
    else:
        parser.error('provide --profile or both --audio and --output')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
