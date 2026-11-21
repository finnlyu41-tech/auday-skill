# Opt-in complete audio workflow

Read this when the user wants audio included in routine syncs, re-transcription,
or anonymous speaker labels. An ordinary text sync does not authorize enabling
this workflow. Save it only after the user explicitly chooses all-audio import
and local re-transcription. A saved choice remains effective on later syncs.

## Prepare and enable

The optional Apple Silicon route uses Qwen3-ASR-1.7B 8-bit through a separately
locked MLX Audio environment. It is appropriate to evaluate for Chinese,
Cantonese and English mixed speech. The model download is about 2.47 GB,
with about 30 MB of speaker models plus runtime packages. Weights are pinned
and verified. Models run locally; no embedding or transcription API is used.

Use the installed platform bootstrap prefix from first-run.md:

```text
<auday> setup --audio --engine qwen3-mlx
<auday> audio-workflow --enable
<auday> audio-workflow
```

Qwen's model and the MLX conversion are separate upstream artifacts. This route
currently requires Apple Silicon macOS. Existing Whisper setup remains available
on other supported hosts. Setup uses a public sample to verify execution and
speaker output; it does not establish accuracy on private recordings.

With the workflow enabled, the ordinary `sync` command:

1. Completes one ordinary text sync and readable file-library refresh.
2. Imports the complete audio catalogue through one resumable `audio_batch`.
3. Re-transcribes verified local files not already processed with the current
   source, model and pipeline identity; retains failed-run evidence.

`sync --skip-audio-workflow` performs a text-only invocation while preserving the
saved preference. `audio-workflow --disable` disables future automatic audio work.
Do not change phone permissions automatically. A text-only phone scope reports
that audio access is needed while retaining the completed text sync.

## Local processing and evidence

To finish processing already downloaded audio without the phone:

```text
<auday> retranscribe-local
```

Results live in the selected profile's `derivatives/qwen3-asr/`, separate from
phone records, original media and user notes. Each completed run includes
`result.json` and `transcript.md`, source/model hashes, detected language,
anonymous speaker turns and the original timeline mapping. Do not overwrite
phone transcripts with these derivatives. The temporary decoded WAV is removed
after the transcript and result are saved. Failed runs retain it for diagnosis.
When answering about an asset with a
completed re-transcription, inspect that derivative alongside its original
source and state which one is quoted. The phone-text semantic index remains
an index of the captured phone text; it does not silently index derivatives.

Diarization runs across each recording; ASR processes speaker turns in windows
up to 30 seconds and detects language per window. Timestamps are turn boundaries,
not word-level forced alignment. Preserve ambiguous/overlapping attribution as
unknown. Speaker labels do not establish identity or match people across files.
No persistent voiceprint enrollment is created.

Process assets sequentially (each at most two hours). For a new route, review a
representative recording for omissions, short replies, language changes, names,
numbers and speaker changes. Do not call successful execution measured accuracy.
All-audio archives may contain hundreds of hours: report actual counts, separate
import from transcription progress, and preserve completed work on interruption.
Do not run repeated phone syncs to chase new arrivals during a fixed snapshot.

Optional semantic indexing remains separate: update an already configured index
once after the user's sync workflow, following semantic-search.md. Do not install
new capabilities just because the user requested an ordinary refresh.
