# Speaker references and text mirror

Set up named matching only after the user has identified the voices and approved
local Auday matching. First test on different recordings and distractor speech.
Use a separate sample from each enrolled person for checking the pilot. Treat
accepted model matches as candidates, and keep anonymous labels when the clip
is too short, overlapping, has a weak score, or the leading two scores are close.

The optional private `speaker-references.json` is stored in the selected profile.
Schema version 1 has `modelSHA256` and `people`, a map of approved aliases to
lists of references. Each entry has `assetID`, `audioSHA256`, `speakerCluster`,
`start`, `end`, and a unit-vector `embedding` from the audited local speaker model.
Use the original verified audio window as input and retain the exact boundaries.
Use at least two people for comparison. Exclude references from the recording
being analyzed. A verified model change requires rebuilding the references.
Never copy the registry, voice features, samples, or local settings to a mirror.
New Qwen runs can suggest a name alongside the anonymous cluster. Existing
transcripts remain intact. For the user-confirmed samples the exported copy adds
an explicit confirmation label to the corresponding time interval.

## Review new voices after sync

With enrolled voices, new local Qwen results keep small private review signals.
The normal sync result includes `audioWorkflow.transcription.speakerReview`
when those signals exist. It prioritizes conflicting names, then recurring or
particularly clear unknown voices. Each item gives one or two time ranges with
the original local audio path and SHA-256. Use that path to make short local
listening clips after checking the hash. A short isolated unknown can wait for
clearer evidence. Similarity only groups examples for review.

Ask for one label per coherent voice, and present conflicting samples as
separate clips. If the user hears different people, preserve exact interval
labels. Save decisions locally under the profile's `speaker-review/` directory
in an `auday-speaker-review-*.json` file with a `decisions` list. Each entry
contains `assetID`, `audioSHA256`, `choice`, the confirmed `label`, and
`windows` containing the confirmed start and end seconds. This suppresses
repeated auditions of reviewed ranges on the next sync. Use the enrollment
procedure above for clear, approved names. A review receipt alone does not
activate a new person or define the identity of an entire recording.

## Refresh the text mirror

When the user requests a synced text copy, run
`context-mirror --destination <absolute-local-sync-directory>` once against an
empty destination. Future `sync` commands update that directory after the text
and audio stages. Repeat `context-mirror` to update from the current local data
without contacting the phone. The synced directory receives Markdown only. Other
computers can open `index.md` from the same synced account. An existing user
edit stops publication, and the local publication journal supports retry after
an interrupted copy. The command verifies local files. The sync provider handles
remote propagation, which is separate from local publication.
