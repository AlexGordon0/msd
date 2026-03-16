from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_COOCCURRENCE_INTERVAL: float = 0.1
VIDEO_WEIGHT: float = 0.6
VIDEO_CONFIDENCE_THRESHOLD: float = 0.5

# Prefix used for canonical (output) speaker IDs.
CANONICAL_PREFIX: str = "SPEAKER_"


@dataclass
class AtomicInterval:
    """A sub-segment during which neither channel's labelling changes."""

    start_ms: int  # start time in milliseconds (integer)
    end_ms: int  # end time in milliseconds (integer)
    audio_speaker: Optional[str]  # None if no audio coverage
    video_speaker: Optional[str]  # None if no video coverage
    video_confidence: float = 0.0  # meaningful only when video_speaker is set
    resolved_speaker: Optional[str] = field(default=None, init=False)

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000.0


def _to_ms(t: float) -> int:
    """Convert seconds (3 d.p.) to integer milliseconds."""
    return round(t * 1000)


def _build_atomic_timeline(
    audio_segments: list[dict],
    video_segments: list[dict],
) -> list[AtomicInterval]:
    """
    Sweep across the merged set of boundary points from both channels to
    produce a list of non-overlapping atomic intervals, each annotated with
    the audio and/or video speaker active during that interval.
    """
    # Collect all boundary points (milliseconds)
    boundaries: set[int] = {0}
    for seg in audio_segments + video_segments:
        start = _to_ms(seg["start"])
        end = _to_ms(seg["end"])
        boundaries.add(start)
        boundaries.add(end)

    sorted_boundaries = sorted(boundaries)

    # Build lookup: for each channel, map each ms point to the active segment
    def _build_lookup(segments: list[dict]) -> list[tuple[int, int, dict]]:
        """Returns list of (start_ms, end_ms, segment) sorted by start."""
        result = []
        for seg in segments:
            s = _to_ms(seg["start"])
            e = _to_ms(seg["end"])
            result.append((s, e, seg))
        return sorted(result, key=lambda x: x[0])

    audio_lookup = _build_lookup(audio_segments)
    video_lookup = _build_lookup(video_segments)

    def _active_segment(
        lookup: list[tuple[int, int, dict]], t_ms: int
    ) -> Optional[dict]:
        """Return the segment active at the midpoint of the interval starting at t_ms."""
        for s, e, seg in lookup:
            if s <= t_ms < e:
                return seg
        return None

    intervals: list[AtomicInterval] = []
    for i in range(len(sorted_boundaries) - 1):
        start_ms = sorted_boundaries[i]
        end_ms = sorted_boundaries[i + 1]

        if end_ms <= start_ms:
            continue

        mid_ms = (start_ms + end_ms) // 2

        audio_seg = _active_segment(audio_lookup, mid_ms)
        video_seg = _active_segment(video_lookup, mid_ms)

        # Skip completely silent / uncovered intervals
        if audio_seg is None and video_seg is None:
            continue

        intervals.append(
            AtomicInterval(
                start_ms=start_ms,
                end_ms=end_ms,
                audio_speaker=audio_seg["speaker"] if audio_seg else None,
                video_speaker=video_seg["speaker"] if video_seg else None,
                video_confidence=float(video_seg.get("confidence", 0.0))
                if video_seg
                else 0.0,
            )
        )

    return intervals


def _solve_identity_mapping(
    intervals: list[AtomicInterval],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build a co-occurrence overlap matrix between audio and video speaker IDs,
    then solve the assignment problem to find the best 1-to-1 mapping.

    Returns
    ───────
    audio_to_canonical : dict[audio_speaker_id -> canonical_id]
    video_to_canonical : dict[video_speaker_id -> canonical_id]

    Unmatched speakers (where one channel has more speakers than the other)
    receive their own unique canonical IDs.
    """
    # Collect unique speaker IDs from each channel
    audio_speakers = sorted({iv.audio_speaker for iv in intervals if iv.audio_speaker})
    video_speakers = sorted({iv.video_speaker for iv in intervals if iv.video_speaker})

    if not audio_speakers or not video_speakers:
        # Only one channel has any speakers — no mapping to solve
        audio_to_canonical = {
            spk: f"{CANONICAL_PREFIX}{i:02d}" for i, spk in enumerate(audio_speakers)
        }
        video_to_canonical = {
            spk: f"{CANONICAL_PREFIX}{i:02d}" for i, spk in enumerate(video_speakers)
        }
        return audio_to_canonical, video_to_canonical

    a_idx = {spk: i for i, spk in enumerate(audio_speakers)}
    v_idx = {spk: i for i, spk in enumerate(video_speakers)}

    # Co-occurrence matrix: rows = audio speakers, cols = video speakers
    matrix = np.zeros((len(audio_speakers), len(video_speakers)), dtype=float)

    for iv in intervals:
        if iv.duration_s < MIN_COOCCURRENCE_INTERVAL:
            continue
        if iv.audio_speaker and iv.video_speaker:
            r = a_idx[iv.audio_speaker]
            c = v_idx[iv.video_speaker]
            matrix[r, c] += iv.duration_s

    # Hungarian algorithm maximises overlap (negate for minimisation API)
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Assign canonical IDs to matched pairs
    audio_to_canonical: dict[str, str] = {}
    video_to_canonical: dict[str, str] = {}
    matched_audio: set[str] = set()
    matched_video: set[str] = set()

    canonical_counter = 0

    for r, c in zip(row_ind, col_ind):
        if matrix[r, c] == 0.0:
            # No overlap at all — treat as unmatched
            continue
        canonical_id = f"{CANONICAL_PREFIX}{canonical_counter:02d}"
        canonical_counter += 1
        audio_to_canonical[audio_speakers[r]] = canonical_id
        video_to_canonical[video_speakers[c]] = canonical_id
        matched_audio.add(audio_speakers[r])
        matched_video.add(video_speakers[c])

    # Unmatched audio speakers
    for spk in audio_speakers:
        if spk not in matched_audio:
            audio_to_canonical[spk] = f"{CANONICAL_PREFIX}{canonical_counter:02d}"
            canonical_counter += 1

    # Unmatched video speakers
    for spk in video_speakers:
        if spk not in matched_video:
            video_to_canonical[spk] = f"{CANONICAL_PREFIX}{canonical_counter:02d}"
            canonical_counter += 1

    return audio_to_canonical, video_to_canonical


def _resolve_intervals(
    intervals: list[AtomicInterval],
    audio_to_canonical: dict[str, str],
    video_to_canonical: dict[str, str],
) -> None:
    """
    Mutates each interval in-place, setting resolved_speaker.

    Decision logic
    ──────────────
    Both channels present:
        Use video if confidence >= VIDEO_CONFIDENCE_THRESHOLD,
        otherwise use audio.
        (Extend to a weighted vote if you have calibrated error rates.)
    Only one channel present:
        Use that channel directly.
    """
    for iv in intervals:
        has_audio = iv.audio_speaker is not None
        has_video = iv.video_speaker is not None

        if has_audio and has_video:
            if iv.video_confidence >= VIDEO_CONFIDENCE_THRESHOLD:
                iv.resolved_speaker = video_to_canonical.get(iv.video_speaker)
            else:
                iv.resolved_speaker = audio_to_canonical.get(iv.audio_speaker)

        elif has_audio:
            iv.resolved_speaker = audio_to_canonical.get(iv.audio_speaker)

        elif has_video:
            iv.resolved_speaker = video_to_canonical.get(iv.video_speaker)

        # Fallback — should not happen if mappings are complete
        if iv.resolved_speaker is None:
            iv.resolved_speaker = "UNKNOWN"


def _collapse_intervals(intervals: list[AtomicInterval]) -> list[dict]:
    """
    Merge consecutive atomic intervals that share the same resolved speaker
    into a single output segment.
    """
    if not intervals:
        return []

    output: list[dict] = []
    current_start_ms = intervals[0].start_ms
    current_end_ms = intervals[0].end_ms
    current_speaker = intervals[0].resolved_speaker

    for iv in intervals[1:]:
        if iv.resolved_speaker == current_speaker and iv.start_ms == current_end_ms:
            # Extend the current segment
            current_end_ms = iv.end_ms
        else:
            output.append(
                {
                    "start": round(current_start_ms / 1000, 3),
                    "duration": round((current_end_ms - current_start_ms) / 1000, 3),
                    "speaker": current_speaker,
                }
            )
            current_start_ms = iv.start_ms
            current_end_ms = iv.end_ms
            current_speaker = iv.resolved_speaker

    # Flush the last segment
    output.append(
        {
            "start": round(current_start_ms / 1000, 3),
            "duration": round((current_end_ms - current_start_ms) / 1000, 3),
            "speaker": current_speaker,
        }
    )

    return output


def merge_diarisation(
    audio_segments: list[dict],
    video_segments: list[dict],
) -> list[dict]:
    """Merge audio and video diarisation streams into a single output."""
    # 1. Atomic timeline
    intervals = _build_atomic_timeline(audio_segments, video_segments)

    if not intervals:
        return []

    # 2. Speaker identity mapping
    audio_to_canonical, video_to_canonical = _solve_identity_mapping(intervals)

    # 3. Resolve each interval
    _resolve_intervals(intervals, audio_to_canonical, video_to_canonical)

    # 4. Collapse and return
    return _collapse_intervals(intervals)
