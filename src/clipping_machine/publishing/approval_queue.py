"""Approval queue: every clip is reviewed by a person before it can leave the app.

States: ``candidate → approved → exported → published → archived`` with ``rejected``
as the side exit. ``published`` only records that the *user* published the file
manually elsewhere; this project contains no code that uploads or posts anything.
"""

from __future__ import annotations

from clipping_machine.storage.models import Clip, ClipStatus
from clipping_machine.storage.repositories import ClipRepository

ALLOWED_TRANSITIONS: dict[ClipStatus, frozenset[ClipStatus]] = {
    ClipStatus.CANDIDATE: frozenset({ClipStatus.APPROVED, ClipStatus.REJECTED}),
    ClipStatus.APPROVED: frozenset(
        {ClipStatus.EXPORTED, ClipStatus.REJECTED, ClipStatus.CANDIDATE}
    ),
    ClipStatus.REJECTED: frozenset({ClipStatus.CANDIDATE, ClipStatus.ARCHIVED}),
    ClipStatus.EXPORTED: frozenset(
        {ClipStatus.PUBLISHED, ClipStatus.ARCHIVED, ClipStatus.APPROVED}
    ),
    ClipStatus.PUBLISHED: frozenset({ClipStatus.ARCHIVED}),
    ClipStatus.ARCHIVED: frozenset(),
}

# Statuses whose clips still count as "already made" when scoring novelty.
ACTIVE_STATUSES: tuple[ClipStatus, ...] = (
    ClipStatus.CANDIDATE,
    ClipStatus.APPROVED,
    ClipStatus.EXPORTED,
    ClipStatus.PUBLISHED,
)


class InvalidTransitionError(ValueError):
    def __init__(self, clip_id: str, current: ClipStatus, target: ClipStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"clip {clip_id} cannot go from '{current.value}' to '{target.value}'")


def can_transition(current: ClipStatus, target: ClipStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


class ApprovalQueue:
    def __init__(self, clips: ClipRepository) -> None:
        self._clips = clips

    def transition(self, clip_id: str, target: ClipStatus, note: str = "") -> Clip:
        clip = self._clips.require(clip_id)
        if not can_transition(clip.status, target):
            raise InvalidTransitionError(clip_id, clip.status, target)
        self._clips.set_status(clip_id, clip.status, target, note)
        return self._clips.require(clip_id)

    def approve(self, clip_id: str, note: str = "") -> Clip:
        return self.transition(clip_id, ClipStatus.APPROVED, note)

    def reject(self, clip_id: str, note: str = "") -> Clip:
        return self.transition(clip_id, ClipStatus.REJECTED, note)

    def reopen(self, clip_id: str, note: str = "") -> Clip:
        return self.transition(clip_id, ClipStatus.CANDIDATE, note)

    def mark_exported(self, clip_id: str) -> Clip:
        """Called by the exporter only; requires prior human approval."""
        return self.transition(clip_id, ClipStatus.EXPORTED, "MP4 exported")

    def mark_published_manually(self, clip_id: str, note: str = "") -> Clip:
        """Bookkeeping only: the user tells the app they posted the clip themselves."""
        return self.transition(clip_id, ClipStatus.PUBLISHED, note or "published manually by user")

    def archive(self, clip_id: str, note: str = "") -> Clip:
        return self.transition(clip_id, ClipStatus.ARCHIVED, note)

    def require_approved(self, clip_id: str) -> Clip:
        """Guard used before any final export."""
        clip = self._clips.require(clip_id)
        if clip.status is not ClipStatus.APPROVED:
            raise InvalidTransitionError(clip_id, clip.status, ClipStatus.EXPORTED)
        return clip
