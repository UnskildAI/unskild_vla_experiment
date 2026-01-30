"""Framework-specific exceptions."""


class VLAError(Exception):
    """Base exception for VLA framework."""

    pass


class VLAConfigError(VLAError):
    """Configuration validation or loading error."""

    pass


class VLACheckpointError(VLAError):
    """Checkpoint save/load error."""

    pass


class VLADataError(VLAError):
    """Dataset or dataloader error."""

    pass


class VLATrainingError(VLAError):
    """Training loop or optimizer error."""

    pass
