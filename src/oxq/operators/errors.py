"""Errors raised while loading a provider certification submission."""


class OperatorCertificationError(ValueError):
    """A structured, stable certification intake failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.operator_id = operator_id

    def as_dict(self) -> dict[str, str]:
        result = {
            "status": "fail",
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }
        if self.operator_id is not None:
            result["operator_id"] = self.operator_id
        return result
