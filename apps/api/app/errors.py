"""One error type for every route: a status code plus a message that is safe to show the user."""

from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
