class UserError(Exception):
    """An expected failure caused by the request (bad file, empty query, ...).

    Services raise it without knowing about HTTP; main.py renders it as {"detail": message}.
    """

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code
