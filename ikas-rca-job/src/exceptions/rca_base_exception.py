class RCABaseException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"Task failed.Message: {self.message}"