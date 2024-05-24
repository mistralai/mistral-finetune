class MessageFormatError(Exception):
    def __init__(self, message, data):
        self._message = message
        self._begin_data = data[:20]
        super().__init__()

    def __str__(self):
        return f"A message starting with {self._begin_data} is incorrectly formated." + self._message


class ToolCallFormatError(Exception):
    def __init__(self, message, data):
        self._message = message
        self._begin_data = data[:20]
        super().__init__()

    def __str__(self):
        return f"A tool call assistant message starting with {self._begin_data} of the conversation is incorrectly formated. " + self._message


class FunctionFormatError(Exception):
    def __init__(self, message, data):
        self._message = message
        self._begin_data = data[:20]
        super().__init__()

    def __str__(self):
        return (
            f"A function of the conversation starting with {self._begin_data} is incorrectly formated. "
            + self._message
        )


class ConversationFormatError(Exception):
    def __init__(self, message, data):
        self._message = message
        self._begin_data = data[:20]
        super().__init__()

    def __str__(self):
        return (
            f"A conversation starting with {self._begin_data} is incorrectly formated. " + self._message
        )


class UnrecognizedRoleError(Exception):
    def __init__(self, role, allowed_roles):
        self._role = role
        self._allowed_roles = allowed_roles
        super().__init__()

    def __str__(self):
        return (
            f"The following role: {self._role} is not recognized in line: {self.line} of the dataset {self.dataset}. Make sure that each role is one of {self._allowed_roles}"
            + self._message
        )
