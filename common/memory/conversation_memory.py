class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []   # stores (role, message)

    # ----------------------------------------
    # 🔹 ADD USER MESSAGE
    # ----------------------------------------
    def add_user_message(self, message):
        self.history.append(("user", message))
        self._trim()

    # ----------------------------------------
    # 🔹 ADD AI MESSAGE
    # ----------------------------------------
    def add_ai_message(self, message):
        self.history.append(("assistant", message))
        self._trim()

    # ----------------------------------------
    # 🔹 GET CONTEXT (FOR PROMPT)
    # ----------------------------------------
    def get_context(self):
        context = ""

        for role, msg in self.history:
            if role == "user":
                context += f"User: {msg}\n"
            else:
                context += f"Assistant: {msg}\n"

        return context.strip()

    # ----------------------------------------
    # 🔹 TRIM MEMORY
    # ----------------------------------------
    def _trim(self):
        # keep last N turns (user+assistant pairs)
        max_items = self.max_turns * 2
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]