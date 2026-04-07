class AnswerValidator:
    def validate(self, query, answer, docs):
        context = " ".join([d.get("text", "") for d in docs]).lower()

        answer_words = answer.lower().split()

        overlap = sum(1 for w in answer_words if w in context)

        score = overlap / max(len(answer_words), 1)

        if score < 0.3:
            return False, "⚠️ Answer not grounded in context"

        return True, answer