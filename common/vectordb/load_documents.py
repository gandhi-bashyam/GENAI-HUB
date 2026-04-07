# from langchain.docstore.document import Document
from langchain_core.documents import Document

def load_documents(folder_path: str):
    """
    Load documents from a folder and return standardized format
    """
    docs = [
        {
            "text": "William Shakespeare wrote Hamlet.",
            "metadata": {"category": "Literature", "project": "KnowledgeBase"}
        },
        {
            "text": "Hamlet is a tragic play.",
            "metadata": {"category": "Literature", "project": "KnowledgeBase"}
        },
        {
            "text": "Paris is the capital of France.",
            "metadata": {"category": "Geography", "project": "KnowledgeBase"}
        },
        {
            "text": "Employees are entitled to 20 leave days",
            "metadata": {"category": "HR", "project": "Benefits"}
        }
    ]

    # Example (you can expand later)
    docs.append({
        "text": "Employees are entitled to 20 leave days",
        "metadata": {"category": "HR", "project": "Policy"}
    })

    docs.append({
        "text": "Leave approval must come from manager",
        "metadata": {"category": "HR", "project": "Policy"}
    })

    docs.append({
        "text": "Health insurance benefits are provided",
        "metadata": {"category": "HR", "project": "Benefits"}
    })

    docs.append({
        "text": "Hamlet Hamlet Hamlet wrote Hamlet movie adaptation",
        "metadata": {"category": "Geography", "project": "KnowledgeBase"}
    })

    return docs