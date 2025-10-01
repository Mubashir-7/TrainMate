from fastapi import Request

def flash(request: Request, message: str, category: str = "primary") -> None:
    """
    Adds a flash message to the session.
    `category` can be 'success', 'error', 'warning', 'info'.
    """
    if "_messages" not in request.session:
        request.session["_messages"] = []
    request.session["_messages"].append({"message": message, "category": category})

def get_flashed_messages(request: Request):
    """Retrieves and clears flash messages from the session."""
    return request.session.pop("_messages") if "_messages" in request.session else []