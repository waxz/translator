from fastapi import HTTPException, status


class TranslationError(Exception):
    """Base exception for translation errors"""
    pass


class OpenAIAPIError(Exception):
    """Exception for OpenAI API errors"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


def handle_openai_error(error: Exception) -> HTTPException:
    """Convert OpenAI errors to Claude-compatible error format"""
    if isinstance(error, OpenAIAPIError):
        return HTTPException(
            status_code=error.status_code,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error.message
                }
            }
        )
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred"
            }
        }
    )