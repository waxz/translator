from fastapi import FastAPI, Depends, HTTPException, status, Request as FastAPIRequest
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from contextlib import asynccontextmanager
import logging
import traceback
import json

from claude_to_openai_forwarder.config import get_settings
from claude_to_openai_forwarder.models.claude import ClaudeRequest, ClaudeResponse, ClaudeUsage
from claude_to_openai_forwarder.backends import get_backend, get_backend_name
from claude_to_openai_forwarder.translators.request import RequestTranslator
from claude_to_openai_forwarder.translators.response import ResponseTranslator
from claude_to_openai_forwarder.translators.streaming import StreamingTranslator
from claude_to_openai_forwarder.middleware.auth import verify_claude_api_key
from claude_to_openai_forwarder.utils.exceptions import handle_openai_error, OpenAIAPIError

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    settings = get_settings()
    backend_name = get_backend_name()
    logger.info(f"Starting Claude-to-OpenAI API Forwarder")
    logger.info(f"Backend: {backend_name}")
    if settings.model_provider:
        logger.info(f"Model Provider: {settings.model_provider}")
    logger.info(f"OpenAI Base URL: {settings.openai_base_url}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Claude-to-OpenAI API Forwarder",
    description="Forwards Claude API requests to OpenAI API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: FastAPIRequest, exc: RequestValidationError
):
    """Handle validation errors with detailed info"""
    logger.error(f"Validation error: {exc}")
    logger.error(f"Errors: {exc.errors()}")

    # Try to get the raw body
    try:
        body = await request.body()
        logger.error(f"Request body that failed: {body.decode()[:1000]}")
    except:
        pass

    return JSONResponse(
        status_code=422,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"Validation error: {str(exc.errors())}",
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: FastAPIRequest, exc: HTTPException):
    """Return Claude-compatible error bodies without FastAPI's detail wrapper."""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": str(exc.detail),
            },
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: FastAPIRequest, exc: Exception):
    """Global exception handler for better error messages"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": f"Internal server error: {str(exc)}",
            },
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "claude-to-openai-forwarder"}


@app.post("/v1/messages")
async def create_message(
    raw_request: FastAPIRequest,
    api_key: str = Depends(verify_claude_api_key),
):
    """
    Claude Messages API endpoint
    Accepts Claude-formatted requests and returns Claude-formatted responses
    """
    try:
        settings = get_settings()
        client = get_backend()

        logger.info(f"Using backend: {client.get_backend_name()}")

        # Parse body manually first for debugging
        body = await raw_request.body()
        body_str = body.decode()

        logger.info("=" * 80)
        logger.info("RAW REQUEST BODY:")
        logger.info(body_str[:2000])  # First 2000 chars
        logger.info("=" * 80)

        # Parse JSON
        try:
            body_dict = json.loads(body_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid JSON: {str(e)}",
                    },
                },
            )

        logger.info(f"Parsed JSON keys: {list(body_dict.keys())}")
        logger.info(f"Model: {body_dict.get('model')}")
        logger.info(f"Max tokens: {body_dict.get('max_tokens')}")
        logger.info(f"Stream: {body_dict.get('stream')}")
        logger.info(f"Messages count: {len(body_dict.get('messages', []))}")
        logger.info(f"Tools count: {len(body_dict.get('tools', []))}")

        if body_dict.get("tools"):
            logger.info("Tools in request:")
            for tool in body_dict.get("tools", []):
                logger.info(
                    f"  - {tool.get('name')}: {tool.get('description', '')[:50]}"
                )

        # Try to parse into Pydantic model
        try:
            request = ClaudeRequest(**body_dict)
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            logger.error(f"Errors: {e.errors()}")
            raise HTTPException(
                status_code=422,
                detail={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Request validation failed: {str(e.errors())}",
                    },
                },
            )

        logger.info(f"Successfully parsed request")
        logger.info(f"  Model: {request.model}")
        logger.info(f"  Stream: {request.stream}")
        logger.info(f"  Max tokens: {request.max_tokens}")
        logger.info(f"  Messages: {len(request.messages)}")
        logger.info(f"  Tools: {len(request.tools) if request.tools else 0}")

        # Translate Claude request to OpenAI format
        openai_request = RequestTranslator.translate(
            request, default_model=settings.default_openai_model
        )

        logger.info(f"Translated OpenAI Request:")
        logger.info(f"  Model: {openai_request.model}")
        logger.info(f"  Messages: {len(openai_request.messages)}")
        logger.info(
            f"  Tools: {len(openai_request.tools) if openai_request.tools else 0}"
        )

        # Handle streaming
        if request.stream:

            async def generate():
                try:
                    logger.info("Starting stream generation")
                    openai_stream = client.create_completion_stream(openai_request)
                    event_count = 0
                    async for event in StreamingTranslator.translate_stream(
                        openai_stream
                    ):
                        event_count += 1
                        logger.info(f"Yielding event {event_count}: {event[:100]}...")
                        yield event
                    logger.info(f"Stream complete, yielded {event_count} events")
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Handle non-streaming
        openai_response = await client.create_completion(openai_request)

        # Translate OpenAI response to Claude format
        claude_response = ResponseTranslator.translate(openai_response)

        logger.info(f"Request completed: {claude_response.id}")
        return claude_response

    except HTTPException:
        raise
    except OpenAIAPIError as e:
        raise handle_openai_error(e)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {"type": "internal_error", "message": str(e)},
            },
        )


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    raw_request: FastAPIRequest,
    api_key: str = Depends(verify_claude_api_key),
):
    """
    Token‑Counting endpoint – mirrors the official Claude API.
    Returns the number of input tokens for the supplied messages (including tools, images, etc.).
    The response shape follows the Claude response model but contains only the ``usage`` field.
    """
    try:
        # Parse raw body
        body = await raw_request.body()
        body_dict = json.loads(body.decode())
        request = ClaudeRequest(**body_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid request for count_tokens: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": f"Invalid JSON: {str(e)}"},
            },
        )

    # Translate to OpenAI request – clear max_tokens to request only token usage
    settings = get_settings()
    openai_req = RequestTranslator.translate(request, default_model=settings.default_openai_model)
    openai_req.max_tokens = None

    try:
        backend = get_backend()
        logger.info(f"Counting tokens via backend: {backend.get_backend_name()}")
        openai_resp = await backend.create_completion(openai_req)
    except OpenAIAPIError as e:
        raise handle_openai_error(e)

    # Build a Claude‑compatible usage object
    usage = ClaudeUsage(
        input_tokens=openai_resp.usage.prompt_tokens,
        output_tokens=openai_resp.usage.completion_tokens,
    )

    # Return a ClaudeResponse with empty content and the usage info
    return ClaudeResponse(
        id=openai_resp.id,
        type="message",
        role="assistant",
        content=[],
        model=request.model,
        stop_reason=None,
        usage=usage,
    )

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_claude_api_key)):
    """
    List available models (Claude-compatible format)
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-5-sonnet-20240620",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-haiku-20240307",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            },
        ],
    }


import uvicorn

import argparse


def run_server():
    parser = argparse.ArgumentParser(description='Run Claude-to-OpenAI API Forwarder')
    parser.add_argument('--port', type=int, help='Port to bind the API to')
    args = parser.parse_args()

    settings = get_settings()
    port = args.port if args.port is not None else settings.port
    uvicorn.run(
        "claude_to_openai_forwarder.app:app",
        host=settings.host,
        port=port,
        log_level=settings.log_level.lower(),
        reload=True,
    )

if __name__ == "__main__":
    run_server()
