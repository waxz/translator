from fastapi import (
    FastAPI,
    Response,
    Depends,
    HTTPException,
    status,
    Request as FastAPIRequest,
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError


from pydantic import ValidationError
from contextlib import asynccontextmanager, suppress
import asyncio
import logging
import traceback
import json
import time
import uuid
from typing import AsyncGenerator

from claude_to_openai_forwarder.config import get_settings
from claude_to_openai_forwarder.models.claude import (
    ClaudeRequest,
    ClaudeResponse,
    ClaudeUsage,
    ClaudeContentBlock,
)
from claude_to_openai_forwarder.backends import get_backend, get_backend_name
from claude_to_openai_forwarder.translators.request import RequestTranslator
from claude_to_openai_forwarder.translators.response import ResponseTranslator
from claude_to_openai_forwarder.translators.streaming import StreamingTranslator
from claude_to_openai_forwarder.middleware.auth import (
    verify_claude_api_key,
    rotate_by_key_string,
)
from claude_to_openai_forwarder.utils.exceptions import (
    handle_openai_error,
    OpenAIAPIError,
)
from claude_to_openai_forwarder.utils.rate_limit import check_rate_limit

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    settings = get_settings()
    backend_name = get_backend_name()
    logger.info("=" * 80)
    logger.info("Starting Claude-to-OpenAI API Forwarder")
    logger.info(f"Backend: {backend_name}")
    if settings.model_provider:
        logger.info(f"Model Provider: {settings.model_provider}")
    logger.info(f"OpenAI Base URL: {settings.openai_base_url}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info("=" * 80)
    yield
    logger.info("Shutting down...")
    # Clean up backend resources
    try:
        backend = get_backend()
        if hasattr(backend, 'close'):
            await backend.close()
            logger.info("Closed backend resources")
    except Exception as e:
        logger.warning(f"Error closing backend: {e}")


app = FastAPI(
    title="Claude-to-OpenAI API Forwarder",
    description="Forwards Claude API requests to OpenAI-compatible backends",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: FastAPIRequest, exc: RequestValidationError
):
    """
    Handle validation errors with detailed information.

    This handler logs the error details and returns a Claude-compatible error response.
    """
    logger.error("=" * 80)
    logger.error("REQUEST VALIDATION ERROR")
    logger.error(f"Error details: {exc.errors()}")

    # Try to get the raw body
    try:
        body = await request.body()
        body_str = body.decode()
        logger.error(f"Request body that failed validation:")
        logger.error(body_str[:2000])

        # Try to parse as JSON for better display
        try:
            body_dict = json.loads(body_str)
            logger.error("Parsed JSON:")
            logger.error(json.dumps(body_dict, indent=2)[:2000])
        except:
            pass
    except Exception as e:
        logger.error(f"Could not read request body: {e}")

    logger.error("=" * 80)

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
    """
    Return Claude-compatible error bodies without FastAPI's detail wrapper.

    This handler logs the error and returns a JSON response with the error details.
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")

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
    """
    Global exception handler for better error messages.

    This handler logs the error and returns a JSON response with the error details.
    """
    logger.error("=" * 80)
    logger.error("UNHANDLED EXCEPTION")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())
    logger.error("=" * 80)

    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": f"Internal server error: {type(exc).__name__}: {str(exc)}",
            },
        },
    )


@app.get("/rotate")
async def rotate(
    # api_key: str = Depends(verify_claude_api_key),
):

    settings = get_settings()
    next_key = rotate_by_key_string(
        settings.openai_api_key, settings.openai_api_key_list
    )
    settings.openai_api_key = next_key
    return Response(
        content=None, headers={"X-Model-Ready": "true", "X-Rate-Limit-Remaining": "500"}
    )


@app.post("/v1/messages")
async def create_message(
    request: FastAPIRequest,
    api_key: str = Depends(verify_claude_api_key),
):
    """
    Claude Messages API endpoint.

    Accepts Claude-formatted requests and returns Claude-formatted responses.
    """
    request_start = time.time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        settings = get_settings()
        check_rate_limit(api_key, settings.rate_limit_rpm)
        client = get_backend()

        logger.info("\n" + "=" * 80)
        logger.info(f"NEW REQUEST [{request_id}]")
        logger.info(f"Backend: {client.get_backend_name()}")
        logger.info("=" * 80)

        # Parse body
        body = await request.body()
        body_str = body.decode()

        try:
            body_dict = json.loads(body_str)
        except json.JSONDecodeError as e:
            logger.error(f"[{request_id}] Invalid JSON: {e}")
            logger.error(f"Body preview: {body_str[:500]}")
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

        # Log incoming request details
        # log_request_details(body_dict, "CLAUDE REQUEST")

        # Parse into Pydantic model
        try:
            request = ClaudeRequest(**body_dict)
        except ValidationError as e:
            logger.error(f"[{request_id}] Pydantic validation error: {e}")
            logger.error(f"Validation errors: {e.errors()}")
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

        # Translate Claude request to OpenAI format
        logger.info(f"[{request_id}] Translating to OpenAI format...")
        openai_request = RequestTranslator.translate(request)

        # Handle streaming
        if request.stream:
            logger.info(f"[{request_id}] Starting streaming response")

            async def generate() -> AsyncGenerator[str, None]:
                openai_stream = None
                translated_stream = None
                try:
                    openai_stream = client.create_completion_stream(openai_request)
                    translated_stream = StreamingTranslator.translate_stream(openai_stream)
                    event_count = 0
                    async for event in translated_stream:
                        event_count += 1
                        if event_count % 100 == 0:
                            logger.debug(
                                f"[{request_id}] Streamed {event_count} events"
                            )
                        yield event
                    logger.info(f"[{request_id}] Stream complete: {event_count} events")
                except asyncio.CancelledError:
                    logger.info(f"[{request_id}] Streaming request cancelled")
                    return
                except Exception as e:
                    logger.error(f"[{request_id}] Streaming error: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                finally:
                    if translated_stream is not None and hasattr(
                        translated_stream, "aclose"
                    ):
                        with suppress(RuntimeError):
                            await translated_stream.aclose()
                    if openai_stream is not None and hasattr(openai_stream, "aclose"):
                        with suppress(RuntimeError):
                            await openai_stream.aclose()

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                },
            )

        # Handle non-streaming
        logger.info(f"[{request_id}] Calling backend...")
        backend_start = time.time()
        openai_response = await client.create_completion(openai_request)
        backend_duration = time.time() - backend_start
        logger.info(f"[{request_id}] Backend responded in {backend_duration:.2f}s")

        # Log OpenAI response
        # log_response_details(openai_response, "OPENAI RESPONSE")

        # Translate OpenAI response to Claude format
        logger.info(f"[{request_id}] Translating to Claude format...")
        claude_response = ResponseTranslator.translate(openai_response)

        # Log Claude response
        # log_response_details(claude_response, "CLAUDE RESPONSE")

        # Serialize response
        serialized = claude_response.model_dump(exclude_none=True)
        
        # Final validation - check for tool_use blocks
        content_blocks = serialized.get("content", [])
        has_tool_use = any(b.get("type") == "tool_use" for b in content_blocks)
        logger.info(
            f"[{request_id}] Final response has {len(content_blocks)} content blocks, tool_use={has_tool_use}"
        )

        if has_tool_use:
            logger.info(f"[{request_id}] Tool use details:")
            for i, block in enumerate(content_blocks):
                if block.get("type") == "tool_use":
                    logger.info(f"  [{i}] {block.get('name')} (id: {block.get('id')})")
                    logger.info(
                        f"      input: {json.dumps(block.get('input', {}))[:200]}..."
                    )

        request_duration = time.time() - request_start
        logger.info(f"[{request_id}] Request completed in {request_duration:.2f}s")
        logger.info("=" * 80 + "\n")

        return claude_response

    except HTTPException:
        raise
    except OpenAIAPIError as e:
        logger.error(f"[{request_id}] OpenAI API error: {e}")
        raise handle_openai_error(e)
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {"type": "internal_error", "message": str(e)},
            },
        )


@app.head("/")
async def get_status():
    """
    Returns headers indicating model availability and version
    without running any inference.
    """
    return Response(
        content=None,
        headers={
            "X-Model-Ready": "true",
            "X-Model-Version": "llama-3.1-70b",
            "X-Rate-Limit-Remaining": "500",
        },
    )


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    raw_request: FastAPIRequest,
    api_key: str = Depends(verify_claude_api_key),
):
    """
    Token-counting endpoint – mirrors the official Claude API.

    Returns the number of input tokens for the supplied messages.
    """
    try:
        body = await raw_request.body()
        body_dict = json.loads(body.decode())
        request = ClaudeRequest(**body_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid request for count_tokens: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Invalid request: {str(e)}",
                },
            },
        )

    settings = get_settings()
    openai_req = RequestTranslator.translate(request)
    openai_req.max_tokens = 1  # Minimal tokens for counting

    try:
        backend = get_backend()
        logger.info(f"Counting tokens via backend: {backend.get_backend_name()}")
        openai_resp = await backend.create_completion(openai_req)
    except OpenAIAPIError as e:
        raise handle_openai_error(e)

    usage = ClaudeUsage(
        input_tokens=openai_resp.usage.prompt_tokens,
        output_tokens=0,  # We didn't generate anything
    )

    return ClaudeResponse(
        id=openai_resp.id,
        type="message",
        role="assistant",
        content=[],
        model=request.model,
        stop_reason=None,
        usage=usage,
    ).model_dump(exclude_none=True)


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_claude_api_key)):
    """
    List available models (Claude-compatible format).

    Returns a list of models in the Claude-compatible format.
    """

    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1729641600,
                "owned_by": "anthropic",
            },
            # ...
        ],
    }


def run_server():
    """
    Run the server with command-line arguments.

    This function parses command-line arguments and runs the server using uvicorn.
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Claude-to-OpenAI API Forwarder")
    parser.add_argument("--host", type=str, help="Host to bind the API to")
    parser.add_argument("--port", type=int, help="Port to bind the API to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args = parser.parse_args()

    settings = get_settings()

    uvicorn.run(
        "claude_to_openai_forwarder.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=args.log_level or settings.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    run_server()
