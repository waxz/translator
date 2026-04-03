from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import json

logger = logging.getLogger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request details for /v1/messages
        if request.url.path == "/v1/messages":
            logger.info(f"=== Raw Request ===")
            logger.info(f"Method: {request.method}")
            logger.info(f"URL: {request.url}")
            logger.info(f"Headers: {dict(request.headers)}")
            
            # Read and log body
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body.decode())
                    logger.info(f"Body keys: {list(body_json.keys())}")
                    if 'tools' in body_json:
                        logger.info(f"Tools in request: {len(body_json['tools'])}")
                        logger.info(f"Tools: {json.dumps(body_json['tools'], indent=2)}")
                    else:
                        logger.warning(f"NO TOOLS IN REQUEST!")
                except Exception as e:
                    logger.error(f"Error parsing body: {e}")
            
            # We need to create a new request with the body for downstream
            from starlette.requests import Request as StarletteRequest
            
            async def receive():
                return {"type": "http.request", "body": body}
            
            request = StarletteRequest(request.scope, receive)
        
        response = await call_next(request)
        return response