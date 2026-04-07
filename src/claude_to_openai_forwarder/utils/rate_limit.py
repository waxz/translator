import time
from collections import defaultdict
from fastapi import HTTPException

# Store timestamps: { api_key: [timestamp1, timestamp2, ...] }
visit_records = defaultdict(list)

def check_rate_limit(identifier: str, rpm_limit: int):
    now = time.time()
    # Clean up records older than 60 seconds
    visit_records[identifier] = [t for t in visit_records[identifier] if now - t < 60]
    
    if len(visit_records[identifier]) >= rpm_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": f"Rate limit exceeded. Max {rpm_limit} requests per minute."
                }
            }
        )
    
    # Record the current visit
    visit_records[identifier].append(now)
