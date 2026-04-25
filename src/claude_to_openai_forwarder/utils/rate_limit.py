import time
from collections import defaultdict, deque
from fastapi import HTTPException

# Store timestamps: { api_key: deque([timestamp1, timestamp2, ...]) }
# Using deque for O(1) popleft operations and automatic oldest-first removal
visit_records = defaultdict(deque)
_last_cleanup = time.time()
_cleanup_interval = 300  # Clean up old identifiers every 5 minutes
_record_ttl = 60  # Keep records for 60 seconds


def check_rate_limit(identifier: str, rpm_limit: int):
    """
    Check if identifier exceeds rate limit.
    
    Enforces up to rpm_limit requests per minute.
    Automatically cleans up records older than 60 seconds.
    """
    global _last_cleanup
    
    now = time.time()
    
    # Periodic cleanup of old identifier entries (prevent unbounded dict growth)
    if now - _last_cleanup > _cleanup_interval:
        _cleanup_old_identifiers(now)
        _last_cleanup = now
    
    # Remove records older than 60 seconds for this identifier
    records = _prune_identifier_records(identifier, now)
    
    if len(records) >= rpm_limit:
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
    records.append(now)


def _cleanup_old_identifiers(current_time: float):
    """
    Remove identifier entries that have no recent records.
    Prevents unbounded growth of the visit_records dictionary.
    """
    to_remove = []
    for identifier in list(visit_records.keys()):
        records = _prune_identifier_records(identifier, current_time)
        if not records:
            to_remove.append(identifier)
    
    for identifier in to_remove:
        del visit_records[identifier]


def _prune_identifier_records(identifier: str, current_time: float):
    records = visit_records[identifier]
    while records and current_time - records[0] >= _record_ttl:
        records.popleft()
    return records
