from fastapi import Header, HTTPException, Request, status


async def verify_api_key(
    request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """Simple API-key auth suitable for demos and internal services."""
    expected = getattr(request.app.state, "settings", None)
    api_key = expected.api_key if expected is not None else None
    if api_key is None or x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
