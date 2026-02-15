"""
Stripe Webhook Handler (Phase 5 - Payments)

Handles Stripe webhook events for payment processing.
"""

from fastapi import APIRouter, Request, HTTPException
import structlog

from ..features.payments import payment_service

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    try:
        body = await request.body()
        signature = request.headers.get("Stripe-Signature", "")
        
        result = await payment_service.handle_webhook(body, signature)
        
        if "error" in result:
            logger.error("webhook_error", error=result["error"])
            # Return 200 anyway to prevent Stripe from retrying
            return {"received": True, "error": result["error"]}
        
        logger.info(
            "webhook_processed",
            event_type=result.get("event_type"),
            session_id=result.get("session_id"),
        )
        
        return {"received": True, **result}
        
    except Exception as e:
        logger.error("webhook_exception", error=str(e))
        # Return 200 to prevent retries
        return {"received": True, "error": str(e)}
