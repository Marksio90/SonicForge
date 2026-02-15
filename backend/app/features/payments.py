"""
Stripe Payment Integration (Phase 5 - Monetization)

Implements subscription-based premium features:
- Premium subscription plans
- One-time purchases (credits)
- Checkout sessions
- Payment webhooks
- Transaction tracking
"""

import os
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import structlog

load_dotenv()

logger = structlog.get_logger(__name__)

# Premium subscription plans (server-side defined - NEVER accept from frontend)
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0.0,
        "features": {
            "generations_per_day": 3,
            "priority_queue": False,
            "exclusive_genres": False,
            "hd_audio": False,
            "download_tracks": False,
            "api_access": False,
        },
    },
    "pro": {
        "name": "Pro",
        "price": 9.99,
        "features": {
            "generations_per_day": 20,
            "priority_queue": True,
            "exclusive_genres": False,
            "hd_audio": True,
            "download_tracks": True,
            "api_access": False,
        },
    },
    "premium": {
        "name": "Premium",
        "price": 19.99,
        "features": {
            "generations_per_day": 100,
            "priority_queue": True,
            "exclusive_genres": True,
            "hd_audio": True,
            "download_tracks": True,
            "api_access": True,
        },
    },
    "enterprise": {
        "name": "Enterprise",
        "price": 99.99,
        "features": {
            "generations_per_day": -1,  # Unlimited
            "priority_queue": True,
            "exclusive_genres": True,
            "hd_audio": True,
            "download_tracks": True,
            "api_access": True,
            "dedicated_support": True,
            "custom_branding": True,
        },
    },
}

# Credit packages (one-time purchases)
CREDIT_PACKAGES = {
    "starter": {"credits": 10, "price": 4.99},
    "basic": {"credits": 25, "price": 9.99},
    "standard": {"credits": 60, "price": 19.99},
    "pro": {"credits": 150, "price": 39.99},
}


class SubscriptionPlan(BaseModel):
    """Subscription plan schema."""
    plan_id: str
    name: str
    price: float
    features: dict


class UserSubscription(BaseModel):
    """User subscription status."""
    user_id: str
    plan_id: str
    plan_name: str
    status: str  # active, cancelled, expired
    features: dict
    credits_remaining: int
    generations_today: int
    started_at: datetime
    expires_at: Optional[datetime] = None


class CheckoutRequest(BaseModel):
    """Checkout request schema."""
    plan_id: Optional[str] = None  # For subscriptions
    credit_package: Optional[str] = None  # For one-time purchases
    origin_url: str  # Frontend origin for success/cancel URLs


class CheckoutResponse(BaseModel):
    """Checkout response schema."""
    checkout_url: str
    session_id: str


class PaymentTransaction(BaseModel):
    """Payment transaction record."""
    transaction_id: str
    session_id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    amount: float
    currency: str = "usd"
    product_type: str  # subscription, credits
    product_id: str
    status: str  # initiated, pending, paid, failed, expired
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


# In-memory storage (use MongoDB in production)
_user_subscriptions: dict[str, dict] = {}
_user_credits: dict[str, int] = {}
_payment_transactions: dict[str, PaymentTransaction] = {}
_daily_generations: dict[str, dict] = {}  # user_id -> {date: count}


class PaymentService:
    """Service for handling payments and subscriptions."""
    
    def __init__(self):
        self.stripe_api_key = os.environ.get("STRIPE_API_KEY", "")
        self._stripe_checkout = None
    
    async def initialize(self):
        """Initialize Stripe client."""
        if not self.stripe_api_key:
            logger.warning("stripe_api_key_not_configured")
            return
        
        try:
            from emergentintegrations.payments.stripe.checkout import StripeCheckout
            self._stripe_checkout = StripeCheckout
            logger.info("stripe_initialized")
        except ImportError:
            logger.error("emergentintegrations_not_installed")
    
    def get_plans(self) -> list[SubscriptionPlan]:
        """Get all available subscription plans."""
        return [
            SubscriptionPlan(
                plan_id=plan_id,
                name=plan["name"],
                price=plan["price"],
                features=plan["features"],
            )
            for plan_id, plan in SUBSCRIPTION_PLANS.items()
        ]
    
    def get_credit_packages(self) -> dict:
        """Get all available credit packages."""
        return CREDIT_PACKAGES
    
    async def create_checkout_session(
        self,
        user_id: str,
        plan_id: Optional[str] = None,
        credit_package: Optional[str] = None,
        origin_url: str = "",
    ) -> Optional[CheckoutResponse]:
        """Create a Stripe checkout session."""
        if not self.stripe_api_key:
            logger.error("stripe_not_configured")
            return None
        
        # Determine product and price (SERVER-SIDE ONLY)
        if plan_id:
            if plan_id not in SUBSCRIPTION_PLANS:
                raise ValueError(f"Invalid plan: {plan_id}")
            plan = SUBSCRIPTION_PLANS[plan_id]
            amount = plan["price"]
            product_type = "subscription"
            product_id = plan_id
        elif credit_package:
            if credit_package not in CREDIT_PACKAGES:
                raise ValueError(f"Invalid credit package: {credit_package}")
            package = CREDIT_PACKAGES[credit_package]
            amount = package["price"]
            product_type = "credits"
            product_id = credit_package
        else:
            raise ValueError("Must specify plan_id or credit_package")
        
        if amount <= 0:
            raise ValueError("Cannot create checkout for free plan")
        
        # Build URLs from frontend origin
        success_url = f"{origin_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{origin_url}/payment/cancel"
        
        try:
            from emergentintegrations.payments.stripe.checkout import (
                StripeCheckout,
                CheckoutSessionRequest,
            )
            
            # Initialize Stripe
            webhook_url = f"{origin_url}/api/webhook/stripe"
            stripe_checkout = StripeCheckout(
                api_key=self.stripe_api_key,
                webhook_url=webhook_url,
            )
            
            # Create checkout request
            checkout_request = CheckoutSessionRequest(
                amount=float(amount),
                currency="usd",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "user_id": user_id,
                    "product_type": product_type,
                    "product_id": product_id,
                },
            )
            
            # Create session
            session = await stripe_checkout.create_checkout_session(checkout_request)
            
            # Record transaction BEFORE redirect
            import uuid
            transaction = PaymentTransaction(
                transaction_id=str(uuid.uuid4()),
                session_id=session.session_id,
                user_id=user_id,
                amount=amount,
                currency="usd",
                product_type=product_type,
                product_id=product_id,
                status="initiated",
                metadata={"origin_url": origin_url},
            )
            _payment_transactions[session.session_id] = transaction
            
            logger.info(
                "checkout_session_created",
                session_id=session.session_id,
                user_id=user_id,
                product_type=product_type,
                amount=amount,
            )
            
            return CheckoutResponse(
                checkout_url=session.url,
                session_id=session.session_id,
            )
            
        except Exception as e:
            logger.error("checkout_session_failed", error=str(e))
            raise
    
    async def check_payment_status(self, session_id: str) -> Optional[dict]:
        """Check payment status and update transaction."""
        if not self.stripe_api_key:
            return None
        
        try:
            from emergentintegrations.payments.stripe.checkout import StripeCheckout
            
            stripe_checkout = StripeCheckout(
                api_key=self.stripe_api_key,
                webhook_url="",
            )
            
            status = await stripe_checkout.get_checkout_status(session_id)
            
            # Update transaction
            if session_id in _payment_transactions:
                transaction = _payment_transactions[session_id]
                old_status = transaction.status
                
                # Only process if not already paid
                if old_status != "paid":
                    transaction.status = status.payment_status
                    transaction.updated_at = datetime.now(timezone.utc)
                    
                    # Process successful payment
                    if status.payment_status == "paid":
                        await self._process_successful_payment(transaction)
                        logger.info(
                            "payment_successful",
                            session_id=session_id,
                            user_id=transaction.user_id,
                        )
            
            return {
                "status": status.status,
                "payment_status": status.payment_status,
                "amount_total": status.amount_total,
                "currency": status.currency,
            }
            
        except Exception as e:
            logger.error("payment_status_check_failed", error=str(e))
            return None
    
    async def _process_successful_payment(self, transaction: PaymentTransaction):
        """Process a successful payment."""
        user_id = transaction.user_id
        
        if transaction.product_type == "subscription":
            # Activate subscription
            plan = SUBSCRIPTION_PLANS.get(transaction.product_id, SUBSCRIPTION_PLANS["free"])
            _user_subscriptions[user_id] = {
                "plan_id": transaction.product_id,
                "plan_name": plan["name"],
                "status": "active",
                "features": plan["features"],
                "started_at": datetime.now(timezone.utc),
            }
            logger.info(
                "subscription_activated",
                user_id=user_id,
                plan=transaction.product_id,
            )
            
        elif transaction.product_type == "credits":
            # Add credits
            package = CREDIT_PACKAGES.get(transaction.product_id, {})
            credits = package.get("credits", 0)
            _user_credits[user_id] = _user_credits.get(user_id, 0) + credits
            logger.info(
                "credits_added",
                user_id=user_id,
                credits=credits,
                total=_user_credits[user_id],
            )
    
    async def handle_webhook(self, body: bytes, signature: str) -> dict:
        """Handle Stripe webhook."""
        if not self.stripe_api_key:
            return {"error": "Stripe not configured"}
        
        try:
            from emergentintegrations.payments.stripe.checkout import StripeCheckout
            
            stripe_checkout = StripeCheckout(
                api_key=self.stripe_api_key,
                webhook_url="",
            )
            
            event = await stripe_checkout.handle_webhook(body, signature)
            
            logger.info(
                "webhook_received",
                event_type=event.event_type,
                session_id=event.session_id,
            )
            
            # Process webhook event
            if event.payment_status == "paid":
                if event.session_id in _payment_transactions:
                    transaction = _payment_transactions[event.session_id]
                    if transaction.status != "paid":
                        transaction.status = "paid"
                        await self._process_successful_payment(transaction)
            
            return {
                "event_type": event.event_type,
                "session_id": event.session_id,
                "status": event.payment_status,
            }
            
        except Exception as e:
            logger.error("webhook_processing_failed", error=str(e))
            return {"error": str(e)}
    
    def get_user_subscription(self, user_id: str) -> UserSubscription:
        """Get user's current subscription status."""
        sub = _user_subscriptions.get(user_id, {})
        plan_id = sub.get("plan_id", "free")
        plan = SUBSCRIPTION_PLANS.get(plan_id, SUBSCRIPTION_PLANS["free"])
        
        # Get today's generations
        today = datetime.now(timezone.utc).date().isoformat()
        user_daily = _daily_generations.get(user_id, {})
        generations_today = user_daily.get(today, 0)
        
        return UserSubscription(
            user_id=user_id,
            plan_id=plan_id,
            plan_name=plan["name"],
            status=sub.get("status", "active" if plan_id == "free" else "inactive"),
            features=plan["features"],
            credits_remaining=_user_credits.get(user_id, 0),
            generations_today=generations_today,
            started_at=sub.get("started_at", datetime.now(timezone.utc)),
            expires_at=sub.get("expires_at"),
        )
    
    def can_generate(self, user_id: str) -> tuple[bool, str]:
        """Check if user can generate a track."""
        subscription = self.get_user_subscription(user_id)
        max_generations = subscription.features.get("generations_per_day", 3)
        
        # Unlimited generations
        if max_generations == -1:
            return True, "OK"
        
        # Check daily limit
        if subscription.generations_today >= max_generations:
            # Check if user has credits
            if subscription.credits_remaining > 0:
                return True, "Using credit"
            return False, f"Daily limit reached ({max_generations}/day). Upgrade or buy credits."
        
        return True, "OK"
    
    def use_generation(self, user_id: str) -> bool:
        """Record a generation use."""
        can_gen, reason = self.can_generate(user_id)
        if not can_gen:
            return False
        
        today = datetime.now(timezone.utc).date().isoformat()
        
        if reason == "Using credit":
            _user_credits[user_id] = max(0, _user_credits.get(user_id, 0) - 1)
        else:
            if user_id not in _daily_generations:
                _daily_generations[user_id] = {}
            _daily_generations[user_id][today] = _daily_generations[user_id].get(today, 0) + 1
        
        return True


# Global instance
payment_service = PaymentService()
