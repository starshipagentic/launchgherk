Please generate a Python implementation that:

Uses MVVM to separate concerns:

ViewModels that represent workflow states and transitions
Models that encapsulate domain concepts but remain hidden from API
Views (API endpoints) that expose only workflow actions


Implements HATEOAS by:

Including available actions in each response
Providing context-aware navigation links
Specifying required fields and conditions for actions


Preserves key DDD concepts through:

Value Objects for immutable concepts
Entities for domain objects with identity
Workflow states as explicit domain concepts


Exposes customer journeys by:

Defining clear workflow states
Showing valid state transitions
Hiding internal model implementations
Focusing on business processes rather than CRUD



Please structure the code with:

Domain models (hidden from API)
ViewModels (one per workflow state)
API endpoints (reflecting workflow actions)
HATEOAS responses
Example usage

Focus on making the API guide users through the workflow rather than exposing data structures.

Here is an example of a previous project, but a different subject.  You will be generating code for a different domain, so just learn and understand the patterns and the detail you are expected to generate.

Start Example:
``` python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncio
from functools import wraps

# Domain Models (Hidden from API)
@dataclass(frozen=True)
class Money:
    amount: float
    currency: str

@dataclass
class OrderItem:
    id: UUID = field(default_factory=uuid4)
    product_id: UUID = field(default_factory=uuid4)
    quantity: int = 0
    price: Money = field(default_factory=lambda: Money(0.0, "USD"))

@dataclass
class Order:
    id: UUID = field(default_factory=uuid4)
    customer_id: UUID
    status: str = "DRAFT"
    items: List[OrderItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

# Workflow States and Transitions
class OrderWorkflowState(str, Enum):
    SHOPPING = "shopping"
    REVIEWING = "reviewing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

# Workflow-Centric ViewModels
class ShoppingCartViewModel:
    """Handles the shopping phase of the order workflow"""
    def __init__(self):
        self._draft_orders: Dict[UUID, Order] = {}

    async def start_shopping_session(self, customer_id: UUID) -> dict:
        """Begin a new shopping session"""
        order = Order(customer_id=customer_id)
        self._draft_orders[order.id] = order
        return self._create_shopping_response(order)

    async def add_to_cart(self, session_id: UUID, product_id: UUID, 
                         quantity: int, price: float) -> dict:
        """Add item to shopping cart"""
        order = self._draft_orders.get(session_id)
        if not order or order.status != "DRAFT":
            raise HTTPException(status_code=404, detail="Shopping session not found")
        
        item = OrderItem(
            product_id=product_id,
            quantity=quantity,
            price=Money(price, "USD")
        )
        order.items.append(item)
        return self._create_shopping_response(order)

    def _create_shopping_response(self, order: Order) -> dict:
        """Create HATEOAS response for shopping workflow"""
        return {
            "workflow_state": OrderWorkflowState.SHOPPING,
            "session_id": str(order.id),
            "cart_summary": {
                "item_count": len(order.items),
                "total_amount": sum(item.price.amount * item.quantity for item in order.items)
            },
            "available_actions": {
                "add_item": {
                    "href": f"/api/shopping/{order.id}/items",
                    "method": "POST",
                    "required_fields": ["product_id", "quantity", "price"]
                },
                "proceed_to_review": {
                    "href": f"/api/shopping/{order.id}/review",
                    "method": "POST",
                    "conditions": ["cart_not_empty"]
                },
                "abandon_cart": {
                    "href": f"/api/shopping/{order.id}",
                    "method": "DELETE"
                }
            }
        }

class OrderReviewViewModel:
    """Handles the review phase of the order workflow"""
    def __init__(self, shopping_cart_vm: ShoppingCartViewModel):
        self._shopping_cart_vm = shopping_cart_vm
        self._reviews: Dict[UUID, Dict] = {}

    async def start_review(self, session_id: UUID) -> dict:
        """Transform shopping cart into review state"""
        order = self._shopping_cart_vm._draft_orders.get(session_id)
        if not order:
            raise HTTPException(status_code=404, detail="Shopping session not found")
        
        review_data = self._calculate_review_data(order)
        self._reviews[session_id] = review_data
        return self._create_review_response(session_id, review_data)

    def _calculate_review_data(self, order: Order) -> dict:
        """Calculate totals and prepare review data"""
        subtotal = sum(item.price.amount * item.quantity for item in order.items)
        tax = subtotal * 0.1  # Example tax calculation
        return {
            "order_id": order.id,
            "items": [
                {
                    "product_id": str(item.product_id),
                    "quantity": item.quantity,
                    "unit_price": item.price.amount,
                    "total": item.price.amount * item.quantity
                } for item in order.items
            ],
            "subtotal": subtotal,
            "tax": tax,
            "total": subtotal + tax
        }

    def _create_review_response(self, session_id: UUID, review_data: dict) -> dict:
        """Create HATEOAS response for review workflow"""
        return {
            "workflow_state": OrderWorkflowState.REVIEWING,
            "session_id": str(session_id),
            "review_details": review_data,
            "available_actions": {
                "modify_cart": {
                    "href": f"/api/shopping/{session_id}",
                    "method": "GET",
                    "description": "Return to shopping"
                },
                "proceed_to_checkout": {
                    "href": f"/api/checkout/{session_id}/begin",
                    "method": "POST",
                    "required_fields": []
                }
            }
        }

class CheckoutViewModel:
    """Handles the checkout/confirmation phase"""
    def __init__(self, review_vm: OrderReviewViewModel):
        self._review_vm = review_vm
        self._checkouts: Dict[UUID, Dict] = {}

    async def start_checkout(self, session_id: UUID) -> dict:
        """Begin checkout process"""
        review_data = self._review_vm._reviews.get(session_id)
        if not review_data:
            raise HTTPException(status_code=404, detail="Review session not found")
        
        self._checkouts[session_id] = {
            "review_data": review_data,
            "payment_status": "pending"
        }
        
        return self._create_checkout_response(session_id)

    def _create_checkout_response(self, session_id: UUID) -> dict:
        """Create HATEOAS response for checkout workflow"""
        checkout_data = self._checkouts[session_id]
        return {
            "workflow_state": OrderWorkflowState.CONFIRMING,
            "session_id": str(session_id),
            "order_summary": checkout_data["review_data"],
            "available_actions": {
                "add_payment_method": {
                    "href": f"/api/checkout/{session_id}/payment",
                    "method": "POST",
                    "required_fields": ["payment_method", "billing_address"]
                },
                "return_to_review": {
                    "href": f"/api/review/{session_id}",
                    "method": "GET"
                },
                "confirm_order": {
                    "href": f"/api/checkout/{session_id}/confirm",
                    "method": "POST",
                    "conditions": ["payment_method_added"]
                }
            }
        }

# API Layer (Workflow-Oriented Endpoints)
app = FastAPI()

# Initialize ViewModels with dependencies
shopping_cart_vm = ShoppingCartViewModel()
order_review_vm = OrderReviewViewModel(shopping_cart_vm)
checkout_vm = CheckoutViewModel(order_review_vm)

# Shopping Cart Workflow
@app.post("/api/shopping/start")
async def start_shopping(customer_id: UUID):
    """Start a new shopping session"""
    response = await shopping_cart_vm.start_shopping_session(customer_id)
    return JSONResponse(content=response)

@app.post("/api/shopping/{session_id}/items")
async def add_to_cart(
    session_id: UUID,
    product_id: UUID,
    quantity: int,
    price: float
):
    """Add item to shopping cart"""
    response = await shopping_cart_vm.add_to_cart(
        session_id, product_id, quantity, price
    )
    return JSONResponse(content=response)

# Review Workflow
@app.post("/api/shopping/{session_id}/review")
async def start_review(session_id: UUID):
    """Transform cart into review state"""
    response = await order_review_vm.start_review(session_id)
    return JSONResponse(content=response)

# Checkout Workflow
@app.post("/api/checkout/{session_id}/begin")
async def start_checkout(session_id: UUID):
    """Begin checkout process"""
    response = await checkout_vm.start_checkout(session_id)
    return JSONResponse(content=response)

# Example Usage
if __name__ == "__main__":
    async def example_workflow():
        # Start shopping
        customer_id = uuid4()
        shopping_response = await shopping_cart_vm.start_shopping_session(customer_id)
        session_id = UUID(shopping_response["session_id"])
        
        # Add items
        cart_response = await shopping_cart_vm.add_to_cart(
            session_id,
            uuid4(),
            2,
            29.99
        )
        
        # Review
        review_response = await order_review_vm.start_review(session_id)
        
        # Checkout
        checkout_response = await checkout_vm.start_checkout(session_id)
        
        print("Shopping:", shopping_response)
        print("Cart:", cart_response)
        print("Review:", review_response)
        print("Checkout:", checkout_response)
    
    asyncio.run(example_workflow())
  ```
