# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Customer Support Ticket Resolution Environment.

Simulates a customer support agent handling support tickets using a knowledge base and customer data.
"""

from typing import List, Optional, Dict, Any, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FinalAction(Action):
    """Action for the Customer Support environment."""

    action_type: Literal["search_kb", "get_customer_details", "send_message", "resolve_ticket"] = Field(
        ..., description="The type of action to perform"
    )
    query: Optional[str] = Field(None, description="Search query for knowledge base")
    customer_id: Optional[str] = Field(None, description="Customer ID for retrieving details")
    message: Optional[str] = Field(None, description="Message to send to the customer")
    resolution: Optional[str] = Field(None, description="Summary of resolution when closing the ticket")


class FinalObservation(Observation):
    """Observation from the Customer Support environment."""

    ticket_id: str = Field(..., description="The unique ID of the ticket")
    ticket_description: str = Field(..., description="The problem description from the customer")
    customer_id: str = Field(..., description="The customer's ID")
    search_results: List[str] = Field(default_factory=list, description="Results from knowledge base search")
    customer_details: Dict[str, Any] = Field(default_factory=dict, description="Retrieved customer information")
    last_response: str = Field(default="", description="The response from the last action taken")
    is_closed: bool = Field(default=False, description="Whether the ticket has been closed")
    reward: float = Field(default=0.001, description="The reward for the current step")
    done: bool = Field(default=False, description="Whether the episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")
    task_score: float = Field(default=0.001, description="The current score of the task (strictly 0-1)")


class ShoppingCartAction(Action):
    """Action for the Shopping Cart environment."""

    action_type: Literal["add_item", "remove_item", "update_quantity", "checkout", "view_cart"] = Field(
        ..., description="The type of action to perform"
    )
    item_id: Optional[str] = Field(None, description="ID of the item to add, remove, or update")
    quantity: Optional[int] = Field(None, description="Quantity of the item")


class ShoppingCartObservation(Observation):
    """Observation from the Shopping Cart environment."""

    cart_id: str = Field(..., description="The unique ID of the shopping cart")
    items_in_cart: List[Dict[str, Any]] = Field(default_factory=list, description="List of items currently in the cart")
    total_price: float = Field(default=0.0, description="Total price of items in the cart")
    last_action_status: str = Field(default="", description="Status message from the last action")
    is_checked_out: bool = Field(default=False, description="Whether the cart has been checked out")
    reward: float = Field(default=0.001, description="The reward for the current step")
    done: bool = Field(default=False, description="Whether the episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")
    task_score: float = Field(default=0.001, description="The current score of the task (strictly 0-1)")
