"""
Message Bus Module

Enables asynchronous communication between agents using a publish-subscribe pattern.

KEY CONCEPTS:
1. Pub/Sub Pattern - Agents publish messages, interested parties subscribe
2. Message Types - Different categories of messages (request, response, broadcast)
3. Topics - Named channels for message routing
4. Async Processing - Non-blocking message handling

WHY A MESSAGE BUS?
- Decouples agents (they don't need to know about each other directly)
- Enables parallel processing
- Makes the system more scalable
- Allows for message persistence and replay

LEARNING POINTS:
- This is a simplified in-memory implementation
- Production systems might use Redis, RabbitMQ, or Kafka
- The pattern is the same regardless of the underlying transport
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    Categories of messages in the system.

    REQUEST - Agent asking another agent to do something
    RESPONSE - Reply to a request
    BROADCAST - Message to all agents
    EVENT - Notification of something that happened
    ERROR - Error notification
    """
    REQUEST = auto()
    RESPONSE = auto()
    BROADCAST = auto()
    EVENT = auto()
    ERROR = auto()


@dataclass
class Message:
    """
    A message passed between agents.

    Messages are immutable and contain all information needed for routing.

    Example:
        msg = Message(
            sender="web_agent",
            receiver="research_agent",
            content={"url": "https://example.com", "html": "..."},
            msg_type=MessageType.RESPONSE
        )
    """
    sender: str
    receiver: str  # Use "*" for broadcast
    content: Any
    msg_type: MessageType
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Links request/response pairs
    topic: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_response(self, content: Any) -> 'Message':
        """
        Create a response message to this message.

        Automatically sets the correlation_id for tracking.
        """
        return Message(
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            msg_type=MessageType.RESPONSE,
            correlation_id=self.id,
            topic=self.topic
        )


# Type alias for message handlers
MessageHandler = Callable[[Message], Awaitable[None]]


class MessageBus:
    """
    Central message routing system for the multi-agent architecture.

    PATTERN: Publisher-Subscriber (Pub/Sub)

    Publishers:
        - Any agent can publish messages
        - Messages go to specific receivers or broadcast to all

    Subscribers:
        - Agents subscribe to receive messages addressed to them
        - Can also subscribe to topics for broadcast messages

    USAGE:
        bus = MessageBus()

        # Subscribe to messages
        async def handler(msg):
            print(f"Received: {msg.content}")

        bus.subscribe("my_agent", handler)

        # Publish a message
        await bus.publish(Message(
            sender="other_agent",
            receiver="my_agent",
            content="Hello!",
            msg_type=MessageType.REQUEST
        ))
    """

    def __init__(self):
        # Maps agent name -> their message handler
        self._subscribers: Dict[str, MessageHandler] = {}

        # Maps topic -> set of subscribed agent names
        self._topic_subscribers: Dict[str, Set[str]] = {}

        # Message history for debugging/replay
        self._message_history: List[Message] = []
        self._max_history = 1000

        # Pending responses (for request-response pattern)
        self._pending_responses: Dict[str, asyncio.Future] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("MessageBus initialized")

    def subscribe(self, agent_name: str, handler: MessageHandler) -> None:
        """
        Subscribe an agent to receive messages.

        Args:
            agent_name: The agent's identifier
            handler: Async function to call when message arrives
        """
        self._subscribers[agent_name] = handler
        logger.debug(f"Agent subscribed: {agent_name}")

    def unsubscribe(self, agent_name: str) -> None:
        """
        Remove an agent's subscription.
        """
        if agent_name in self._subscribers:
            del self._subscribers[agent_name]
            logger.debug(f"Agent unsubscribed: {agent_name}")

    def subscribe_topic(self, agent_name: str, topic: str) -> None:
        """
        Subscribe an agent to a topic for broadcast messages.

        Topics allow grouping agents by interest:
        - "web_events" - All agents interested in web-related events
        - "errors" - All agents that handle errors
        """
        if topic not in self._topic_subscribers:
            self._topic_subscribers[topic] = set()
        self._topic_subscribers[topic].add(agent_name)

    async def publish(self, message: Message) -> None:
        """
        Publish a message to its intended recipient(s).

        Routing logic:
        1. If receiver is "*" -> broadcast to all
        2. If receiver is a topic name prefixed with "#" -> send to topic subscribers
        3. Otherwise -> send to specific agent
        """
        async with self._lock:
            # Store in history
            self._message_history.append(message)
            if len(self._message_history) > self._max_history:
                self._message_history.pop(0)

        logger.debug(f"Message published: {message.sender} -> {message.receiver} [{message.msg_type.name}]")

        # Check if this is a response to a pending request
        if message.correlation_id and message.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)
            return

        # Route the message
        if message.receiver == "*":
            # Broadcast to all subscribers
            await self._broadcast(message)
        elif message.receiver.startswith("#"):
            # Send to topic subscribers
            topic = message.receiver[1:]
            await self._send_to_topic(message, topic)
        else:
            # Send to specific agent
            await self._send_direct(message)

    async def _send_direct(self, message: Message) -> None:
        """Send message to a specific agent."""
        if message.receiver in self._subscribers:
            handler = self._subscribers[message.receiver]
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message in {message.receiver}: {e}")

    async def _broadcast(self, message: Message) -> None:
        """Send message to all subscribers."""
        for agent_name, handler in self._subscribers.items():
            if agent_name != message.sender:  # Don't send to self
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {agent_name}: {e}")

    async def _send_to_topic(self, message: Message, topic: str) -> None:
        """Send message to all topic subscribers."""
        if topic in self._topic_subscribers:
            for agent_name in self._topic_subscribers[topic]:
                if agent_name in self._subscribers and agent_name != message.sender:
                    try:
                        await self._subscribers[agent_name](message)
                    except Exception as e:
                        logger.error(f"Error sending to topic subscriber {agent_name}: {e}")

    async def request(self, message: Message, timeout: float = 30.0) -> Optional[Message]:
        """
        Send a request and wait for a response.

        This implements the request-response pattern on top of pub/sub.

        Args:
            message: The request message
            timeout: How long to wait for response (seconds)

        Returns:
            The response message, or None if timeout
        """
        # Create a future to wait for the response
        future = asyncio.Future()
        self._pending_responses[message.id] = future

        # Send the request
        await self.publish(message)

        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            # Clean up on timeout
            self._pending_responses.pop(message.id, None)
            logger.warning(f"Request timeout: {message.id}")
            return None

    def get_history(self, limit: int = 100, sender: str = None, receiver: str = None) -> List[Message]:
        """
        Get message history, optionally filtered.

        Useful for debugging and auditing.
        """
        history = self._message_history[-limit:]

        if sender:
            history = [m for m in history if m.sender == sender]
        if receiver:
            history = [m for m in history if m.receiver == receiver]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "total_messages": len(self._message_history),
            "subscribers": list(self._subscribers.keys()),
            "topics": {t: list(s) for t, s in self._topic_subscribers.items()},
            "pending_requests": len(self._pending_responses)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """
    Demonstrates basic message bus functionality.
    """
    bus = MessageBus()

    # Create message handlers
    received_messages = []

    async def agent_a_handler(msg: Message):
        print(f"Agent A received: {msg.content}")
        received_messages.append(msg)
        # Send a response
        response = msg.create_response(f"Got it: {msg.content}")
        await bus.publish(response)

    async def agent_b_handler(msg: Message):
        print(f"Agent B received: {msg.content}")
        received_messages.append(msg)

    # Subscribe agents
    bus.subscribe("agent_a", agent_a_handler)
    bus.subscribe("agent_b", agent_b_handler)

    # Direct message
    await bus.publish(Message(
        sender="agent_b",
        receiver="agent_a",
        content="Hello Agent A!",
        msg_type=MessageType.REQUEST
    ))

    # Broadcast
    await bus.publish(Message(
        sender="system",
        receiver="*",
        content="System announcement",
        msg_type=MessageType.BROADCAST
    ))

    print(f"Total messages received: {len(received_messages)}")


if __name__ == "__main__":
    asyncio.run(example_usage())
