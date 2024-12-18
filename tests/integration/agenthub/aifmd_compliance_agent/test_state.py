"""Test State class for integration tests."""
from typing import Any, Dict, List, Optional

from openhands.events.observation import Observation
from openhands.events.action import Action

class TestState:
    """Test State class that mimics the behavior of the real State class."""

    def __init__(self):
        self.observations: List[Observation] = []
        self.user_messages: List[str] = []
        self.actions: List[Action] = []
        self.max_iterations = 10
        self.iteration = 0

    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the state."""
        self.observations.append(observation)

    def add_user_message(self, message: str) -> None:
        """Add a user message to the state."""
        self.user_messages.append(message)

    def add_action(self, action: Action) -> None:
        """Add an action to the state."""
        self.actions.append(action)

    def get_last_observation(self) -> Optional[Observation]:
        """Get the last observation."""
        return self.observations[-1] if self.observations else None

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        return self.user_messages[-1] if self.user_messages else None

    def get_last_action(self) -> Optional[Action]:
        """Get the last action."""
        return self.actions[-1] if self.actions else None