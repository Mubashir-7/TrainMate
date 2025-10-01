from dataclasses import dataclass, field
import asyncio

@dataclass
class TrainingStatus:
    """A simple thread-safe class to hold the current training status."""
    message: str = "Idle"
    running: bool = False
    # An asyncio.Event to signal updates to any listeners
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def set(self, message: str, running: bool):
        self.message = message
        self.running = running
        self._event.set()  # Signal that an update is available
        self._event.clear() # Reset the event for the next update

    async def wait(self):
        """Wait until the status is updated."""
        await self._event.wait()

# Singleton instance to be shared across the app
training_status = TrainingStatus()