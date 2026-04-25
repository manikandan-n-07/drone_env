from .models import DroneAction, DroneObservation, DroneState

# Client import is optional — it requires httpx which may not be installed
# in all runtime environments. The graders must work independently.
try:
    from .client import DroneEnvClient
    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False
    DroneEnvClient = None  # type: ignore

__all__ = [
    "DroneAction",
    "DroneObservation",
    "DroneState",
    "DroneEnvClient",
]
