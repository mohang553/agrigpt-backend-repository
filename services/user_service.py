import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient


class UserService:
    """
    Lightweight MongoDB helper for looking up users by email.
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        # Default to the database/collection visible in Compass
        self.db_name = os.getenv("MONGODB_DB_NAME", "agriculture")
        self.collection_name = os.getenv("MONGODB_USERS_COLLECTION", "users")
        self.client = None
        self.collection = None

        if self.mongo_uri:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Create the Mongo client and cache the collection handle."""
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.collection = self.client[self.db_name][self.collection_name]

    async def ensure_user(self, email: str) -> str:
        """
        Ensure a user record exists for the given email.

        Returns the userType (existing value if present, otherwise "user"
        after inserting a new record).
        """
        if not self.mongo_uri:
            raise RuntimeError("MONGODB_URI is not configured")

        if self.client is None or self.collection is None:
            self._initialize_client()

        doc = await self.collection.find_one({"email": email})
        if doc:
            return doc.get("userType", "user")

        # Insert new user with default type "user"
        await self.collection.insert_one({"email": email, "userType": "user"})
        return "user"


user_service = UserService()

