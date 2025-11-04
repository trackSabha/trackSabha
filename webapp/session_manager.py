#!/usr/bin/env python3
"""
MongoDB-based Session Manager for ADK Chat History
==================================================

Handles persistent storage of chat sessions and message history in MongoDB.
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message in the conversation."""
    message_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary loaded from MongoDB."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ChatSession:
    """Represents a chat session with its metadata and message history."""
    session_id: str
    user_id: str
    created_at: datetime
    last_updated: datetime
    messages: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create from dictionary loaded from MongoDB."""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            messages=[ChatMessage.from_dict(msg) for msg in data.get('messages', [])],
            metadata=data.get('metadata', {})
        )

class MongoSessionManager:
    """MongoDB-based session manager for ADK chat history."""
    
    def __init__(self, connection_string: str = None):
        """Initialize the MongoDB session manager."""
        self.connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable not set")
        
        self.client = None
        self.db = None
        self.sessions_collection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and collections."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=10,
                minPoolSize=1,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.client.admin.command("ping", maxTimeMS=3000)
            
            # Initialize database and collections (use youtube_data as primary DB for indexing)
            # Sessions are stored alongside other data in the project-level database `youtube_data`.
            self.db = self.client["youtube_data"]
            self.sessions_collection = self.db.chat_sessions
            
            # Create indexes for efficient querying
            self._create_indexes()
            
            logger.info("✅ Connected to MongoDB for session management")
            
        except Exception as e:
            logger.error(f"❌ MongoDB session manager connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
    
    def _create_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            # Index on session_id for fast lookups
            self.sessions_collection.create_index("session_id", unique=True)
            
            # Index on user_id for user's session history
            self.sessions_collection.create_index("user_id")
            
            # Index on archived status for filtering
            self.sessions_collection.create_index("archived")
            
            # Compound index for user_id and last_updated for recent sessions
            self.sessions_collection.create_index([
                ("user_id", ASCENDING),
                ("last_updated", DESCENDING)
            ])
            
            # Compound index for active sessions by user
            self.sessions_collection.create_index([
                ("user_id", ASCENDING),
                ("archived", ASCENDING),
                ("last_updated", DESCENDING)
            ])
            
            # Index on created_at for time-based queries
            self.sessions_collection.create_index("created_at")
            
            logger.info("✅ Session indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    async def create_session(self, user_id: str, session_id: str = None, metadata: Dict[str, Any] = None) -> ChatSession:
        """Create a new chat session."""
        try:
            session_id = session_id or str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_updated=now,
                messages=[],
                metadata=metadata
            )
            
            # Store in MongoDB
            self.sessions_collection.insert_one(session.to_dict())
            
            logger.info(f"✅ Created new session: {session_id[:8]}... for user {user_id}")
            return session
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a chat session by ID."""
        try:
            session_data = self.sessions_collection.find_one({"session_id": session_id})
            
            if session_data:
                return ChatSession.from_dict(session_data)
            else:
                logger.warning(f"Session not found: {session_id}")
                return None
                
        except PyMongoError as e:
            logger.error(f"❌ Failed to retrieve session: {e}")
            return None
    
    async def get_user_sessions(self, user_id: str, limit: int = 10, include_archived: bool = False) -> List[ChatSession]:
        """Get recent sessions for a user."""
        try:
            # Build query filter
            query_filter = {"user_id": user_id}
            if not include_archived:
                query_filter["archived"] = {"$ne": True}
            
            cursor = self.sessions_collection.find(
                query_filter
            ).sort("last_updated", DESCENDING).limit(limit)
            
            sessions = []
            for session_data in cursor:
                sessions.append(ChatSession.from_dict(session_data))
            
            logger.info(f"Retrieved {len(sessions)} sessions for user {user_id} (archived: {include_archived})")
            return sessions
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to retrieve user sessions: {e}")
            return []
    
    async def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to a session."""
        try:
            message = ChatMessage(
                message_id=str(uuid.uuid4()),
                role=role,
                content=content,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata
            )
            
            # Update session with new message and timestamp
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": message.to_dict()},
                    "$set": {"last_updated": datetime.now(timezone.utc).isoformat()}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Added {role} message to session {session_id[:8]}...")
                return True
            else:
                logger.warning(f"Failed to add message to session {session_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"❌ Failed to add message to session: {e}")
            return False
    
    async def get_session_messages(self, session_id: str, limit: int = None) -> List[ChatMessage]:
        """Get messages from a session, optionally limited to recent messages."""
        try:
            session_data = self.sessions_collection.find_one(
                {"session_id": session_id},
                {"messages": 1}
            )
            
            if not session_data:
                return []
            
            messages = [ChatMessage.from_dict(msg) for msg in session_data.get("messages", [])]
            
            if limit:
                messages = messages[-limit:]  # Get the most recent messages
            
            return messages
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to retrieve session messages: {e}")
            return []
    
    async def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Update session metadata."""
        try:
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "metadata": metadata,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to update session metadata: {e}")
            return False
    
    async def archive_session(self, session_id: str, reason: str = "User requested") -> bool:
        """Archive a session instead of deleting it for audit purposes."""
        try:
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "archived": True,
                        "archived_at": datetime.now(timezone.utc).isoformat(),
                        "archive_reason": reason,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Archived session: {session_id} (Reason: {reason})")
                return True
            else:
                logger.warning(f"Session not found for archiving: {session_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"❌ Failed to archive session: {e}")
            return False
    
    async def get_session_count(self, user_id: str = None, include_archived: bool = True) -> int:
        """Get total number of sessions, optionally for a specific user."""
        try:
            filter_query = {}
            if user_id:
                filter_query["user_id"] = user_id
            if not include_archived:
                filter_query["archived"] = {"$ne": True}
                
            count = self.sessions_collection.count_documents(filter_query)
            return count
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to get session count: {e}")
            return 0
    
    async def cleanup_old_sessions(self, days_old: int = 365) -> int:
        """Archive (don't delete) sessions older than specified days for audit compliance."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timezone.timedelta(days=days_old)
            
            result = self.sessions_collection.update_many(
                {
                    "last_updated": {"$lt": cutoff_date.isoformat()},
                    "archived": {"$ne": True}  # Only archive non-archived sessions
                },
                {
                    "$set": {
                        "archived": True,
                        "archived_at": datetime.now(timezone.utc).isoformat(),
                        "archive_reason": f"Auto-archived after {days_old} days of inactivity"
                    }
                }
            )
            
            archived_count = result.modified_count
            logger.info(f"✅ Archived {archived_count} old sessions (older than {days_old} days)")
            return archived_count
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to archive old sessions: {e}")
            return 0
    
    def get_session_stats(self, include_archived: bool = True) -> Dict[str, Any]:
        """Get overall session statistics."""
        try:
            # Base aggregation pipeline
            match_stage = {}
            if not include_archived:
                match_stage = {"$match": {"archived": {"$ne": True}}}
            
            pipeline = []
            if match_stage:
                pipeline.append(match_stage)
            
            pipeline.extend([
                {
                    "$group": {
                        "_id": None,
                        "total_sessions": {"$sum": 1},
                        "total_messages": {"$sum": {"$size": "$messages"}},
                        "unique_users": {"$addToSet": "$user_id"},
                        "archived_sessions": {
                            "$sum": {"$cond": [{"$eq": ["$archived", True]}, 1, 0]}
                        }
                    }
                },
                {
                    "$project": {
                        "total_sessions": 1,
                        "total_messages": 1,
                        "unique_users": {"$size": "$unique_users"},
                        "archived_sessions": 1,
                        "active_sessions": {"$subtract": ["$total_sessions", "$archived_sessions"]}
                    }
                }
            ])
            
            result = list(self.sessions_collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            else:
                return {
                    "total_sessions": 0,
                    "active_sessions": 0,
                    "archived_sessions": 0,
                    "total_messages": 0,
                    "unique_users": 0
                }
                
        except PyMongoError as e:
            logger.error(f"❌ Failed to get session stats: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "archived_sessions": 0,
                "total_messages": 0,
                "unique_users": 0,
                "error": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("✅ Closed MongoDB session manager connection")

# Context manager for automatic cleanup
class SessionManagerContext:
    """Context manager for automatic session manager cleanup."""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.session_manager = None
    
    async def __aenter__(self) -> MongoSessionManager:
        self.session_manager = MongoSessionManager(self.connection_string)
        return self.session_manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session_manager:
            self.session_manager.close()

# Convenience function for creating session manager
def create_session_manager(connection_string: str = None) -> MongoSessionManager:
    """Create a new session manager instance."""
    return MongoSessionManager(connection_string)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_session_manager():
        """Test the session manager functionality."""
        try:
            # Create session manager
            session_manager = MongoSessionManager()
            
            # Create a test session
            session = await session_manager.create_session(
                user_id="test_user",
                metadata={"source": "test", "version": "1.0"}
            )
            print(f"Created session: {session.session_id}")
            
            # Add some messages
            await session_manager.add_message(
                session.session_id,
                "user",
                "Hello, can you help me with parliamentary information?"
            )
            
            await session_manager.add_message(
                session.session_id,
                "assistant",
                "Of course! I'm TrackSabha — here to help with Indian Parliamentary information."
            )
            
            # Retrieve messages
            messages = await session_manager.get_session_messages(session.session_id)
            print(f"Retrieved {len(messages)} messages")
            
            for msg in messages:
                print(f"{msg.role}: {msg.content[:50]}...")
            
            # Test archiving instead of deleting
            await session_manager.archive_session(session.session_id, "Test archiving")
            print(f"Archived session: {session.session_id}")
            
            # Get session stats
            stats = session_manager.get_session_stats()
            print(f"Session stats: {stats}")
            
            # Cleanup
            session_manager.close()
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run the test
    asyncio.run(test_session_manager())
