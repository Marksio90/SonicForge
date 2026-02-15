"""Connection Pool Manager - Ready to use!

Usage:
    from app.core.connection_pool import pool_manager
    
    # Initialize in main.py lifespan
    await pool_manager.initialize()
    
    # Use in code
    async with pool_manager.get_db_connection() as conn:
        result = await conn.fetch("SELECT * FROM tracks")
    
    # HTTP client
    client = pool_manager.http_client
    response = await client.get(url)
"""

from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as aioredis
import httpx
import structlog

logger = structlog.get_logger(__name__)


class ConnectionPoolManager:
    """Centralized connection pool management with health checks."""
    
    def __init__(self):
        self.db_pool = None
        self.redis_pool = None
        self.http_client = None
        self._initialized = False
    
    async def initialize(self, database_url: str = None, redis_url: str = None):
        """Initialize all connection pools.
        
        Args:
            database_url: PostgreSQL connection URL
            redis_url: Redis connection URL
        """
        if self._initialized:
            logger.warning("connection_pool_already_initialized")
            return
        
        # PostgreSQL connection pool (asyncpg)
        if database_url:
            try:
                self.db_pool = await asyncpg.create_pool(
                    database_url,
                    min_size=10,
                    max_size=100,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60,
                )
                logger.info("postgres_pool_initialized", min_size=10, max_size=100)
            except Exception as e:
                logger.error("postgres_pool_init_failed", error=str(e))
        
        # Redis connection pool
        if redis_url:
            try:
                self.redis_pool = aioredis.ConnectionPool.from_url(
                    redis_url,
                    max_connections=200,
                    decode_responses=False,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                )
                logger.info("redis_pool_initialized", max_connections=200)
            except Exception as e:
                logger.error("redis_pool_init_failed", error=str(e))
        
        # HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=30.0,
            ),
            http2=True,  # Enable HTTP/2 for better performance
        )
        logger.info("http_client_initialized", max_connections=200, http2=True)
        
        self._initialized = True
        logger.info("connection_pool_manager_ready")
    
    async def close(self):
        """Gracefully close all pools."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("postgres_pool_closed")
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
            logger.info("redis_pool_closed")
        
        if self.http_client:
            await self.http_client.aclose()
            logger.info("http_client_closed")
        
        self._initialized = False
        logger.info("connection_pool_manager_closed")
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool.
        
        Usage:
            async with pool_manager.get_db_connection() as conn:
                result = await conn.fetch("SELECT * FROM tracks")
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.db_pool.acquire() as conn:
            yield conn
    
    async def get_redis_client(self):
        """Get Redis client from pool.
        
        Usage:
            redis = await pool_manager.get_redis_client()
            await redis.set("key", "value")
        """
        if not self.redis_pool:
            raise RuntimeError("Redis pool not initialized")
        
        return aioredis.Redis(connection_pool=self.redis_pool)
    
    async def health_check(self) -> dict:
        """Check health of all connections."""
        health = {
            "postgres": "unknown",
            "redis": "unknown",
            "http": "healthy",
        }
        
        # Check PostgreSQL
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health["postgres"] = "healthy"
            except Exception as e:
                health["postgres"] = f"unhealthy: {str(e)}"
        
        # Check Redis
        if self.redis_pool:
            try:
                redis = await self.get_redis_client()
                await redis.ping()
                health["redis"] = "healthy"
            except Exception as e:
                health["redis"] = f"unhealthy: {str(e)}"
        
        return health
    
    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        stats = {}
        
        if self.db_pool:
            stats["postgres"] = {
                "size": self.db_pool.get_size(),
                "free": self.db_pool.get_idle_size(),
                "min_size": self.db_pool.get_min_size(),
                "max_size": self.db_pool.get_max_size(),
            }
        
        if self.http_client:
            stats["http"] = {
                "is_closed": self.http_client.is_closed,
            }
        
        return stats


# Global singleton
pool_manager = ConnectionPoolManager()
