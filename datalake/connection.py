"""
Database connection management with async support.
Provides session factory and dependency injection for FastAPI.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator

from app.config import get_settings

settings = get_settings()

from sqlalchemy.engine.url import make_url

# Convert postgres:// to postgresql+asyncpg:// if needed
database_url = settings.database_url
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Fix for asyncpg requiring integer for statement_cache_size
# If passed in URL params, it comes as a string which causes TypeError in asyncpg
url_obj = make_url(database_url)
connect_args = {}

# Handle statement_cache_size specifically
if "statement_cache_size" in url_obj.query:
    # We must remove it from the URL string to avoid string-typing conflicts
    # Reconstruct URL without this param
    new_query = {k: v for k, v in url_obj.query.items() if k != "statement_cache_size"}
    url_obj = url_obj._replace(query=new_query)
    # Pass strictly as integer in connect_args
    connect_args["statement_cache_size"] = 0

# Also enforce it for Supabase transaction pooler if not present
if "statement_cache_size" not in connect_args:
    connect_args["statement_cache_size"] = 0

# Create async engine
engine = create_async_engine(
    url_obj,
    echo=settings.debug,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes to get database session.
    Automatically handles session lifecycle with proper error handling.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
        finally:
            await session.close()


async def init_db():
    """
    Initialize database tables.
    Called on application startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    Close database connections.
    Called on application shutdown.
    """
    await engine.dispose()
