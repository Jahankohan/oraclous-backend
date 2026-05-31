"""SQLAlchemy model package.

Importing this package imports every model module, which registers every table
on ``Base.metadata``. The fresh-database bootstrap in
``app.core.database.init_database_schema`` calls ``Base.metadata.create_all`` —
a model module that is not reachable from this package is invisible to that
call, so its table silently goes missing on a fresh install.

Every new model module must be added here.
"""

from app.models import chat, graph, organization, recipe

__all__ = ["chat", "graph", "organization", "recipe"]
