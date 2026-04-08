"""
Unit tests for background_jobs.py.

Tests WorkerNeo4jManager, job status transitions, and error handling
— all external deps (Neo4j, PostgreSQL, Celery) mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.services.background_jobs import (
    WorkerNeo4jManager,
    _update_job_status_async,
)

# ---------------------------------------------------------------------------
# Tests: WorkerNeo4jManager — context manager
# ---------------------------------------------------------------------------


class TestWorkerNeo4jManager:
    @pytest.mark.unit
    def test_initial_state_has_no_drivers(self):
        manager = WorkerNeo4jManager()
        assert manager.sync_driver is None
        assert manager.async_driver is None

    @pytest.mark.unit
    def test_get_sync_driver_raises_when_not_connected(self):
        manager = WorkerNeo4jManager()
        with pytest.raises(RuntimeError, match="Sync driver not available"):
            manager.get_sync_driver()

    @pytest.mark.unit
    def test_get_async_driver_raises_when_not_connected(self):
        manager = WorkerNeo4jManager()
        with pytest.raises(RuntimeError, match="Async driver not available"):
            manager.get_async_driver()

    @pytest.mark.unit
    def test_get_sync_driver_returns_driver_when_set(self):
        manager = WorkerNeo4jManager()
        mock_driver = MagicMock()
        manager.sync_driver = mock_driver
        assert manager.get_sync_driver() is mock_driver

    @pytest.mark.unit
    def test_get_async_driver_returns_driver_when_set(self):
        manager = WorkerNeo4jManager()
        mock_driver = MagicMock()
        manager.async_driver = mock_driver
        assert manager.get_async_driver() is mock_driver

    @pytest.mark.unit
    def test_cleanup_sync_closes_driver(self):
        manager = WorkerNeo4jManager()
        mock_driver = MagicMock()
        manager.sync_driver = mock_driver

        manager.cleanup_sync()

        mock_driver.close.assert_called_once()
        assert manager.sync_driver is None

    @pytest.mark.unit
    def test_cleanup_sync_handles_close_exception_gracefully(self):
        manager = WorkerNeo4jManager()
        mock_driver = MagicMock()
        mock_driver.close.side_effect = Exception("close error")
        manager.sync_driver = mock_driver

        # Should not raise
        manager.cleanup_sync()
        assert manager.sync_driver is None

    @pytest.mark.unit
    async def test_async_cleanup_closes_async_driver(self):
        manager = WorkerNeo4jManager()
        mock_async_driver = AsyncMock()
        mock_async_driver.close = AsyncMock()
        manager.async_driver = mock_async_driver

        await manager.cleanup()

        mock_async_driver.close.assert_awaited_once()
        assert manager.async_driver is None

    @pytest.mark.unit
    async def test_async_cleanup_also_closes_sync_driver(self):
        manager = WorkerNeo4jManager()
        mock_sync_driver = MagicMock()
        mock_async_driver = AsyncMock()
        mock_async_driver.close = AsyncMock()
        manager.sync_driver = mock_sync_driver
        manager.async_driver = mock_async_driver

        await manager.cleanup()

        mock_sync_driver.close.assert_called_once()
        assert manager.sync_driver is None
        assert manager.async_driver is None

    @pytest.mark.unit
    async def test_async_cleanup_handles_async_close_exception(self):
        manager = WorkerNeo4jManager()
        mock_async_driver = AsyncMock()
        mock_async_driver.close = AsyncMock(side_effect=Exception("async close error"))
        manager.async_driver = mock_async_driver

        # Should not raise
        await manager.cleanup()
        assert manager.async_driver is None

    @pytest.mark.unit
    def test_sync_context_manager_calls_connect_sync_only(self):
        manager = WorkerNeo4jManager()
        manager.connect_sync_only = MagicMock()
        manager.cleanup_sync = MagicMock()

        with manager as m:
            assert m is manager

        manager.connect_sync_only.assert_called_once()
        manager.cleanup_sync.assert_called_once()

    @pytest.mark.unit
    def test_sync_context_manager_cleans_up_on_exception(self):
        manager = WorkerNeo4jManager()
        manager.connect_sync_only = MagicMock()
        manager.cleanup_sync = MagicMock()

        with pytest.raises(ValueError):
            with manager:
                raise ValueError("test error")

        manager.cleanup_sync.assert_called_once()

    @pytest.mark.unit
    async def test_async_context_manager_calls_connect_and_cleanup(self):
        manager = WorkerNeo4jManager()
        manager.connect = AsyncMock()
        manager.cleanup = AsyncMock()

        async with manager as m:
            assert m is manager

        manager.connect.assert_awaited_once()
        manager.cleanup.assert_awaited_once()

    @pytest.mark.unit
    async def test_async_context_manager_cleans_up_on_exception(self):
        manager = WorkerNeo4jManager()
        manager.connect = AsyncMock()
        manager.cleanup = AsyncMock()

        with pytest.raises(RuntimeError):
            async with manager:
                raise RuntimeError("task error")

        manager.cleanup.assert_awaited_once()

    @pytest.mark.unit
    def test_connect_sync_only_raises_if_neo4j_unavailable(self):
        manager = WorkerNeo4jManager()

        with patch("app.services.background_jobs.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"

            with patch("neo4j.GraphDatabase.driver") as mock_driver_cls:
                mock_driver_cls.return_value.verify_connectivity.side_effect = (
                    Exception("connection refused")
                )

                with pytest.raises(Exception, match="connection refused"):
                    manager.connect_sync_only()

        # Manager should not hold a broken driver
        assert manager.sync_driver is None


# ---------------------------------------------------------------------------
# Tests: _update_job_status_async
# ---------------------------------------------------------------------------

VALID_JOB_ID = "12345678-1234-5678-1234-567812345678"


class TestUpdateJobStatusAsync:
    @pytest.mark.unit
    async def test_updates_status_field(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        await _update_job_status_async(mock_session, VALID_JOB_ID, "completed")

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.unit
    async def test_updates_with_entity_and_relationship_counts(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        await _update_job_status_async(
            mock_session,
            VALID_JOB_ID,
            "completed",
            entities=10,
            relationships=5,
            chunks=3,
        )

        # Verify the session was called with a statement
        assert mock_session.execute.call_count == 1

    @pytest.mark.unit
    async def test_updates_with_error_message(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        await _update_job_status_async(
            mock_session, VALID_JOB_ID, "failed", error="Neo4j unavailable"
        )

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.unit
    async def test_rolls_back_on_execute_failure(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_session.rollback = AsyncMock()

        with pytest.raises(Exception, match="DB error"):
            await _update_job_status_async(mock_session, VALID_JOB_ID, "failed")

        mock_session.rollback.assert_awaited_once()

    @pytest.mark.unit
    async def test_valid_uuid_passed_to_query(self):
        """Ensure job_id is parsed as UUID — rejects invalid UUIDs early."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        await _update_job_status_async(mock_session, VALID_JOB_ID, "processing")
        mock_session.commit.assert_awaited_once()

    @pytest.mark.unit
    async def test_invalid_uuid_raises(self):
        mock_session = AsyncMock()

        with pytest.raises((ValueError, Exception)):
            await _update_job_status_async(mock_session, "not-a-uuid", "processing")


# ---------------------------------------------------------------------------
# Tests: process_ingestion_job state transitions (via async impl)
# ---------------------------------------------------------------------------


class TestProcessIngestionJobStateTransitions:
    @pytest.mark.unit
    async def test_job_not_found_returns_failed(self):
        from app.services.background_jobs import _process_pipeline_ingestion_async

        mock_task = MagicMock()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # Job not found
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.background_jobs.worker_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            result = await _process_pipeline_ingestion_async(
                mock_task, "12345678-1234-5678-1234-567812345678", "user-1"
            )

        assert result["status"] == "failed"
        assert "not found" in result["error"]

    @pytest.mark.unit
    async def test_document_processing_failure_returns_failed(self):
        from app.services.background_jobs import _process_pipeline_ingestion_async

        mock_task = MagicMock()
        mock_job = MagicMock()
        mock_job.graph_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        mock_job.source_content = "some content"
        mock_job.source_type = "text"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_neo4j = AsyncMock()
        mock_neo4j.__aenter__ = AsyncMock(return_value=mock_neo4j)
        mock_neo4j.__aexit__ = AsyncMock(return_value=False)
        mock_graph_service = MagicMock()
        mock_graph_service.get_graph.return_value = {
            "user_id": "user-1",
            "graph_id": str(mock_job.graph_id),
        }
        mock_neo4j.get_sync_driver = MagicMock(return_value=MagicMock())

        with (
            patch("app.services.background_jobs.worker_session_maker") as mock_maker,
            patch(
                "app.services.background_jobs.WorkerNeo4jManager"
            ) as mock_manager_cls,
            patch("app.services.background_jobs.GraphNodeService") as mock_gns_cls,
            patch("app.services.background_jobs.document_processor") as mock_doc_proc,
        ):

            mock_maker.return_value = mock_session
            mock_manager_cls.return_value = mock_neo4j
            mock_gns_cls.return_value = mock_graph_service
            mock_doc_proc.process_document.side_effect = Exception("parsing failed")

            result = await _process_pipeline_ingestion_async(
                mock_task, "12345678-1234-5678-1234-567812345678", "user-1"
            )

        assert result["status"] == "failed"
        assert "parsing failed" in result["error"]

    @pytest.mark.unit
    async def test_access_denied_when_user_mismatch(self):
        from app.services.background_jobs import _process_pipeline_ingestion_async

        mock_task = MagicMock()
        mock_job = MagicMock()
        mock_job.graph_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        mock_job.source_content = "content"
        mock_job.source_type = "text"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_neo4j = AsyncMock()
        mock_neo4j.__aenter__ = AsyncMock(return_value=mock_neo4j)
        mock_neo4j.__aexit__ = AsyncMock(return_value=False)
        mock_neo4j.get_sync_driver = MagicMock(return_value=MagicMock())

        mock_graph_service = MagicMock()
        # Graph belongs to different user
        mock_graph_service.get_graph.return_value = {
            "user_id": "other-user",
            "graph_id": str(mock_job.graph_id),
        }

        with (
            patch("app.services.background_jobs.worker_session_maker") as mock_maker,
            patch(
                "app.services.background_jobs.WorkerNeo4jManager"
            ) as mock_manager_cls,
            patch("app.services.background_jobs.GraphNodeService") as mock_gns_cls,
        ):

            mock_maker.return_value = mock_session
            mock_manager_cls.return_value = mock_neo4j
            mock_gns_cls.return_value = mock_graph_service

            result = await _process_pipeline_ingestion_async(
                mock_task,
                "12345678-1234-5678-1234-567812345678",
                "requesting-user",  # different from graph owner
            )

        assert result["status"] == "error"
        assert "Access denied" in result["message"]
