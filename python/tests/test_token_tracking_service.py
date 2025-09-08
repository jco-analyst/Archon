"""
Tests for TokenTrackingService - manages daily token usage limits.
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.server.services.token_tracking_service import TokenTrackingService


class TestTokenTrackingService:
    """Test suite for TokenTrackingService"""

    @pytest.fixture
    def token_service(self):
        """Create TokenTrackingService instance for testing"""
        return TokenTrackingService()

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client with table operations"""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        
        # Setup method chaining for table operations
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.lt.return_value = mock_table
        
        return mock_client, mock_table

    @pytest.mark.asyncio
    async def test_track_token_usage_new_record(self, token_service, mock_supabase_client):
        """Test tracking token usage when no existing record exists"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock no existing record
        mock_table.execute.return_value = MagicMock(data=[])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.track_token_usage("openai_free", "gpt-4o-mini", 1000)
        
        assert result["success"] is True
        assert result["tokens_used"] == 1000
        assert result["token_limit"] == 2500000  # gpt-4o-mini limit
        assert result["remaining_tokens"] == 2499000
        assert result["limit_exceeded"] is False
        
        # Verify upsert was called
        mock_table.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_token_usage_existing_record(self, token_service, mock_supabase_client):
        """Test tracking token usage when record already exists"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock existing record with 5000 tokens already used
        existing_record = {
            "tokens_used": 5000,
            "token_limit": 2500000,
            "usage_date": date.today().isoformat()
        }
        mock_table.execute.return_value = MagicMock(data=[existing_record])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.track_token_usage("openai_free", "gpt-4o-mini", 2000)
        
        assert result["success"] is True
        assert result["tokens_used"] == 7000  # 5000 + 2000
        assert result["remaining_tokens"] == 2493000
        assert result["limit_exceeded"] is False

    @pytest.mark.asyncio
    async def test_track_token_usage_exceeds_limit(self, token_service, mock_supabase_client):
        """Test tracking token usage that exceeds daily limit"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock existing record near the limit
        existing_record = {
            "tokens_used": 249500,
            "token_limit": 250000,
            "usage_date": date.today().isoformat()
        }
        mock_table.execute.return_value = MagicMock(data=[existing_record])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.track_token_usage("openai_free", "gpt-4.1", 1000)
        
        assert result["success"] is True
        assert result["tokens_used"] == 250500  # Exceeds 250k limit
        assert result["remaining_tokens"] == 0
        assert result["limit_exceeded"] is True

    @pytest.mark.asyncio
    async def test_get_daily_usage_existing_record(self, token_service, mock_supabase_client):
        """Test getting daily usage when record exists"""
        mock_client, mock_table = mock_supabase_client
        
        usage_record = {
            "tokens_used": 15000,
            "token_limit": 250000,
            "usage_date": date.today().isoformat()
        }
        mock_table.execute.return_value = MagicMock(data=[usage_record])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.get_daily_usage("openai_free", "gpt-4.1")
        
        assert result["tokens_used"] == 15000
        assert result["token_limit"] == 250000
        assert result["remaining_tokens"] == 235000
        assert result["limit_exceeded"] is False

    @pytest.mark.asyncio
    async def test_get_daily_usage_no_record(self, token_service, mock_supabase_client):
        """Test getting daily usage when no record exists"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock no existing record
        mock_table.execute.return_value = MagicMock(data=[])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.get_daily_usage("openai_free", "gpt-4o-mini")
        
        assert result["tokens_used"] == 0
        assert result["token_limit"] == 2500000  # gpt-4o-mini limit
        assert result["remaining_tokens"] == 2500000
        assert result["limit_exceeded"] is False

    @pytest.mark.asyncio
    async def test_check_token_limit_allowed(self, token_service, mock_supabase_client):
        """Test token limit check when request is within limits"""
        mock_client, mock_table = mock_supabase_client
        
        usage_record = {
            "tokens_used": 10000,
            "token_limit": 250000,
            "usage_date": date.today().isoformat()
        }
        mock_table.execute.return_value = MagicMock(data=[usage_record])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.check_token_limit("openai_free", "gpt-4.1", 5000)
        
        assert result["allowed"] is True
        assert result["current_usage"] == 10000
        assert result["tokens_needed"] == 5000
        assert result["would_exceed_limit"] is False
        assert result["reason"] == "within_limit"

    @pytest.mark.asyncio
    async def test_check_token_limit_denied(self, token_service, mock_supabase_client):
        """Test token limit check when request would exceed limits"""
        mock_client, mock_table = mock_supabase_client
        
        usage_record = {
            "tokens_used": 240000,
            "token_limit": 250000,
            "usage_date": date.today().isoformat()
        }
        mock_table.execute.return_value = MagicMock(data=[usage_record])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.check_token_limit("openai_free", "gpt-4.1", 20000)
        
        assert result["allowed"] is False
        assert result["current_usage"] == 240000
        assert result["tokens_needed"] == 20000
        assert result["would_exceed_limit"] is True
        assert result["reason"] == "limit_exceeded"

    @pytest.mark.asyncio
    async def test_get_provider_usage_summary(self, token_service, mock_supabase_client):
        """Test getting usage summary for all models of a provider"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock usage records for multiple models
        usage_records = [
            {
                "model_name": "gpt-4o-mini",
                "tokens_used": 50000,
                "token_limit": 2500000
            },
            {
                "model_name": "gpt-4.1",
                "tokens_used": 25000,
                "token_limit": 250000
            }
        ]
        mock_table.execute.return_value = MagicMock(data=usage_records)
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.get_provider_usage_summary("openai_free")
        
        assert result["provider_name"] == "openai_free"
        assert result["total_tokens_used"] == 75000
        assert "gpt-4o-mini" in result["models"]
        assert "gpt-4.1" in result["models"]
        assert result["models"]["gpt-4o-mini"]["tokens_used"] == 50000
        assert result["models"]["gpt-4.1"]["tokens_used"] == 25000

    @pytest.mark.asyncio
    async def test_cleanup_old_usage_records(self, token_service, mock_supabase_client):
        """Test cleanup of old token usage records"""
        mock_client, mock_table = mock_supabase_client
        
        # Mock successful deletion
        mock_table.execute.return_value = MagicMock(data=["deleted1", "deleted2", "deleted3"])
        
        with patch.object(token_service, '_get_supabase_client', return_value=mock_client):
            result = await token_service.cleanup_old_usage_records(30)
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        
        # Verify delete was called with date filter
        mock_table.delete.assert_called_once()
        mock_table.lt.assert_called_once()

    def test_get_token_limit_openai_free_models(self, token_service):
        """Test getting token limits for OpenAI Free models"""
        # Test premium models
        assert token_service._get_token_limit("openai_free", "gpt-4.1") == 250000
        assert token_service._get_token_limit("openai_free", "gpt-4o") == 250000
        assert token_service._get_token_limit("openai_free", "o1") == 250000
        
        # Test mini models
        assert token_service._get_token_limit("openai_free", "gpt-4o-mini") == 2500000
        assert token_service._get_token_limit("openai_free", "gpt-4.1-mini") == 2500000
        assert token_service._get_token_limit("openai_free", "o1-mini") == 2500000
        
        # Test unknown model
        assert token_service._get_token_limit("openai_free", "unknown-model") == 0
        
        # Test unknown provider
        assert token_service._get_token_limit("unknown_provider", "gpt-4o") == 0

    @pytest.mark.asyncio
    async def test_error_handling_track_usage(self, token_service):
        """Test error handling in track_token_usage"""
        with patch.object(token_service, '_get_supabase_client', side_effect=Exception("DB Error")):
            result = await token_service.track_token_usage("openai_free", "gpt-4o", 1000)
        
        assert result["success"] is False
        assert "DB Error" in result["error"]
        assert result["tokens_used"] == 0
        assert result["limit_exceeded"] is False

    @pytest.mark.asyncio
    async def test_error_handling_check_limit(self, token_service):
        """Test error handling in check_token_limit - should allow on error"""
        with patch.object(token_service, 'get_daily_usage', side_effect=Exception("DB Error")):
            result = await token_service.check_token_limit("openai_free", "gpt-4o", 1000)
        
        # Should allow request when error occurs to avoid breaking functionality
        assert result["allowed"] is True
        assert result["reason"] == "limit_check_failed"
        assert "DB Error" in result["error"]