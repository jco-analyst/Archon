# Database Migrations for Archon

This directory contains database migration scripts for Archon's data schema updates.

## Token Usage Migration

The OpenAI Free provider requires a token usage tracking table to monitor daily limits and implement automatic fallback functionality.

### Files

- `create_token_usage_table.sql` - Basic SQL schema for development
- `migrate_token_usage_production.py` - Production-ready migration script with safety features

### Running the Migration

#### Prerequisites

1. Install required dependencies:
```bash
pip install asyncpg python-dotenv
```

2. Set environment variables:
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_KEY="your-service-role-key"
```

#### Production Migration

**Step 1: Test with Dry Run**
```bash
python migrate_token_usage_production.py --dry-run
```

**Step 2: Run Migration with Backup**
```bash
python migrate_token_usage_production.py --backup
```

**Step 3: Verify Migration**
```bash
python migrate_token_usage_production.py --sample-data
```

#### Command Line Options

- `--dry-run`: Preview changes without executing
- `--rollback`: Remove the token usage table
- `--sample-data`: Insert test data for verification
- `--backup`: Create backup before migration (default: enabled)

#### Migration Features

✅ **Safety First**
- Automatic backup creation before changes
- Comprehensive error handling and logging
- Rollback functionality for emergency recovery
- Dry-run mode for testing

✅ **Production Ready**
- Connection pooling and timeout handling
- Detailed logging with timestamps
- Verification steps after migration
- Progress tracking and status reports

✅ **Database Optimizations**
- Proper indexes for query performance
- Check constraints for data integrity
- Updated_at triggers for audit trails
- Unique constraints to prevent duplicates

### Table Schema

```sql
CREATE TABLE archon_token_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_name VARCHAR(50) NOT NULL CHECK (provider_name IN ('openai_free')),
    model_name VARCHAR(100) NOT NULL,
    usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
    tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
    token_limit INTEGER NOT NULL CHECK (token_limit > 0),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider_name, model_name, usage_date)
);
```

### Token Limits

The table tracks daily usage for OpenAI Free models:

**Premium Models (250,000 tokens/day):**
- gpt-5, gpt-5-chat-latest, gpt-4.1, gpt-4o, o1, o3

**Mini Models (2,500,000 tokens/day):**
- gpt-5-mini, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini, o1-mini, o3-mini, o4-mini, codex-mini-latest

### Troubleshooting

#### Connection Issues
```bash
# Verify environment variables
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY

# Test connection manually
psql postgresql://postgres:$SUPABASE_SERVICE_KEY@db.your-project.supabase.co:5432/postgres
```

#### Permission Issues
- Ensure the service key has database modification permissions
- Check that the database allows connections from your IP
- Verify the service key is not expired

#### Migration Failures
1. Check the log file created in the current directory
2. Use `--rollback` to clean up partial migrations
3. Run with `--dry-run` to test without changes
4. Contact support with the log file if issues persist

### Maintenance

#### Regular Cleanup (Optional)
Token usage data grows daily. Consider archiving old records:

```sql
-- Archive records older than 90 days
INSERT INTO archon_token_usage_archive 
SELECT * FROM archon_token_usage 
WHERE usage_date < CURRENT_DATE - INTERVAL '90 days';

DELETE FROM archon_token_usage 
WHERE usage_date < CURRENT_DATE - INTERVAL '90 days';
```

#### Monitoring Queries

**Check current usage:**
```sql
SELECT provider_name, model_name, tokens_used, token_limit,
       ROUND((tokens_used::FLOAT / token_limit) * 100, 2) as usage_percent
FROM archon_token_usage 
WHERE usage_date = CURRENT_DATE;
```

**Find models approaching limits:**
```sql
SELECT * FROM archon_token_usage 
WHERE usage_date = CURRENT_DATE 
AND tokens_used > token_limit * 0.8;
```

### Security Considerations

- Store service keys securely (use environment variables or secret managers)
- Regularly rotate database credentials
- Monitor access logs for unusual activity
- Keep migration logs for audit purposes
- Use least-privilege access for migration accounts