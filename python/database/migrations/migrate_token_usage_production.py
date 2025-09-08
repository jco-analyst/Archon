#!/usr/bin/env python3
"""
Production Migration Script for OpenAI Free Token Usage Table
============================================================

This script safely creates the archon_token_usage table in production environments.
It includes proper error handling, rollback procedures, and safety checks.

Usage:
    python migrate_token_usage_production.py [--dry-run] [--rollback]

Requirements:
    - SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables
    - PostgreSQL database with appropriate permissions
"""

import asyncio
import logging
import argparse
import sys
import os
from typing import Optional
from datetime import datetime

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

try:
    import asyncpg
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install: {e}")
    print("Run: pip install asyncpg python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'token_usage_migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class TokenUsageMigration:
    """Handles the migration of the token usage table with proper safety checks."""
    
    def __init__(self):
        self.connection: Optional[asyncpg.Connection] = None
        
    async def connect(self) -> bool:
        """Establish database connection with proper error handling."""
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY")
            return False
            
        # Extract database connection details from Supabase URL
        # Format: https://xyz.supabase.co -> postgresql://postgres:key@db.xyz.supabase.co:5432/postgres
        try:
            project_id = supabase_url.replace('https://', '').replace('.supabase.co', '')
            db_url = f"postgresql://postgres:{supabase_key}@db.{project_id}.supabase.co:5432/postgres"
            
            logger.info("Connecting to database...")
            self.connection = await asyncpg.connect(db_url)
            logger.info("✅ Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection safely."""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            result = await self.connection.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = $1
                )
            """, table_name)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    async def backup_existing_data(self) -> bool:
        """Create a backup of existing token usage data if table exists."""
        try:
            if not await self.table_exists('archon_token_usage'):
                logger.info("No existing token usage table found - no backup needed")
                return True
                
            # Create backup table with timestamp
            backup_table = f"archon_token_usage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            await self.connection.execute(f"""
                CREATE TABLE {backup_table} AS 
                SELECT * FROM archon_token_usage
            """)
            
            count = await self.connection.fetchval(f"SELECT COUNT(*) FROM {backup_table}")
            logger.info(f"✅ Backed up {count} records to {backup_table}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    async def create_table(self) -> bool:
        """Create the token usage table with proper constraints and indexes."""
        try:
            # Check if table already exists
            if await self.table_exists('archon_token_usage'):
                logger.info("Table 'archon_token_usage' already exists")
                return True
            
            logger.info("Creating archon_token_usage table...")
            
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS archon_token_usage (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    provider_name VARCHAR(50) NOT NULL CHECK (provider_name IN ('openai_free')),
                    model_name VARCHAR(100) NOT NULL,
                    usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
                    token_limit INTEGER NOT NULL CHECK (token_limit > 0),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(provider_name, model_name, usage_date)
                )
            """)
            
            # Create indexes for better performance
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_usage_provider_model_date 
                ON archon_token_usage(provider_name, model_name, usage_date)
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_usage_date 
                ON archon_token_usage(usage_date)
            """)
            
            # Create updated_at trigger
            await self.connection.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """)
            
            await self.connection.execute("""
                CREATE TRIGGER update_archon_token_usage_updated_at 
                BEFORE UPDATE ON archon_token_usage 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
            """)
            
            logger.info("✅ Table 'archon_token_usage' created successfully with indexes and triggers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create table: {e}")
            return False
    
    async def insert_sample_data(self, dry_run: bool = False) -> bool:
        """Insert sample data for testing purposes (optional)."""
        if dry_run:
            logger.info("[DRY RUN] Would insert sample token usage data")
            return True
            
        try:
            # Insert sample data for common models
            sample_data = [
                ('openai_free', 'gpt-4o-mini', '2025-09-08', 1000, 2500000),
                ('openai_free', 'gpt-4o', '2025-09-08', 5000, 250000),
            ]
            
            for provider, model, date, used, limit in sample_data:
                await self.connection.execute("""
                    INSERT INTO archon_token_usage (provider_name, model_name, usage_date, tokens_used, token_limit)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (provider_name, model_name, usage_date) DO NOTHING
                """, provider, model, date, used, limit)
            
            logger.info("✅ Sample data inserted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            return False
    
    async def verify_migration(self) -> bool:
        """Verify the migration was successful."""
        try:
            # Check table exists
            if not await self.table_exists('archon_token_usage'):
                logger.error("❌ Table was not created")
                return False
            
            # Check table structure
            columns = await self.connection.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'archon_token_usage'
                ORDER BY ordinal_position
            """)
            
            expected_columns = {
                'id', 'provider_name', 'model_name', 'usage_date', 
                'tokens_used', 'token_limit', 'created_at', 'updated_at'
            }
            actual_columns = {col['column_name'] for col in columns}
            
            if not expected_columns.issubset(actual_columns):
                missing = expected_columns - actual_columns
                logger.error(f"❌ Missing columns: {missing}")
                return False
            
            # Check constraints
            constraints = await self.connection.fetch("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_schema = 'public' AND table_name = 'archon_token_usage'
            """)
            
            constraint_types = {c['constraint_type'] for c in constraints}
            if 'PRIMARY KEY' not in constraint_types or 'UNIQUE' not in constraint_types:
                logger.error("❌ Missing required constraints")
                return False
            
            # Check indexes
            indexes = await self.connection.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'archon_token_usage' AND schemaname = 'public'
            """)
            
            index_names = {idx['indexname'] for idx in indexes}
            expected_indexes = {'idx_token_usage_provider_model_date', 'idx_token_usage_date'}
            
            if not expected_indexes.issubset(index_names):
                missing_indexes = expected_indexes - index_names
                logger.warning(f"⚠️  Missing indexes (non-critical): {missing_indexes}")
            
            logger.info("✅ Migration verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False
    
    async def rollback(self) -> bool:
        """Rollback the migration by dropping the table."""
        try:
            logger.info("Rolling back migration...")
            
            # Drop triggers first
            await self.connection.execute("""
                DROP TRIGGER IF EXISTS update_archon_token_usage_updated_at ON archon_token_usage
            """)
            
            # Drop the table
            await self.connection.execute("DROP TABLE IF EXISTS archon_token_usage CASCADE")
            
            logger.info("✅ Migration rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

async def main():
    """Main migration function with command line argument handling."""
    parser = argparse.ArgumentParser(description='Migrate token usage table for Archon')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    parser.add_argument('--sample-data', action='store_true', help='Insert sample data for testing')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup before migration (default: True)')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
    
    migration = TokenUsageMigration()
    
    try:
        # Connect to database
        if not await migration.connect():
            logger.error("Failed to connect to database")
            return 1
        
        if args.rollback:
            logger.info("🔄 Rolling back migration...")
            success = await migration.rollback()
            return 0 if success else 1
        
        # Run migration
        logger.info("🚀 Starting token usage table migration...")
        
        # Create backup if requested and not dry run
        if args.backup and not args.dry_run:
            if not await migration.backup_existing_data():
                logger.error("Backup failed - aborting migration")
                return 1
        
        # Create table
        if args.dry_run:
            logger.info("[DRY RUN] Would create archon_token_usage table with:")
            logger.info("  - id (UUID PRIMARY KEY)")
            logger.info("  - provider_name (VARCHAR(50) NOT NULL)")
            logger.info("  - model_name (VARCHAR(100) NOT NULL)")
            logger.info("  - usage_date (DATE NOT NULL)")
            logger.info("  - tokens_used (INTEGER NOT NULL)")
            logger.info("  - token_limit (INTEGER NOT NULL)")
            logger.info("  - created_at, updated_at (TIMESTAMPTZ)")
            logger.info("  - Unique constraint on (provider_name, model_name, usage_date)")
            logger.info("  - Indexes on provider/model/date and date")
            logger.info("  - Updated_at trigger")
        else:
            if not await migration.create_table():
                logger.error("Table creation failed")
                return 1
        
        # Insert sample data if requested
        if args.sample_data:
            if not await migration.insert_sample_data(args.dry_run):
                logger.warning("Sample data insertion failed (non-critical)")
        
        # Verify migration (skip for dry run)
        if not args.dry_run:
            if not await migration.verify_migration():
                logger.error("Migration verification failed")
                return 1
        
        if args.dry_run:
            logger.info("✅ DRY RUN completed - no changes made")
        else:
            logger.info("✅ Migration completed successfully!")
            logger.info("Token usage tracking is now enabled for OpenAI Free provider")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        return 1
    finally:
        await migration.disconnect()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Failed to run migration: {e}")
        sys.exit(1)