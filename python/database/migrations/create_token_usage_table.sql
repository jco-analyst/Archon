-- Migration: Create archon_token_usage table for tracking daily token consumption
-- This table tracks token usage per provider per model per day for enforcing limits

CREATE TABLE IF NOT EXISTS archon_token_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_name VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    usage_date DATE NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    token_limit INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Ensure one record per provider/model/date combination
    UNIQUE(provider_name, model_name, usage_date)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_token_usage_provider_date 
    ON archon_token_usage(provider_name, usage_date);

CREATE INDEX IF NOT EXISTS idx_token_usage_model_date 
    ON archon_token_usage(provider_name, model_name, usage_date);

-- Create index for cleanup operations
CREATE INDEX IF NOT EXISTS idx_token_usage_date 
    ON archon_token_usage(usage_date);

-- Add comments for documentation
COMMENT ON TABLE archon_token_usage IS 'Tracks daily token usage per provider per model for enforcing usage limits';
COMMENT ON COLUMN archon_token_usage.provider_name IS 'Name of the provider (e.g., openai_free)';
COMMENT ON COLUMN archon_token_usage.model_name IS 'Name of the model (e.g., gpt-4o-mini)';
COMMENT ON COLUMN archon_token_usage.usage_date IS 'Date for which token usage is tracked (resets daily)';
COMMENT ON COLUMN archon_token_usage.tokens_used IS 'Cumulative tokens used for this provider/model/date';
COMMENT ON COLUMN archon_token_usage.token_limit IS 'Daily token limit for this provider/model combination';

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_token_usage_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER trigger_update_token_usage_updated_at
    BEFORE UPDATE ON archon_token_usage
    FOR EACH ROW
    EXECUTE FUNCTION update_token_usage_updated_at();