-- Migration: Add Instance Management Tables
-- Version: 001_add_instance_tables.sql

-- Create tool_instances table
CREATE TABLE IF NOT EXISTS tool_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Relationships
    workflow_id UUID NOT NULL,
    tool_definition_id UUID NOT NULL REFERENCES tool_definitions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    
    -- Configuration
    configuration JSONB DEFAULT '{}',
    settings JSONB DEFAULT '{}',
    
    -- Credential Management
    credential_mappings JSONB DEFAULT '{}',
    required_credentials JSONB DEFAULT '[]',
    
    -- Status and State
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    
    -- Execution History
    last_execution_id UUID,
    execution_count NUMERIC(10, 0) DEFAULT 0,
    total_credits_consumed NUMERIC(10, 4) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for tool_instances
CREATE INDEX idx_tool_instances_workflow_id ON tool_instances(workflow_id);
CREATE INDEX idx_tool_instances_user_id ON tool_instances(user_id);
CREATE INDEX idx_tool_instances_status ON tool_instances(status);
CREATE INDEX idx_tool_instances_tool_definition_id ON tool_instances(tool_definition_id);

-- Create executions table
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Execution identification
    workflow_id UUID NOT NULL,
    instance_id UUID NOT NULL REFERENCES tool_instances(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    
    -- Execution details
    status VARCHAR(50) NOT NULL DEFAULT 'QUEUED',
    input_data JSONB,
    output_data JSONB,
    
    -- Error handling
    error_message TEXT,
    error_type VARCHAR(100),
    retry_count NUMERIC(3, 0) DEFAULT 0,
    max_retries NUMERIC(3, 0) DEFAULT 3,
    
    -- Timing
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Resource usage
    credits_consumed NUMERIC(10, 4) DEFAULT 0,
    processing_time_ms NUMERIC(10, 0),
    
    -- Metadata
    execution_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for executions
CREATE INDEX idx_executions_workflow_id ON executions(workflow_id);
CREATE INDEX idx_executions_instance_id ON executions(instance_id);
CREATE INDEX idx_executions_user_id ON executions(user_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_created_at ON executions(created_at DESC);

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job identification
    job_type VARCHAR(100) NOT NULL,
    execution_id UUID NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    
    -- Job queue information
    queue_name VARCHAR(100) DEFAULT 'default',
    priority NUMERIC(3, 0) DEFAULT 0,
    
    -- Job status
    status VARCHAR(50) NOT NULL DEFAULT 'QUEUED',
    worker_id VARCHAR(255),
    
    -- Job data
    job_data JSONB NOT NULL,
    result_data JSONB,
    
    -- Error handling
    error_details JSONB,
    retry_count NUMERIC(3, 0) DEFAULT 0,
    
    -- Timing
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for jobs
CREATE INDEX idx_jobs_execution_id ON jobs(execution_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_job_type ON jobs(job_type);
CREATE INDEX idx_jobs_queue_name ON jobs(queue_name);
CREATE INDEX idx_jobs_scheduled_at ON jobs(scheduled_at);

-- Add foreign key constraint for last_execution_id in tool_instances
-- (Done separately to avoid circular dependency)
ALTER TABLE tool_instances 
ADD CONSTRAINT fk_tool_instances_last_execution 
FOREIGN KEY (last_execution_id) REFERENCES executions(id) ON DELETE SET NULL;

-- Create updated_at trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_tool_instances_updated_at 
    BEFORE UPDATE ON tool_instances 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_executions_updated_at 
    BEFORE UPDATE ON executions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at 
    BEFORE UPDATE ON jobs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add constraints for status values
ALTER TABLE tool_instances 
ADD CONSTRAINT chk_instance_status 
CHECK (status IN ('PENDING', 'CONFIGURATION_REQUIRED', 'READY', 'RUNNING', 'SUCCESS', 'FAILED', 'PAUSED'));

ALTER TABLE executions 
ADD CONSTRAINT chk_execution_status 
CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', 'RETRYING'));

ALTER TABLE jobs 
ADD CONSTRAINT chk_job_status 
CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', 'RETRYING'));

-- Add check constraints for numeric ranges
ALTER TABLE executions 
ADD CONSTRAINT chk_execution_retry_count 
CHECK (retry_count >= 0 AND retry_count <= max_retries);

ALTER TABLE executions 
ADD CONSTRAINT chk_execution_max_retries 
CHECK (max_retries >= 0 AND max_retries <= 10);

ALTER TABLE jobs 
ADD CONSTRAINT chk_job_retry_count 
CHECK (retry_count >= 0);

ALTER TABLE jobs 
ADD CONSTRAINT chk_job_priority 
CHECK (priority >= 0 AND priority <= 999);

-- Add comments for documentation
COMMENT ON TABLE tool_instances IS 'Tool instances - configured instances of tools for workflow execution';
COMMENT ON TABLE executions IS 'Execution records - tracks individual tool executions';
COMMENT ON TABLE jobs IS 'Job queue records - background jobs for processing executions';

COMMENT ON COLUMN tool_instances.credential_mappings IS 'Maps credential_type to credential_id or provider';
COMMENT ON COLUMN tool_instances.required_credentials IS 'List of required credential types for this tool';
COMMENT ON COLUMN executions.execution_metadata IS 'Additional metadata about the execution';
COMMENT ON COLUMN jobs.job_data IS 'Input data and configuration for job processing';
