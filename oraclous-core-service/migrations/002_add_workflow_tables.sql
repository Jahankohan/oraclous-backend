-- Migration: Add Workflow Management Tables
-- Version: 002_add_workflow_tables.sql

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL,
    
    -- Workflow structure
    nodes JSONB NOT NULL DEFAULT '[]',
    edges JSONB NOT NULL DEFAULT '[]',
    
    -- LangGraph integration
    chat_history JSONB DEFAULT '[]',
    generation_prompt TEXT,
    generation_metadata JSONB DEFAULT '{}',
    
    -- Configuration
    settings JSONB DEFAULT '{}',
    variables JSONB DEFAULT '{}',
    
    -- State and status
    status VARCHAR(50) NOT NULL DEFAULT 'DRAFT',
    version VARCHAR(50) DEFAULT '1.0.0',
    is_template BOOLEAN DEFAULT FALSE,
    
    -- Execution tracking
    last_execution_id UUID,
    total_executions NUMERIC(10, 0) DEFAULT 0,
    successful_executions NUMERIC(10, 0) DEFAULT 0,
    
    -- Resource estimates
    estimated_credits_per_run NUMERIC(10, 4) DEFAULT 0,
    total_credits_consumed NUMERIC(12, 4) DEFAULT 0,
    
    -- Metadata and organization
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    category VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflows
CREATE INDEX idx_workflows_owner_id ON workflows(owner_id);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_category ON workflows(category);
CREATE INDEX idx_workflows_is_template ON workflows(is_template);
CREATE INDEX idx_workflows_is_public ON workflows(is_public);
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);
CREATE INDEX idx_workflows_tags ON workflows USING gin(tags);

-- Create workflow_executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Execution identification
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    
    -- Execution details
    status VARCHAR(50) NOT NULL DEFAULT 'QUEUED',
    trigger_type VARCHAR(50) DEFAULT 'MANUAL',
    
    -- Input/Output
    input_parameters JSONB DEFAULT '{}',
    output_data JSONB,
    
    -- Progress tracking
    total_steps NUMERIC(5, 0) DEFAULT 0,
    completed_steps NUMERIC(5, 0) DEFAULT 0,
    failed_steps NUMERIC(5, 0) DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    error_type VARCHAR(100),
    failed_node_id VARCHAR(255),
    
    -- Timing
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Resource usage
    credits_consumed NUMERIC(10, 4) DEFAULT 0,
    processing_time_ms NUMERIC(12, 0),
    
    -- Control features
    can_pause BOOLEAN DEFAULT TRUE,
    can_resume BOOLEAN DEFAULT TRUE,
    paused_at TIMESTAMP WITH TIME ZONE,
    resumed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    execution_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflow_executions
CREATE INDEX idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_user_id ON workflow_executions(user_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_trigger_type ON workflow_executions(trigger_type);
CREATE INDEX idx_workflow_executions_created_at ON workflow_executions(created_at DESC);

-- Create workflow_templates table
CREATE TABLE IF NOT EXISTS workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Template information
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    
    -- Template structure
    template_nodes JSONB NOT NULL,
    template_edges JSONB NOT NULL,
    
    -- Configuration
    parameters JSONB DEFAULT '{}',
    required_tools TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Metadata
    author_id UUID,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    difficulty_level VARCHAR(20) DEFAULT 'BEGINNER',
    estimated_time_minutes NUMERIC(5, 0),
    estimated_credits NUMERIC(10, 4),
    
    -- Usage tracking
    usage_count NUMERIC(10, 0) DEFAULT 0,
    average_rating NUMERIC(3, 2) DEFAULT 0,
    
    -- Publication
    is_published BOOLEAN DEFAULT FALSE,
    is_featured BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflow_templates
CREATE INDEX idx_workflow_templates_category ON workflow_templates(category);
CREATE INDEX idx_workflow_templates_author_id ON workflow_templates(author_id);
CREATE INDEX idx_workflow_templates_difficulty_level ON workflow_templates(difficulty_level);
CREATE INDEX idx_workflow_templates_is_published ON workflow_templates(is_published);
CREATE INDEX idx_workflow_templates_is_featured ON workflow_templates(is_featured);
CREATE INDEX idx_workflow_templates_tags ON workflow_templates USING gin(tags);
CREATE INDEX idx_workflow_templates_usage_count ON workflow_templates(usage_count DESC);
CREATE INDEX idx_workflow_templates_average_rating ON workflow_templates(average_rating DESC);

-- Create workflow_shares table
CREATE TABLE IF NOT EXISTS workflow_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Sharing details
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    owner_id UUID NOT NULL,
    shared_with_id UUID, -- NULL for public shares
    
    -- Permissions
    permission_type VARCHAR(20) NOT NULL DEFAULT 'VIEW',
    can_reshare BOOLEAN DEFAULT FALSE,
    
    -- Sharing metadata
    share_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Usage tracking
    access_count NUMERIC(10, 0) DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflow_shares
CREATE INDEX idx_workflow_shares_workflow_id ON workflow_shares(workflow_id);
CREATE INDEX idx_workflow_shares_owner_id ON workflow_shares(owner_id);
CREATE INDEX idx_workflow_shares_shared_with_id ON workflow_shares(shared_with_id);
CREATE INDEX idx_workflow_shares_share_token ON workflow_shares(share_token);
CREATE INDEX idx_workflow_shares_permission_type ON workflow_shares(permission_type);

-- Add foreign key constraint for last_execution_id in workflows
ALTER TABLE workflows 
ADD CONSTRAINT fk_workflows_last_execution 
FOREIGN KEY (last_execution_id) REFERENCES workflow_executions(id) ON DELETE SET NULL;

-- Update tool_instances to add foreign key to workflows
ALTER TABLE tool_instances 
ADD CONSTRAINT fk_tool_instances_workflow 
FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE;

-- Add updated_at triggers for workflow tables
CREATE TRIGGER update_workflows_updated_at 
    BEFORE UPDATE ON workflows 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_executions_updated_at 
    BEFORE UPDATE ON workflow_executions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_templates_updated_at 
    BEFORE UPDATE ON workflow_templates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_shares_updated_at 
    BEFORE UPDATE ON workflow_shares 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add constraints for status values
ALTER TABLE workflows 
ADD CONSTRAINT chk_workflow_status 
CHECK (status IN ('DRAFT', 'ACTIVE', 'PAUSED', 'COMPLETED', 'FAILED', 'ARCHIVED'));

ALTER TABLE workflow_executions 
ADD CONSTRAINT chk_workflow_execution_status 
CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', 'PAUSED'));

ALTER TABLE workflow_executions 
ADD CONSTRAINT chk_workflow_trigger_type 
CHECK (trigger_type IN ('MANUAL', 'SCHEDULED', 'API', 'WEBHOOK'));

ALTER TABLE workflow_templates 
ADD CONSTRAINT chk_template_difficulty_level 
CHECK (difficulty_level IN ('BEGINNER', 'INTERMEDIATE', 'ADVANCED'));

ALTER TABLE workflow_shares 
ADD CONSTRAINT chk_share_permission_type 
CHECK (permission_type IN ('VIEW', 'EXECUTE', 'EDIT', 'ADMIN'));

-- Add check constraints for numeric ranges
ALTER TABLE workflow_executions 
ADD CONSTRAINT chk_execution_steps 
CHECK (completed_steps >= 0 AND failed_steps >= 0 AND completed_steps + failed_steps <= total_steps);

ALTER TABLE workflow_templates 
ADD CONSTRAINT chk_template_rating 
CHECK (average_rating >= 0 AND average_rating <= 5);

ALTER TABLE workflow_templates 
ADD CONSTRAINT chk_template_time 
CHECK (estimated_time_minutes > 0);

ALTER TABLE workflows 
ADD CONSTRAINT chk_workflow_executions 
CHECK (successful_executions >= 0 AND successful_executions <= total_executions);

-- Add unique constraints
ALTER TABLE workflow_shares 
ADD CONSTRAINT unq_workflow_user_share 
UNIQUE (workflow_id, shared_with_id);

-- Add comments for documentation
COMMENT ON TABLE workflows IS 'Workflows - user-defined data processing pipelines';
COMMENT ON TABLE workflow_executions IS 'Workflow execution records - tracks workflow runs';
COMMENT ON TABLE workflow_templates IS 'Workflow templates - reusable workflow patterns';
COMMENT ON TABLE workflow_shares IS 'Workflow sharing - permissions for workflow access';

COMMENT ON COLUMN workflows.nodes IS 'Workflow nodes - list of processing steps';
COMMENT ON COLUMN workflows.edges IS 'Workflow edges - connections between nodes';
COMMENT ON COLUMN workflows.chat_history IS 'LangGraph conversation history for workflow creation';
COMMENT ON COLUMN workflows.generation_prompt IS 'Original user prompt that generated this workflow';
COMMENT ON COLUMN workflow_executions.failed_node_id IS 'Node ID where execution failed';
COMMENT ON COLUMN workflow_shares.share_token IS 'Public sharing token for anonymous access';

-- Create views for common queries

-- View for active workflows with execution stats
CREATE VIEW active_workflows_with_stats AS
SELECT 
    w.*,
    COALESCE(exec_stats.recent_executions, 0) as recent_executions,
    COALESCE(exec_stats.avg_processing_time, 0) as avg_processing_time_ms,
    COALESCE(exec_stats.success_rate, 0) as success_rate
FROM workflows w
LEFT JOIN (
    SELECT 
        workflow_id,
        COUNT(*) as recent_executions,
        AVG(processing_time_ms) as avg_processing_time,
        ROUND(
            (COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) * 100.0 / COUNT(*)), 2
        ) as success_rate
    FROM workflow_executions 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
    GROUP BY workflow_id
) exec_stats ON w.id = exec_stats.workflow_id
WHERE w.status IN ('ACTIVE', 'DRAFT');

-- View for public workflows
CREATE VIEW public_workflows AS
SELECT 
    w.*,
    COUNT(ws.id) as share_count,
    SUM(ws.access_count) as total_access_count
FROM workflows w
LEFT JOIN workflow_shares ws ON w.id = ws.workflow_id
WHERE w.is_public = TRUE OR ws.shared_with_id IS NULL
GROUP BY w.id, w.name, w.description, w.owner_id, w.nodes, w.edges, 
         w.chat_history, w.generation_prompt, w.generation_metadata,
         w.settings, w.variables, w.status, w.version, w.is_template,
         w.last_execution_id, w.total_executions, w.successful_executions,
         w.estimated_credits_per_run, w.total_credits_consumed,
         w.tags, w.category, w.is_public, w.created_at, w.updated_at;

-- View for workflow execution summary
CREATE VIEW workflow_execution_summary AS
SELECT 
    w.id as workflow_id,
    w.name as workflow_name,
    w.owner_id,
    COUNT(we.id) as total_executions,
    COUNT(CASE WHEN we.status = 'SUCCESS' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN we.status = 'FAILED' THEN 1 END) as failed_executions,
    COUNT(CASE WHEN we.status IN ('QUEUED', 'RUNNING', 'PAUSED') THEN 1 END) as active_executions,
    AVG(we.processing_time_ms) as avg_processing_time,
    SUM(we.credits_consumed) as total_credits_consumed,
    MAX(we.created_at) as last_execution_at
FROM workflows w
LEFT JOIN workflow_executions we ON w.id = we.workflow_id
GROUP BY w.id, w.name, w.owner_id;

-- Create function to calculate workflow complexity
CREATE OR REPLACE FUNCTION calculate_workflow_complexity(workflow_id UUID)
RETURNS INTEGER AS $
DECLARE
    node_count INTEGER;
    edge_count INTEGER;
    complexity_score INTEGER;
BEGIN
    SELECT 
        jsonb_array_length(nodes),
        jsonb_array_length(edges)
    INTO node_count, edge_count
    FROM workflows 
    WHERE id = workflow_id;
    
    -- Simple complexity calculation: nodes + (edges * 0.5)
    complexity_score := COALESCE(node_count, 0) + COALESCE(ROUND(edge_count * 0.5), 0);
    
    RETURN complexity_score;
END;
$ LANGUAGE plpgsql;

-- Create function to update workflow execution counts
CREATE OR REPLACE FUNCTION update_workflow_execution_counts()
RETURNS TRIGGER AS $
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update total executions
        UPDATE workflows 
        SET total_executions = total_executions + 1,
            last_execution_id = NEW.id
        WHERE id = NEW.workflow_id;
        
    ELSIF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        -- Update successful executions count
        IF NEW.status = 'SUCCESS' AND OLD.status != 'SUCCESS' THEN
            UPDATE workflows 
            SET successful_executions = successful_executions + 1
            WHERE id = NEW.workflow_id;
        ELSIF OLD.status = 'SUCCESS' AND NEW.status != 'SUCCESS' THEN
            UPDATE workflows 
            SET successful_executions = successful_executions - 1
            WHERE id = NEW.workflow_id;
        END IF;
        
        -- Update credits consumed
        UPDATE workflows 
        SET total_credits_consumed = total_credits_consumed + (NEW.credits_consumed - COALESCE(OLD.credits_consumed, 0))
        WHERE id = NEW.workflow_id;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$ LANGUAGE plpgsql;

-- Create trigger to maintain workflow execution counts
CREATE TRIGGER trigger_update_workflow_execution_counts
    AFTER INSERT OR UPDATE ON workflow_executions
    FOR EACH ROW EXECUTE FUNCTION update_workflow_execution_counts();