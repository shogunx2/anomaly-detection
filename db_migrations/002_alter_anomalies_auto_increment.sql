-- Migration to alter existing anomalies table to use auto-increment
-- This handles the case where the table already exists with BIGINT id

-- First, create a sequence for the id
CREATE SEQUENCE IF NOT EXISTS anomalies_id_seq;

-- Set the sequence to start from the maximum id value + 1
SELECT setval('anomalies_id_seq', COALESCE((SELECT MAX(id) FROM anomalies), 0) + 1);

-- Alter the table to use the sequence
ALTER TABLE anomalies ALTER COLUMN id SET DEFAULT nextval('anomalies_id_seq');
ALTER TABLE anomalies ALTER COLUMN id SET DATA TYPE INTEGER USING id::INTEGER;

-- Make sure the sequence is owned by the id column
ALTER SEQUENCE anomalies_id_seq OWNED BY anomalies.id; 