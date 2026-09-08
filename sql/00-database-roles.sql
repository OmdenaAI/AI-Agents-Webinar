-- LiteLLM Proxy stores virtual keys, budgets and spend here.
-- Separate database, same instance: budget state must survive a proxy restart.
CREATE DATABASE litellm;
