-- Scoreline: database schema
-- Database: PostgreSQL (Supabase)
-- Tables are listed in dependency order.

CREATE TABLE users (
    user_id INT PRIMARY KEY
        GENERATED ALWAYS AS IDENTITY (START WITH 100 MINVALUE 100 MAXVALUE 999),
    first_name VARCHAR(50) NOT NULL,
    last_name  VARCHAR(50) NOT NULL,
    email      VARCHAR(100) UNIQUE,
    phone      VARCHAR(20)  UNIQUE,
    CONSTRAINT user_id_3_digits CHECK (user_id BETWEEN 100 AND 999),
    CONSTRAINT contact_required CHECK (email IS NOT NULL OR phone IS NOT NULL)
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE TABLE user_passwords (
    user_id INT PRIMARY KEY
        REFERENCES users(user_id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE user_passwords ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON user_passwords FROM anon, authenticated;