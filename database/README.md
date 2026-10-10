# Database (Supabase / PostgreSQL)

Tables are defined in `schema.sql`:
- `users`: user ID (100-999), first and last name, email and/or phone (at least one required)
- `user_passwords`: password hash, linked to `users` by foreign key

To set up: open the Supabase SQL Editor and run `schema.sql`.

Never commit the database password or connection string. Keep them in a local `.env` file (see `.env.example`).