# PostgreSQL Setup for FRED Pipeline

## Prerequisites

### Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Install Python PostgreSQL Driver

```bash
pip install psycopg2-binary
```

## Database Setup

### 1. Create Database

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL prompt:
CREATE DATABASE fred;
CREATE USER fred_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE fred TO fred_user;
\q
```

### 2. Set Environment Variables

```bash
# Add to your ~/.bashrc or ~/.zshrc
export POSTGRES_USER='fred_user'
export POSTGRES_PASSWORD='your_password_here'

# Or set for current session
export POSTGRES_PASSWORD='your_password_here'
```

### 3. Initialize Schema

```bash
# Option A: Using psql directly
psql -U fred_user -d fred -f create_tables.sql

# Option B: Using the Python script
python3 03_load_to_postgres.py --init --user fred_user --password 'your_password'
```

## Connection Parameters

Default connection settings:
- **Host:** localhost
- **Port:** 5432
- **Database:** fred
- **User:** postgres (or set POSTGRES_USER)
- **Password:** (set POSTGRES_PASSWORD env var)

## Usage Examples

### Load Data

```bash
# With environment variables set
export POSTGRES_PASSWORD='your_password'
python3 03_load_to_postgres.py --init --data data/fred_series_metadata.parquet

# With explicit parameters
python3 03_load_to_postgres.py \
  --host localhost \
  --port 5432 \
  --database fred \
  --user fred_user \
  --password 'your_password' \
  --init \
  --data data/fred_series_metadata.parquet
```

### Query Data

```bash
# Connect to database
psql -U fred_user -d fred

# Run queries
SELECT category, COUNT(*) 
FROM series_metadata 
GROUP BY category 
ORDER BY COUNT(*) DESC;

SELECT * FROM high_popularity_series LIMIT 10;
```

## Full Pipeline with PostgreSQL

```bash
# 1. Set credentials
export FRED_API_KEY='your-fred-key'
export POSTGRES_PASSWORD='your-postgres-password'

# 2. Run pipeline
python3 99_pipeline.py --full
```

## Troubleshooting

### "password authentication failed"
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Reset password
sudo -u postgres psql
ALTER USER fred_user WITH PASSWORD 'new_password';
```

### "database does not exist"
```bash
sudo -u postgres psql
CREATE DATABASE fred;
GRANT ALL PRIVILEGES ON DATABASE fred TO fred_user;
```

### "connection refused"
```bash
# Check PostgreSQL is listening
sudo netstat -plnt | grep 5432

# Edit postgresql.conf if needed
sudo nano /etc/postgresql/*/main/postgresql.conf
# Set: listen_addresses = 'localhost'
```

## Remote PostgreSQL

If using a remote PostgreSQL server:

```bash
python3 03_load_to_postgres.py \
  --host your-server.com \
  --port 5432 \
  --database fred \
  --user your_user \
  --password 'your_password' \
  --init \
  --data data/fred_series_metadata.parquet
```

## Security Best Practices

1. **Never commit passwords to git**
   ```bash
   # Use environment variables
   export POSTGRES_PASSWORD='...'
   ```

2. **Use .pgpass file for automatic authentication**
   ```bash
   echo "localhost:5432:fred:fred_user:your_password" >> ~/.pgpass
   chmod 600 ~/.pgpass
   ```

3. **Restrict database access**
   ```sql
   REVOKE ALL ON DATABASE fred FROM PUBLIC;
   GRANT CONNECT ON DATABASE fred TO fred_user;
   ```
