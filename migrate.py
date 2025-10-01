import sys
from pathlib import Path
from sqlalchemy import inspect, text

# Add project root to the Python path.
# The script is inside 'app', so we need to go up one level to the project root.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from app.database import engine
from app import models

def run_migration():
    """
    A robust migration script that automatically detects and adds missing
    columns to all tables defined in models.py.
    This script is idempotent and safe to run multiple times.
    """
    print("--- Starting Database Migration ---")

    # First, ensure all tables exist. This is safe to run even if they do.
    print("Ensuring database tables exist...")
    models.Base.metadata.create_all(bind=engine)
    print("✅ Tables checked.")

    inspector = inspect(engine)

    with engine.connect() as connection:
        # Use a transaction to ensure all or no changes are made
        with connection.begin():
            print("🔬 Inspecting database schema for missing columns...")
            # Get all tables defined in our models
            for table_name, table in models.Base.metadata.tables.items():
                print(f"   - Checking table: '{table_name}'")
                
                # Get existing columns from the database for this table
                existing_columns = [col['name'] for col in inspector.get_columns(table_name)]
                
                # Get expected columns from our SQLAlchemy model and find what's missing
                for column in table.columns:
                    if column.name not in existing_columns:
                        # Construct and execute the ALTER TABLE statement
                        col_type = column.type.compile(engine.dialect)
                        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}"
                        print(f"     ⚠️ Found missing column '{column.name}'. Adding it now...")
                        connection.execute(text(alter_sql))
                        print(f"     ✅ Column '{column.name}' added successfully.")

    print("✅ Migration check complete. Database schema is up to date.")

if __name__ == "__main__":
    run_migration()