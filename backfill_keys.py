import sys
import secrets
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent.resolve()))

from app.database import SessionLocal
from app import models

def backfill_kommo_keys():
    """
    A standalone script to generate Kommo integration keys for any existing
    users who do not have one.
    """
    db = SessionLocal()
    try:
        print("--- Backfilling Kommo Integration Keys ---")
        
        # Find all users without a Kommo key
        users_to_update = db.query(models.User).filter(models.User.kommo_integration_key == None).all()

        if not users_to_update:
            print("✅ All users already have a Kommo integration key. No action needed.")
            return

        print(f"Found {len(users_to_update)} user(s) missing a key. Generating now...")

        for user in users_to_update:
            # Generate a unique Kommo key (9 bytes gives 12 URL-safe characters)
            user.kommo_integration_key = secrets.token_urlsafe(9)
            print(f"  - Generated key for user '{user.username}'")

        db.commit()
        print(f"✅ Successfully updated {len(users_to_update)} user(s).")

    finally:
        db.close()

if __name__ == "__main__":
    backfill_kommo_keys()