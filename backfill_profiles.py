import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from app.database import SessionLocal
from app import models

def backfill_user_profiles():
    """
    A standalone script to create a UserProfile for any existing
    users who do not have one.
    """
    db = SessionLocal()
    try:
        print("--- Backfilling User Profiles ---")
        
        # Find all users without a profile
        users_to_update = db.query(models.User).filter(models.User.profile == None).all()

        if not users_to_update:
            print("✅ All users already have a profile. No action needed.")
            return

        print(f"Found {len(users_to_update)} user(s) missing a profile. Creating now...")

        for user in users_to_update:
            user.profile = models.UserProfile()
            print(f"  - Created profile for user '{user.username}'")

        db.commit()
        print(f"✅ Successfully created profiles for {len(users_to_update)} user(s).")

    finally:
        db.close()

if __name__ == "__main__":
    backfill_user_profiles()