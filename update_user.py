import sys
from pathlib import Path

# Add project root to the Python path to allow importing from 'app'
sys.path.append(str(Path(__file__).parent.resolve()))

from app.database import SessionLocal
from app import models, hashing
from app.cli_utils import get_confirmed_password

def update_user_password():
    """
    A standalone script to update an existing user's password.
    """
    db = SessionLocal()
    try:
        print("--- Update User Password ---")
        
        username = input("Enter the username of the user to update: ")
        user = db.query(models.User).filter(models.User.username == username).first()

        if not user:
            print(f"❌ User '{username}' not found. Aborting.")
            return

        password = get_confirmed_password()
        user.hashed_password = hashing.Hash.bcrypt(password)
        db.commit()
        print(f"✅ Password for user '{username}' updated successfully!")
    finally:
        db.close()

if __name__ == "__main__":
    update_user_password()