import sys
import os
import secrets
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))

from app.database import SessionLocal, engine
from app import models, hashing
from app.cli_utils import get_confirmed_password

def create_admin_user():
    """
    A standalone script to create the initial admin user.
    """
    # Create tables if they don't exist
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("Creating initial admin user...")
        
        while True:
            username = input("Enter username: ")
            if db.query(models.User).filter(models.User.username == username).first():
                print("Username already exists. Please choose another.")
            else:
                break

        password = get_confirmed_password()
        hashed_password = hashing.Hash.bcrypt(password)

        # Generate a unique Kommo key (9 bytes gives 12 URL-safe characters)
        kommo_key = secrets.token_urlsafe(9)

        new_user = models.User(
            username=username, 
            hashed_password=hashed_password,
            kommo_integration_key=kommo_key,
            profile=models.UserProfile() # Create an empty profile
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"✅ User '{username}' created successfully!")
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()