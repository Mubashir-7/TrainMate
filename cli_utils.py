from getpass import getpass

def get_confirmed_password() -> str:
    """
    A helper function to securely prompt for and confirm a password.
    It enforces a minimum length and ensures the two entries match.
    """
    while True:
        password = getpass("Enter password (min 8 characters): ")
        if len(password) < 8:
            print("❌ Password must be at least 8 characters long. Please try again.")
            continue
        
        password_confirm = getpass("Confirm password: ")
        if password == password_confirm:
            return password
        
        print("❌ Passwords do not match. Please try again.")