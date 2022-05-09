import sys

def use_legacy_asyncio():
    if sys.version_info >= (3, 8, 10):
        print("Using latest asyncio implementation.")
        return False
    elif sys.version_info >= (3, 6, 9):
        print("Using legacy asyncio implementation.")
        return True
    else:
        print("WARNING: The used Python interpreter is older than what is officially supported. This may cause some functionalities to break")
        return True
