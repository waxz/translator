from .__version__ import VERSION


from . import app

def main():
    """Main entry point for the package."""
    app.run_server()

# Optionally expose other important items at package level
__all__ = ["app", "server"]
