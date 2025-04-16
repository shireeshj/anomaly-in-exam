import os
import sys

print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import main
    print("Successfully imported main module")
    main.main()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
