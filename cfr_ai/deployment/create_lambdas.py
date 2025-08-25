# Script for creating all 66 worker lambdas.
# Requires a pre-generated zip file.
# Fill in the constants.
import subprocess
import sys
import os

ROLE_ARN = ""
LAYERS = "" # A NumPy layer. Create one first if you don't have one
ZIP_FILE = ""

if not os.path.exists(ZIP_FILE):
    print(f"❌ The required zip file '{ZIP_FILE}' was not found.")
    sys.exit(1)

print("Creating 66 Lambda functions...")
for x in range(1, 12):
    for y in range(x, 12):
        function_name = f"blef-aiagent-cfr-worker-{x}-{y}"
        print(f"\nAttempting to create function: {function_name}")
        command = [
            "aws", "lambda", "create-function",
            "--function-name", function_name,
            "--runtime", "python3.12",
            "--role", ROLE_ARN,
            "--layers", LAYERS,
            "--handler", "lambda_function.lambda_handler",
            "--zip-file", f"fileb://{ZIP_FILE}"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ Successfully created {function_name}")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            if "ResourceConflictException" in error_message:
                print(f"🟡 INFO: Function {function_name} already exists. Skipping.")
            else:
                print(f"❌ ERROR on {function_name}: {error_message.strip()}")
        except FileNotFoundError:
            print("❌ The 'aws' command was not found. Ensure AWS CLI is installed and in your system's PATH")
            sys.exit(1)
