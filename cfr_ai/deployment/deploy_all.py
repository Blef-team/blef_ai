import subprocess
import sys

print("Deploying all 66 setups...")
for x in range(1, 12):
    for y in range(x, 12):
        function_name = f"blef-aiagent-cfr-worker-{x}-{y}"
        print(f"\nAttempting to deploy setup: {x},{y}")
        command = ["python", "-m", "cfr_ai.deployment.deploy", "--hand-sizes", str(x), str(y)]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ Successfully deployed {x},{y}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR on {function_name}: {e.stderr.strip()}")
        except FileNotFoundError:
            print("❌ The 'aws' command was not found. Ensure AWS CLI is installed and in your system's PATH")
            sys.exit(1)

print("\n🎉 Script finished.")
