import os
import sys
import shutil
import zipfile
import argparse
import boto3
from botocore.exceptions import ClientError

def create_lambda_zip(setup_name: str, build_dir: str, zip_path: str) -> None:
    """
    Assembles the required files in a temporary directory and creates a zip archive.
    """

    print(f"Assembling files for setup '{setup_name}'...")

    cfr_ai_dir = 'cfr_ai'
    outputs_dir = os.path.join(cfr_ai_dir, 'outputs', setup_name)
    
    # Check if the required outputs directory exists
    if not os.path.isdir(outputs_dir):
        print(f"Error: Output directory not found at '{outputs_dir}'")
        sys.exit(1)

    target_outputs_dir = os.path.join(build_dir, 'cfr_ai', 'outputs', setup_name)
    os.makedirs(target_outputs_dir, exist_ok=True)

    # Files to be copied
    root_files = {
        'lambda_function.py': os.path.join(cfr_ai_dir, 'lambda_function.py'),
        'agent.py': os.path.join(cfr_ai_dir, 'agent.py')
    }
    cfr_ai_files = {
        'information_set.py': os.path.join(cfr_ai_dir, 'information_set.py'),
        'history.csv': os.path.join(cfr_ai_dir, 'history.csv'),
        'encoding.py': os.path.join(cfr_ai_dir, 'encoding.py')
    }

    # Copy root files
    for dest_name, src_path in root_files.items():
        shutil.copy(src_path, os.path.join(build_dir, dest_name))

    # Copy cfr_ai subfolder files
    for dest_name, src_path in cfr_ai_files.items():
        shutil.copy(src_path, os.path.join(build_dir, 'cfr_ai', dest_name))

    # Copy the metadata file
    metadata_path = os.path.join(outputs_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        print(f"  -> Including metadata.csv")
        shutil.copy(metadata_path, os.path.join(build_dir, 'metadata.csv'))
    else:
        print(f"Warning: metadata.csv not found at '{metadata_path}'. Agent will use default parameters.")

    # Copy the specific setup's output directory
    for item_name in os.listdir(outputs_dir):
        source_item_path = os.path.join(outputs_dir, item_name)
        # Check if it's a directory and NOT a diagnostic folder
        if os.path.isdir(source_item_path) and '_diagnostic' not in item_name:
            print(f"  -> Including strategy folder: {item_name}")
            dest_item_path = os.path.join(target_outputs_dir, item_name)
            shutil.copytree(source_item_path, dest_item_path)

    # Create the zip file
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', build_dir)

def deploy_to_lambda(function_name: str, zip_path: str) -> None:
    try:
        print(f"Deploying to Lambda...")
        lambda_client = boto3.client('lambda')
        with open(zip_path, 'rb') as f:
            zipped_code = f.read()
        response = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zipped_code
        )
        print(f"Successfully updated function '{function_name}'.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"Error: Lambda function '{function_name}' not found.")
        else:
            print(f"An AWS error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and deploy a specific Blef CFR AI worker to AWS Lambda.")
    parser.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards per player (e.g. --hand-sizes 1 1)")
    args = parser.parse_args()
    
    hand_sizes = sorted(args.hand_sizes)
    setup_name = f"{hand_sizes[0]}_{hand_sizes[1]}"
    lambda_function_name = f"blef-aiagent-cfr-worker-{hand_sizes[0]}-{hand_sizes[1]}"
    build_dir = "lambda_build_temp"
    zip_filename = f"blef_worker_{setup_name}.zip"

    try:
        create_lambda_zip(setup_name, build_dir, zip_filename)
        deploy_to_lambda(lambda_function_name, zip_filename)

    finally:
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            pass
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            pass


if __name__ == "__main__":
    main()
