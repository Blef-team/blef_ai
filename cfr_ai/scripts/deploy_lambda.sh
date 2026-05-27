#!/usr/bin/env bash
#
# Build the cfr_ai Lambda container image and push it to ECR, then update
# the Lambda function code. Modeled on nfsp_ai/scripts/deploy_lambda.sh.
#
# Bundles ALL 66 strategies (strategy.npz + strategy.abs.json) plus the
# Python package into one container image. The old dispatcher + 66-worker
# Lambda topology is retired; one Lambda serves every (hand_size_a, hand_size_b).
#
# Env vars (required):
#   ACCT     AWS account ID
#   REGION   AWS region code (e.g. eu-west-2 for London, eu-central-1 for Frankfurt)
#
# Env vars (optional):
#   PROFILE   AWS CLI profile name (passed as --profile to aws). Unset = default profile.
#   REPO      ECR repo name (default: blef-cfr-lambda)
#   TAG       Image tag (default: lambda-compatible)
#   FUNCTION  Lambda function name to update (default: blef-aiagent-cfr-container)
#   SKIP_LAMBDA_UPDATE  Set to 1 to push image but not call update-function-code
#
# Example:
#   export ACCT=123456789012 REGION=eu-west-2 PROFILE=a
#   ./cfr_ai/scripts/deploy_lambda.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFR_DIR="${ROOT_DIR}/cfr_ai"
IMAGE_NAME="blef-cfr-lambda:lambda-compatible"

if [[ -z "${ACCT:-}" || -z "${REGION:-}" ]]; then
  cat <<EOF
Environment variables ACCT and REGION must be set before running this script.
  ACCT   - AWS account ID (numeric)
  REGION - AWS region code (e.g., eu-west-2)

Example:
  export ACCT=123456789012
  export REGION=eu-west-2
EOF
  exit 1
fi

AWS_PROFILE_ARG=()
if [[ -n "${PROFILE:-}" ]]; then
  AWS_PROFILE_ARG=(--profile "${PROFILE}")
  echo "[info] Using AWS profile: ${PROFILE}"
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "[error] aws CLI not found in PATH" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker not found in PATH" >&2
  exit 1
fi

# Sanity: every (x, y) for 1 <= x <= y <= 11 must have a strategy.npz.
echo "[check] Verifying all 66 strategy.npz files are present..."
MISSING=0
for x in 1 2 3 4 5 6 7 8 9 10 11; do
  for y in 1 2 3 4 5 6 7 8 9 10 11; do
    if [[ $y -ge $x ]]; then
      setup="${x}_${y}"
      if [[ ! -f "${CFR_DIR}/outputs/${setup}/strategy.npz" ]]; then
        echo "[error] missing cfr_ai/outputs/${setup}/strategy.npz" >&2
        MISSING=$((MISSING+1))
      fi
    fi
  done
done
if [[ $MISSING -gt 0 ]]; then
  echo "[error] ${MISSING} strategy file(s) missing; aborting" >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Stage just the runtime payload (cfr_ai package minus dev cruft, plus
# lambda_function.py and the build files) via the Python stager, so this
# script works from git-bash on Windows without needing rsync installed.
echo "[stage] Staging files into ${TEMP_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" "${CFR_DIR}/scripts/stage_for_docker.py" "${TEMP_DIR}" --root "${ROOT_DIR}"

# Sanity: report staged size.
echo "[stage] Staged payload size:"
du -sh "${TEMP_DIR}" || true

echo "[build] Building Docker image ${IMAGE_NAME}"
DOCKER_BUILDKIT=0 docker build --platform=linux/arm64 -t "${IMAGE_NAME}" "${TEMP_DIR}"

REPO="${REPO:-blef-cfr-lambda}"
TAG="${TAG:-lambda-compatible}"
ECR="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"

echo "[login] Logging into ECR ${ACCT}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "${REGION}" "${AWS_PROFILE_ARG[@]}" \
  | docker login --username AWS --password-stdin "${ACCT}.dkr.ecr.${REGION}.amazonaws.com"

# Create the ECR repo if it doesn't exist yet. Idempotent.
if ! aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" "${AWS_PROFILE_ARG[@]}" >/dev/null 2>&1; then
  echo "[ecr] Creating ECR repository ${REPO}"
  aws ecr create-repository --repository-name "${REPO}" --region "${REGION}" "${AWS_PROFILE_ARG[@]}" >/dev/null
fi

echo "[tag] Tagging image ${IMAGE_NAME} -> ${ECR}"
docker tag "${IMAGE_NAME}" "${ECR}"

echo "[push] Pushing ${ECR}"
docker push "${ECR}"

if [[ "${SKIP_LAMBDA_UPDATE:-0}" == "1" ]]; then
  echo "[done] Image pushed; SKIP_LAMBDA_UPDATE=1 so not updating Lambda."
  echo "       ECR image URI: ${ECR}"
  exit 0
fi

FUNCTION="${FUNCTION:-blef-aiagent-cfr-container}"
echo "[lambda] Updating Lambda function ${FUNCTION} to image ${ECR}"
aws lambda update-function-code \
  --function-name "${FUNCTION}" \
  --image-uri "${ECR}" \
  --region "${REGION}" \
  "${AWS_PROFILE_ARG[@]}"

echo "[done] Deployment complete."
