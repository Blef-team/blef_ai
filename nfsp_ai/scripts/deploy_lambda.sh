#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NFSP_DIR="${ROOT_DIR}/nfsp_ai"
IMAGE_NAME="blef-nfsp-lambda:lambda-compatible"

if [[ -z "${ACCT:-}" || -z "${REGION:-}" ]]; then
  cat <<EOF
Environment variables ACCT and REGION must be set before running this script.
  ACCT   – AWS account ID (numeric)
  REGION – AWS region code (e.g., us-east-1)

Example:
  export ACCT=123456789012
  export REGION=us-east-1
EOF
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "[error] aws CLI not found in PATH" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker not found in PATH" >&2
  exit 1
fi

declare -a REQUIRED_ARTIFACTS=(
  "artifacts/nfsp_inference_24.pt"
  "artifacts/nfsp_inference_32.pt"
  "artifacts/card_embedding_pretrain_24.pt"
  "artifacts/card_embedding_pretrain_32.pt"
  "artifacts/history_embedding_pretrain_24.pt"
  "artifacts/history_embedding_pretrain_32.pt"
)

for artifact in "${REQUIRED_ARTIFACTS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/${artifact}" ]]; then
    echo "[error] missing artifact '${artifact}'. Please export it before deploying." >&2
    exit 1
  fi
done

TEMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TEMP_DIR}/artifacts"

rsync -a --exclude '__pycache__' "${NFSP_DIR}" "${TEMP_DIR}/"
rsync -a --exclude '__pycache__' "${ROOT_DIR}/shared" "${TEMP_DIR}/"
cp "${NFSP_DIR}/production_agent.py" "${TEMP_DIR}/agent.py"
cp "${ROOT_DIR}/deployment/lambda_function.py" "${TEMP_DIR}/lambda_function.py"
rsync -a "${ROOT_DIR}/artifacts/" "${TEMP_DIR}/artifacts/"
rsync -a --exclude '__pycache__' "${ROOT_DIR}/conservative_ai" "${TEMP_DIR}/"
rsync -a --exclude '__pycache__' "${ROOT_DIR}/conservative_crawling_ai" "${TEMP_DIR}/"

cp "${NFSP_DIR}/deployment/requirements.txt" "${TEMP_DIR}/requirements.txt"
cp "${NFSP_DIR}/deployment/Dockerfile.lambda" "${TEMP_DIR}/Dockerfile"

echo "[build] Building Docker image ${IMAGE_NAME}"
DOCKER_BUILDKIT=0 docker build --platform=linux/arm64 -t "${IMAGE_NAME}" "${TEMP_DIR}"

REPO="${REPO:-blef-nfsp-lambda}"
TAG="${TAG:-lambda-compatible}"
ECR="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"

echo "[login] Logging into ECR ${ACCT}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCT}.dkr.ecr.${REGION}.amazonaws.com"

echo "[tag] Tagging image ${IMAGE_NAME} -> ${ECR}"
docker tag "${IMAGE_NAME}" "${ECR}"

echo "[push] Pushing ${ECR}"
docker push "${ECR}"

FUNCTION="${FUNCTION:-blef-aiagent-nfsp}"
# The function swap is gated behind SWAP=1 so build+push can be done without
# touching production. Swap explicitly (SWAP=1) once the new image is verified.
if [[ "${SWAP:-0}" == "1" ]]; then
  echo "[lambda] Updating Lambda function ${FUNCTION} to image ${ECR}"
  aws lambda update-function-code \
    --function-name "${FUNCTION}" \
    --image-uri "${ECR}"
  echo "[done] Build, push and SWAP complete."
else
  echo "[skip] SWAP!=1 — pushed image ${ECR} but did NOT update ${FUNCTION}."
  echo "       To swap:  aws lambda update-function-code --function-name ${FUNCTION} --image-uri ${ECR} --region ${REGION}"
fi
