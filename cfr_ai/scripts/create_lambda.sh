#!/usr/bin/env bash
#
# One-time creation of the cfr_ai Lambda function from an ECR image URI.
# Subsequent code updates go through cfr_ai/scripts/deploy_lambda.sh
# (update-function-code).
#
# Pre-reqs:
#   - Image already pushed to ECR (run deploy_lambda.sh with
#     SKIP_LAMBDA_UPDATE=1 first).
#   - You have the execution role ARN. Easiest: reuse the role the existing
#     CFR worker Lambdas already have. Look it up with:
#
#       aws lambda get-function --function-name blef-aiagent-cfr-worker-1-1 \
#         --query 'Configuration.Role' --output text --profile "$PROFILE" \
#         --region "$REGION"
#
# Env vars (required):
#   ACCT       AWS account ID
#   REGION     AWS region (e.g. eu-west-2)
#   ROLE_ARN   IAM role ARN for the function's execution role
#
# Env vars (optional):
#   PROFILE    AWS CLI profile (passed via --profile)
#   REPO       ECR repo (default: blef-cfr-lambda)
#   TAG        Image tag (default: lambda-compatible)
#   FUNCTION   Lambda function name (default: blef-aiagent-cfr-container)
#   MEMORY_MB  Memory size in MB (default: 1024)
#   TIMEOUT_S  Function timeout in seconds (default: 30 — cold start with numba
#              import + biggest strategy load is ~6s; warm calls are ms)
#
# Example:
#   export ACCT=123456789012 REGION=eu-west-2 PROFILE=a
#   export ROLE_ARN="arn:aws:iam::123456789012:role/lambda_basic_execution"
#   ./cfr_ai/scripts/create_lambda.sh

set -euo pipefail

if [[ -z "${ACCT:-}" || -z "${REGION:-}" || -z "${ROLE_ARN:-}" ]]; then
  echo "ACCT, REGION, and ROLE_ARN must be set. See header comment." >&2
  exit 1
fi

AWS_PROFILE_ARG=()
if [[ -n "${PROFILE:-}" ]]; then
  AWS_PROFILE_ARG=(--profile "${PROFILE}")
fi

REPO="${REPO:-blef-cfr-lambda}"
TAG="${TAG:-lambda-compatible}"
ECR="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"
FUNCTION="${FUNCTION:-blef-aiagent-cfr-container}"
MEMORY_MB="${MEMORY_MB:-1024}"
TIMEOUT_S="${TIMEOUT_S:-30}"

# Refuse if the function already exists - prevents accidental recreate.
if aws lambda get-function --function-name "${FUNCTION}" --region "${REGION}" \
    "${AWS_PROFILE_ARG[@]}" >/dev/null 2>&1; then
  echo "[error] Function '${FUNCTION}' already exists in ${REGION}." >&2
  echo "        Use cfr_ai/scripts/deploy_lambda.sh to update its code." >&2
  exit 1
fi

echo "[lambda] Creating function ${FUNCTION} from ${ECR}"
echo "         memory=${MEMORY_MB}MB timeout=${TIMEOUT_S}s arch=arm64"
echo "         role=${ROLE_ARN}"

aws lambda create-function \
  --function-name "${FUNCTION}" \
  --package-type Image \
  --code ImageUri="${ECR}" \
  --role "${ROLE_ARN}" \
  --architectures arm64 \
  --memory-size "${MEMORY_MB}" \
  --timeout "${TIMEOUT_S}" \
  --region "${REGION}" \
  "${AWS_PROFILE_ARG[@]}"

echo "[done] Function ${FUNCTION} created."
echo "       Next steps:"
echo "        - Hook up your invocation source (API Gateway / SQS / direct invoke)."
echo "        - Subsequent code updates: re-run cfr_ai/scripts/deploy_lambda.sh."
