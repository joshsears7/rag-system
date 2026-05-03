#!/bin/bash
# Deploy the full rag_system repo to Hugging Face Spaces.
#
# One-time setup:
#   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/rag-system
#
# Then deploy anytime with:
#   bash hf_space/deploy.sh

set -e

HF_REMOTE="${HF_REMOTE:-hf}"

# Confirm the remote exists
if ! git remote get-url "$HF_REMOTE" &>/dev/null; then
    echo "ERROR: remote '$HF_REMOTE' not found."
    echo "Run: git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/rag-system"
    exit 1
fi

echo "Pushing to HF Spaces remote: $HF_REMOTE ..."
git push "$HF_REMOTE" main

echo ""
echo "Deployed. Set these secrets in your Space settings:"
echo "  ANTHROPIC_API_KEY = <your key>"
echo "  LLM_BACKEND       = claude"
