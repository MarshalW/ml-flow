#!/bin/bash

# Check if ollama is installed
if ! command -v ollama &> /dev/null
then
    echo "ollama is not installed. Please install ollama first."
    echo "You can visit https://ollama.com/ for installation."
    exit 1
fi

# 1. Temporarily run ollama serve in the background
echo "Starting ollama service..."

# Check if ollama serve is already running
if pgrep -x "ollama" > /dev/null; then
    echo "ollama service is already running. Skipping start."
    # If ollama is already running, we don't have its PID from this script.
    # For subsequent steps that might need the PID, consider finding it or
    # ensure they don't strictly depend on the PID from *this* script's start.
    # For this script's purpose (ollama create), the service just needs to be up.
else
    ollama serve & # Run the command in the background using &
    OLLAMA_PID=$! # Get the PID of the ollama serve process
    echo "ollama service started in the background, PID: $OLLAMA_PID"

    # Wait for ollama service to start, adjust waiting time as needed
    echo "Waiting for ollama service to start (5 seconds)..."
    sleep 5
fi

# 2. Parse model size from DEFAULT_MODEL_NAME environment variable
# Assuming DEFAULT_MODEL_NAME format is "Qwen/Qwen3-4B" or similar "Llama/Llama2-7B"
# Extract the part after the last slash, then the part after the last hyphen
if [ -z "$DEFAULT_MODEL_NAME" ]; then
    echo "Error: Environment variable DEFAULT_MODEL_NAME is not set."
    echo "Please set DEFAULT_MODEL_NAME, e.g.: export DEFAULT_MODEL_NAME=\"Qwen/Qwen3-4B\""
    # Try to set a default value to prevent script interruption
    MODEL_SIZE="4B" # Default value
    echo "Using default model size: $MODEL_SIZE"
else
    # Extract the model name part (e.g., "Qwen3-4B")
    MODEL_NAME_PART=$(basename "$DEFAULT_MODEL_NAME")
    # Extract the size from the model name part (e.g., "4B")
    # Use regex to match the part after the last hyphen
    if [[ "$MODEL_NAME_PART" =~ -([0-9]+[BKMGT]+)$ ]]; then
        MODEL_SIZE="${BASH_REMATCH[1]}"
        echo "Parsed model size from DEFAULT_MODEL_NAME: $MODEL_SIZE"
    else
        echo "Warning: Unable to parse model size from DEFAULT_MODEL_NAME ('$DEFAULT_MODEL_NAME')."
        echo "Please ensure DEFAULT_MODEL_NAME contains a size suffix like '-4B'."
        MODEL_SIZE="4B" # Default value
        echo "Using default model size: $MODEL_SIZE"
    fi
fi

# Convert MODEL_SIZE to lowercase for the full model tag
LOWERCASE_MODEL_SIZE=$(echo "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')

# Construct the full model name with lowercase size
FULL_MODEL_TAG="marshalw/my-nocobase-qwen3-lora-${LOWERCASE_MODEL_SIZE}:0.1"
echo "Ollama model tag to be created: $FULL_MODEL_TAG"

# 3. Execute ollama create command
echo "Creating Ollama model: $FULL_MODEL_TAG ..."
ollama create "$FULL_MODEL_TAG" -f my-nocobase-model2

# Check the exit status of the ollama create command
if [ $? -eq 0 ]; then
    echo "Ollama model '$FULL_MODEL_TAG' created successfully."

    # 4. Push the created Ollama model
    echo "Pushing Ollama model: $FULL_MODEL_TAG ..."
    ollama push "$FULL_MODEL_TAG"

    # Check the exit status of the ollama push command
    if [ $? -eq 0 ]; then
        echo "Ollama model '$FULL_MODEL_TAG' pushed successfully."
    else
        echo "Ollama model '$FULL_MODEL_TAG' push failed."
    fi
else
    echo "Ollama model '$FULL_MODEL_TAG' creation failed. Skipping push."
fi

# Prompt the user how to stop the ollama serve process
echo ""
# Only show the kill message if we actually started ollama serve in this script
if [ -n "$OLLAMA_PID" ]; then
    echo "Note: The ollama serve process (PID: $OLLAMA_PID) is still running in the background."
    echo "If you want to stop it, please run: kill $OLLAMA_PID"
else
    echo "Note: ollama serve was already running when the script started."
    echo "If you wish to stop it, you may need to find its PID using 'pgrep ollama' and then 'kill <PID>'."
fi
