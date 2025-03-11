#!/bin/bash
while ps -p 14637 > /dev/null; do
  echo "Training still running... waiting"
  sleep 300  # Check every 5 minutes
done
echo "Previous training finished, starting next run..."
python -m emg2qwerty.autoencoder_train