#!/bin/bash

# Output CSV file
output_file="secrets_export.csv"
echo "Name,ARN,SecretValue" > "$output_file"

# Get all secrets
secret_names=$(aws secretsmanager list-secrets --query "SecretList[].Name" --output text --profile dentsu-cc --region ap-northeast-1)

# Loop through each secret
for name in $secret_names; do
  echo "Fetching: $name"
  
  arn=$(aws secretsmanager describe-secret --secret-id "$name" --query "ARN" --output text --profile dentsu-cc --region ap-northeast-1)
  value=$(aws secretsmanager get-secret-value --secret-id "$name" --query "SecretString" --output text 2>/dev/null --profile dentsu-cc --region ap-northeast-1)

  # Escape any commas or newlines in the value
  safe_value=$(echo "$value" | tr -d '\n' | sed 's/"/""/g')

  # Append to CSV
  echo "\"$name\",\"$arn\",\"$safe_value\"" >> "$output_file"
done

echo "✅ Exported to $output_file"