from config.api_config import client

# replace with batch id you want to cancel
BATCH_ID = "batch_69dfd2c34b9c8190b477cfd72614ea40"

batch = client.batches.cancel(BATCH_ID)

print(f"Cancelled batch: {batch.id}")
print(f"Status: {batch.status}")