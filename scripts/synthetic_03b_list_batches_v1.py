#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.api_config import client


def main():
    batches = client.batches.list(limit=20)

    print("Recent batches:\n")
    for b in batches.data:
        print(f"Batch ID        : {b.id}")
        print(f"Status          : {b.status}")
        print(f"Created at      : {getattr(b, 'created_at', None)}")
        print(f"In progress at  : {getattr(b, 'in_progress_at', None)}")
        print(f"Completed at    : {getattr(b, 'completed_at', None)}")
        print(f"Output file ID  : {getattr(b, 'output_file_id', None)}")
        print(f"Error file ID   : {getattr(b, 'error_file_id', None)}")
        print("-" * 60)


if __name__ == "__main__":
    main()