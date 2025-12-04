#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify that all modules are imported correctly
"""

import os
import sys

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting module import tests...")

# Test models module
print("\nTesting models module...")
try:
    from mtt.models import CustomBertForSequenceClassification
    print("✓ CustomBertForSequenceClassification imported successfully")
    # Print class information
    print(f"  - Class name: {CustomBertForSequenceClassification.__name__}")
except Exception as e:
    print(f"✗ models module import failed: {str(e)}")

# Test datasets module
print("\nTesting datasets module...")
try:
    from mtt.datasets import EmbeddingDataset, CustomDataCollator
    print("✓ EmbeddingDataset imported successfully")
    print("✓ CustomDataCollator imported successfully")
except Exception as e:
    print(f"✗ datasets module import failed: {str(e)}")

# Test utility functions module
print("\nTesting utils module...")
try:
    from mtt.utils import setup_optimizer, compute_distillation_loss, create_training_schedule, load_model_pairs
    print("✓ setup_optimizer imported successfully")
    print("✓ compute_distillation_loss imported successfully")
    print("✓ create_training_schedule imported successfully")
    print("✓ load_model_pairs imported successfully")
except Exception as e:
    print(f"✗ utils module import failed: {str(e)}")

# Test configs module
print("\nTesting configs module...")
try:
    from mtt.configs import Config
    print("✓ Config imported successfully")
    # Create a config instance and print information
    config = Config()
    print(f"  - Config class has parser: {hasattr(config, 'parser')}")
except Exception as e:
    print(f"✗ configs module import failed: {str(e)}")

# Test scripts module
print("\nTesting scripts module...")
try:
    # We don't import the entire script here, just check if it exists
    import os
    script_path = os.path.join("mtt", "scripts", "train_stu_loop_all_hid_cls.py")
    if os.path.exists(script_path):
        print(f"✓ Training script exists: {script_path}")
    else:
        print(f"✗ Training script does not exist: {script_path}")
except Exception as e:
    print(f"✗ scripts module check failed: {str(e)}")

print("\nTests completed!")