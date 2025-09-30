#!/usr/bin/env python3
"""
Initialize ETHEREYE database
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.models.database import init_database
    
    print("🗄️  Initializing ETHEREYE database...")
    init_database()
    print("✅ Database initialized successfully!")
    
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    sys.exit(1)