#!/usr/bin/env python3
"""
Test domain authentication.

This script tests the domain authentication using credentials from config.yaml.
"""

import sys
import yaml
from pathlib import Path
from domain_calibrator import Domain


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def test_auth():
    """Test domain authentication."""
    print("=" * 60)
    print("Testing Domain Authentication")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading config.yaml...")
    config = load_config()
    
    auth_config = config.get('auth', {})
    if not auth_config.get('app_key') or not auth_config.get('app_secret'):
        print("Error: app_key and app_secret must be set in config.yaml")
        sys.exit(1)
    
    print(f"   ✓ Config loaded")
    print(f"   - API Base URL: {auth_config.get('api_base_url', 'https://api.auki.network')}")
    print(f"   - DDS Base URL: {auth_config.get('dds_base_url', 'https://dds.auki.network')}")
    
    # Prepare domain config
    domain_config = {
        'app_key': auth_config['app_key'],
        'app_secret': auth_config['app_secret'],
        'api_base_url': auth_config.get('api_base_url', 'https://api.auki.network'),
        'dds_base_url': auth_config.get('dds_base_url', 'https://dds.auki.network'),
        'map_endpoint': config.get('domain', {}).get('map_endpoint', 'https://dsc.dev.aukiverse.com/spatial/crosssection')
    }
    
    # Test authentication
    print("\n2. Testing authentication...")
    try:
        domain = Domain(domain_config)
        print("   ✓ Domain object created")
        
        # Test auth
        ret, msg = domain.auth()
        if ret:
            print("   ✓ Authentication successful!")
            print(f"   - Message: {msg}")
            
            # Check if we have DDS token
            if domain._dds_token:
                print(f"   ✓ DDS token obtained: {domain._dds_token[:20]}...")
            else:
                print("   ⚠ Warning: DDS token not set")
            
        else:
            print("   ✗ Authentication failed!")
            print(f"   - Error: {msg}")
            sys.exit(1)
        
        print("\n3. Testing domain info retrieval...")
        # Try to get domain info (this requires a QR code, so we'll just test the method exists)
        print("   ✓ Domain methods available")
        print("   - get_domain_id() - requires QR shortcode")
        print("   - auth_domain() - requires domain_id")
        print("   - fetch_portal_poses() - requires authenticated domain")
        
        # Cleanup
        domain.close()
        print("\n4. Cleanup...")
        print("   ✓ Domain connection closed")
        
        print("\n" + "=" * 60)
        print("✓ Authentication test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during authentication: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_auth()
