"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from domain_calibrator import DomainCalibratorPy, Domain, Calibrator


@pytest.fixture
def test_domain_config():
    """Get test domain configuration."""
    return {
        'app_key': 'test_key',
        'app_secret': 'test_secret',
        'api_base_url': 'https://api.test.com',
        'dds_base_url': 'https://dds.test.com',
        'map_endpoint': 'https://map.test.com'
    }


@pytest.fixture
def test_camera_matrix():
    """Get test camera matrix."""
    return np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)


@pytest.fixture
def test_dist_coeffs():
    """Get test distortion coefficients."""
    return np.zeros(5, dtype=np.float64)


@pytest.fixture
def mock_domain(mocker):
    """Create a mocked Domain instance."""
    domain = Mock(spec=Domain)
    domain.auth_domain = Mock(return_value=(True, ''))
    domain.get_domain_id = Mock(return_value='test-domain-id')
    domain.fetch_portal_poses = Mock(return_value=(True, ''))
    domain.portals = Mock(return_value={})
    domain.get_map = Mock(return_value=(None, None))
    return domain


@pytest.fixture
def mock_calibrator(mocker):
    """Create a mocked Calibrator instance."""
    calibrator = Mock(spec=Calibrator)
    calibrator.detect_markers = Mock(return_value=[])
    calibrator.preprocess_image = Mock(return_value=(np.zeros((100, 100), dtype=np.uint8), 1.0))
    calibrator.camera_pose = Mock(return_value=None)
    return calibrator
