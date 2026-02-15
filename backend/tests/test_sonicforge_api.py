"""
SonicForge API Backend Tests (Phases 4, 5, 6)

Tests cover:
- Authentication: register, login, token refresh, get current user
- Voting system: submit vote, get track votes, get top tracks
- Recommendations: get trending, get personalized (with auth)
- Social sharing: create share link, get social URLs
- Subscription plans: get plans, get credit packages, get subscription status
- Analytics: track event, get public stats, prometheus metrics
- A/B testing: list experiments, get variant (with auth), track conversion
- Dashboard: get widgets, system metrics, business metrics
- Health checks: /health, /health/detailed, /ready, /live
"""

import pytest
import requests
import uuid

# Base URL for API testing
BASE_URL = "http://localhost:8001"


@pytest.fixture(scope="module")
def api_client():
    """Shared requests session."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture(scope="module")
def test_user_credentials():
    """Generate unique test user credentials."""
    unique_id = uuid.uuid4().hex[:8]
    return {
        "email": f"test_{unique_id}@example.com",
        "username": f"testuser_{unique_id}",
        "password": "Test1234!"
    }


@pytest.fixture(scope="module")
def admin_user_credentials():
    """Generate unique admin user credentials."""
    unique_id = uuid.uuid4().hex[:8]
    return {
        "email": f"admin_{unique_id}@sonicforge.ai",
        "username": f"admin_{unique_id}",
        "password": "Admin1234!"
    }


@pytest.fixture(scope="module")
def registered_user(api_client, test_user_credentials):
    """Register a test user and return tokens."""
    response = api_client.post(
        f"{BASE_URL}/api/v1/auth/register",
        json=test_user_credentials
    )
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400 and "already registered" in response.text.lower():
        # Try login instead
        login_response = api_client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={
                "email": test_user_credentials["email"],
                "password": test_user_credentials["password"]
            }
        )
        if login_response.status_code == 200:
            return login_response.json()
    pytest.skip(f"Could not register or login test user: {response.text}")


@pytest.fixture(scope="module")
def registered_admin(api_client, admin_user_credentials):
    """Register an admin user and return tokens."""
    response = api_client.post(
        f"{BASE_URL}/api/v1/auth/register",
        json=admin_user_credentials
    )
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400 and "already registered" in response.text.lower():
        # Try login instead
        login_response = api_client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={
                "email": admin_user_credentials["email"],
                "password": admin_user_credentials["password"]
            }
        )
        if login_response.status_code == 200:
            return login_response.json()
    pytest.skip(f"Could not register or login admin user: {response.text}")


@pytest.fixture(scope="module")
def auth_headers(registered_user):
    """Get authentication headers for regular user."""
    return {"Authorization": f"Bearer {registered_user['access_token']}"}


@pytest.fixture(scope="module")
def admin_headers(registered_admin):
    """Get authentication headers for admin user."""
    return {"Authorization": f"Bearer {registered_admin['access_token']}"}


# ===========================================
# Health Check Tests
# ===========================================
class TestHealthChecks:
    """Test health check endpoints."""

    def test_health_endpoint(self, api_client):
        """Test basic health check."""
        response = api_client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        print(f"Health check passed: {data['status']}")

    def test_detailed_health_endpoint(self, api_client):
        """Test detailed health check."""
        response = api_client.get(f"{BASE_URL}/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "overall" in data or "health" in data
        print(f"Detailed health check response: {data}")

    def test_ready_endpoint(self, api_client):
        """Test kubernetes readiness probe."""
        response = api_client.get(f"{BASE_URL}/ready")
        assert response.status_code in [200, 503]  # May fail if dependencies are down
        data = response.json()
        assert "ready" in data
        print(f"Readiness probe: {data}")

    def test_live_endpoint(self, api_client):
        """Test kubernetes liveness probe."""
        response = api_client.get(f"{BASE_URL}/live")
        assert response.status_code == 200
        data = response.json()
        assert "alive" in data or "status" in data
        print(f"Liveness probe: {data}")


# ===========================================
# Authentication Tests (Phase 4)
# ===========================================
class TestAuthentication:
    """Test authentication endpoints."""

    def test_register_new_user(self, api_client):
        """Test user registration."""
        unique_id = uuid.uuid4().hex[:8]
        user_data = {
            "email": f"newuser_{unique_id}@example.com",
            "username": f"newuser_{unique_id}",
            "password": "SecurePass123!"
        }
        response = api_client.post(f"{BASE_URL}/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        print(f"User registered successfully, token type: {data['token_type']}")

    def test_register_invalid_password(self, api_client):
        """Test registration with weak password."""
        unique_id = uuid.uuid4().hex[:8]
        user_data = {
            "email": f"weakpass_{unique_id}@example.com",
            "username": f"weakpass_{unique_id}",
            "password": "123"  # Too weak
        }
        response = api_client.post(f"{BASE_URL}/api/v1/auth/register", json=user_data)
        # Should fail validation
        assert response.status_code in [400, 422]
        print(f"Weak password correctly rejected: {response.status_code}")

    def test_login_success(self, api_client, test_user_credentials, registered_user):
        """Test successful login."""
        response = api_client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={
                "email": test_user_credentials["email"],
                "password": test_user_credentials["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        print("Login successful")

    def test_login_invalid_credentials(self, api_client):
        """Test login with invalid credentials."""
        response = api_client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "WrongPass123!"
            }
        )
        assert response.status_code == 401
        print("Invalid credentials correctly rejected")

    def test_get_current_user(self, api_client, auth_headers):
        """Test getting current user info."""
        response = api_client.get(f"{BASE_URL}/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "roles" in data
        print(f"Current user: {data['email']}, roles: {data['roles']}")

    def test_refresh_token(self, api_client, registered_user):
        """Test token refresh."""
        response = api_client.post(
            f"{BASE_URL}/api/v1/auth/refresh",
            json={"refresh_token": registered_user["refresh_token"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        print("Token refresh successful")

    def test_logout(self, api_client):
        """Test user logout."""
        # First register a fresh user for logout test
        unique_id = uuid.uuid4().hex[:8]
        reg_response = api_client.post(
            f"{BASE_URL}/api/v1/auth/register",
            json={
                "email": f"logout_{unique_id}@example.com",
                "username": f"logout_{unique_id}",
                "password": "LogoutTest123!"
            }
        )
        if reg_response.status_code != 200:
            pytest.skip("Could not register user for logout test")
        
        token = reg_response.json()["access_token"]
        
        response = api_client.post(
            f"{BASE_URL}/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print(f"Logout response: {data['message']}")


# ===========================================
# Voting System Tests (Phase 5)
# ===========================================
class TestVotingSystem:
    """Test voting system endpoints."""

    def test_submit_vote(self, api_client, auth_headers):
        """Test submitting a vote for a track."""
        track_id = f"track_{uuid.uuid4().hex[:8]}"
        vote_data = {"track_id": track_id, "vote": 5}
        response = api_client.post(
            f"{BASE_URL}/api/v1/vote",
            json=vote_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["track_id"] == track_id
        assert data["vote"] == 5
        assert "average_rating" in data
        print(f"Vote submitted: {data['vote']}, avg: {data['average_rating']}")

    def test_submit_invalid_vote(self, api_client, auth_headers):
        """Test submitting invalid vote (out of range)."""
        track_id = f"track_{uuid.uuid4().hex[:8]}"
        vote_data = {"track_id": track_id, "vote": 10}  # Invalid: > 5
        response = api_client.post(
            f"{BASE_URL}/api/v1/vote",
            json=vote_data,
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        print("Invalid vote correctly rejected")

    def test_get_track_votes(self, api_client):
        """Test getting votes for a track (public endpoint)."""
        track_id = "test_track_001"
        response = api_client.get(f"{BASE_URL}/api/v1/vote/{track_id}")
        assert response.status_code == 200
        data = response.json()
        assert "track_id" in data
        assert "total_votes" in data
        assert "average_rating" in data
        print(f"Track {track_id}: {data['total_votes']} votes, avg: {data['average_rating']}")

    def test_get_top_tracks(self, api_client):
        """Test getting top rated tracks (public endpoint)."""
        response = api_client.get(f"{BASE_URL}/api/v1/top-tracks?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"Top tracks count: {len(data)}")

    def test_get_user_vote(self, api_client, auth_headers):
        """Test getting user's vote for a track."""
        # First submit a vote
        track_id = f"track_user_vote_{uuid.uuid4().hex[:8]}"
        api_client.post(
            f"{BASE_URL}/api/v1/vote",
            json={"track_id": track_id, "vote": 4},
            headers=auth_headers
        )
        
        # Then get user's vote
        response = api_client.get(
            f"{BASE_URL}/api/v1/vote/{track_id}/user",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["track_id"] == track_id
        assert data["vote"] == 4
        print(f"User vote for {track_id}: {data['vote']}")


# ===========================================
# Recommendations Tests (Phase 5)
# ===========================================
class TestRecommendations:
    """Test recommendation endpoints."""

    def test_get_trending(self, api_client):
        """Test getting trending tracks (public endpoint)."""
        response = api_client.get(f"{BASE_URL}/api/v1/recommendations/trending")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        print(f"Trending tracks: {len(data['recommendations'])}")

    def test_get_personalized_recommendations(self, api_client, auth_headers):
        """Test getting personalized recommendations (auth required)."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/recommendations/personalized?limit=5",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "user_id" in data
        print(f"Personalized recommendations: {len(data['recommendations'])}")

    def test_get_similar_tracks(self, api_client):
        """Test getting similar tracks."""
        track_id = "test_track_001"
        response = api_client.get(
            f"{BASE_URL}/api/v1/recommendations/similar/{track_id}?limit=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        print(f"Similar tracks to {track_id}: {len(data['recommendations'])}")

    def test_get_genre_recommendations(self, api_client):
        """Test getting genre-based recommendations."""
        genre = "drum_and_bass"
        response = api_client.get(
            f"{BASE_URL}/api/v1/recommendations/genre/{genre}?limit=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        print(f"Genre {genre} recommendations: {len(data['recommendations'])}")

    def test_record_listen(self, api_client, auth_headers):
        """Test recording a track listen."""
        track_id = f"track_{uuid.uuid4().hex[:8]}"
        response = api_client.post(
            f"{BASE_URL}/api/v1/recommendations/listen/{track_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recorded"] == True
        print(f"Listen recorded for {track_id}")


# ===========================================
# Social Sharing Tests (Phase 5)
# ===========================================
class TestSocialSharing:
    """Test social sharing endpoints."""

    def test_create_share_link(self, api_client, auth_headers):
        """Test creating a share link."""
        share_data = {
            "track_id": f"track_{uuid.uuid4().hex[:8]}",
            "title": "Awesome Track",
            "genre": "drum_and_bass"
        }
        response = api_client.post(
            f"{BASE_URL}/api/v1/share",
            json=share_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "share_code" in data
        assert "share_url" in data
        print(f"Share link created: {data['share_code']}")
        return data["share_code"]

    def test_create_share_link_without_auth(self, api_client):
        """Test creating a share link without authentication (should work)."""
        share_data = {
            "track_id": f"track_{uuid.uuid4().hex[:8]}",
            "title": "Public Track",
            "genre": "house_deep"
        }
        response = api_client.post(
            f"{BASE_URL}/api/v1/share",
            json=share_data
        )
        assert response.status_code == 200
        data = response.json()
        assert "share_code" in data
        print(f"Public share link created: {data['share_code']}")

    def test_get_share_urls(self, api_client):
        """Test getting social share URLs."""
        # First create a share link
        share_data = {
            "track_id": f"track_{uuid.uuid4().hex[:8]}",
            "title": "Test Track"
        }
        create_response = api_client.post(f"{BASE_URL}/api/v1/share", json=share_data)
        if create_response.status_code != 200:
            pytest.skip("Could not create share link for testing")
        
        share_code = create_response.json()["share_code"]
        
        # Get social URLs
        response = api_client.get(f"{BASE_URL}/api/v1/share/{share_code}")
        assert response.status_code == 200
        data = response.json()
        assert "twitter" in data or "facebook" in data or "direct_link" in data
        print(f"Social URLs retrieved for share code: {share_code}")

    def test_get_shared_track(self, api_client):
        """Test getting track info from share code."""
        # First create a share link
        share_data = {
            "track_id": f"track_{uuid.uuid4().hex[:8]}",
            "title": "Shared Track Test"
        }
        create_response = api_client.post(f"{BASE_URL}/api/v1/share", json=share_data)
        if create_response.status_code != 200:
            pytest.skip("Could not create share link for testing")
        
        share_code = create_response.json()["share_code"]
        
        # Get track info
        response = api_client.get(f"{BASE_URL}/api/v1/share/{share_code}/track")
        assert response.status_code == 200
        data = response.json()
        assert "track_id" in data
        print(f"Shared track info retrieved: {data['track_id']}")


# ===========================================
# Subscription Plans Tests (Phase 5)
# ===========================================
class TestSubscriptionPlans:
    """Test subscription and payment endpoints."""

    def test_get_plans(self, api_client):
        """Test getting subscription plans (public endpoint)."""
        response = api_client.get(f"{BASE_URL}/api/v1/plans")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # Verify plan structure
        plan = data[0]
        assert "plan_id" in plan
        assert "name" in plan
        assert "price" in plan
        assert "features" in plan
        print(f"Subscription plans: {[p['name'] for p in data]}")

    def test_get_credit_packages(self, api_client):
        """Test getting credit packages (public endpoint)."""
        response = api_client.get(f"{BASE_URL}/api/v1/plans/credits")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) >= 1
        print(f"Credit packages: {list(data.keys())}")

    def test_get_subscription_status(self, api_client, auth_headers):
        """Test getting user subscription status (auth required)."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/subscription",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "plan_id" in data
        assert "plan_name" in data
        assert "features" in data
        print(f"User subscription: {data['plan_name']} (credits: {data.get('credits_remaining', 0)})")

    def test_can_generate(self, api_client, auth_headers):
        """Test checking if user can generate a track."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/can-generate",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "can_generate" in data
        assert "reason" in data
        print(f"Can generate: {data['can_generate']}, reason: {data['reason']}")


# ===========================================
# Analytics Tests (Phase 6)
# ===========================================
class TestAnalytics:
    """Test analytics endpoints."""

    def test_track_event(self, api_client, auth_headers):
        """Test tracking an analytics event."""
        event_data = {
            "event_type": "user_action",
            "event_name": "track_played",
            "properties": {"track_id": "test_track_001", "duration": 180},
            "metadata": {"source": "test"}
        }
        response = api_client.post(
            f"{BASE_URL}/api/v1/analytics/track",
            json=event_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tracked"] == True
        assert "event_id" in data
        print(f"Event tracked: {data['event_id']}")

    def test_track_event_without_auth(self, api_client):
        """Test tracking an event without authentication (should work)."""
        event_data = {
            "event_type": "page_view",
            "event_name": "home_page",
            "properties": {"path": "/"},
            "metadata": {}
        }
        response = api_client.post(
            f"{BASE_URL}/api/v1/analytics/track",
            json=event_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tracked"] == True
        print("Anonymous event tracked successfully")

    def test_get_public_stats(self, api_client):
        """Test getting public statistics (no auth required)."""
        response = api_client.get(f"{BASE_URL}/api/v1/analytics/public/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_tracks" in data
        assert "total_plays" in data
        assert "total_users" in data
        print(f"Public stats: tracks={data['total_tracks']}, plays={data['total_plays']}")

    def test_prometheus_metrics(self, api_client):
        """Test prometheus metrics endpoint (public)."""
        response = api_client.get(f"{BASE_URL}/api/v1/analytics/metrics/prometheus")
        assert response.status_code == 200
        # Prometheus format is plain text
        assert response.headers.get("content-type", "").startswith("text/plain")
        print(f"Prometheus metrics retrieved (length: {len(response.text)})")


# ===========================================
# A/B Testing Tests (Phase 6)
# ===========================================
class TestABTesting:
    """Test A/B testing endpoints."""

    def test_list_experiments_admin(self, api_client, admin_headers):
        """Test listing experiments (requires moderator/admin role)."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/experiments",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        experiments = data["experiments"]
        print(f"Experiments found: {len(experiments)}")
        for exp in experiments[:3]:  # Print first 3
            print(f"  - {exp.get('experiment_id')}: {exp.get('status')}")

    def test_list_experiments_unauthorized(self, api_client, auth_headers):
        """Test listing experiments with regular user (should fail)."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/experiments",
            headers=auth_headers
        )
        # Should be 403 Forbidden for non-moderator users
        assert response.status_code == 403
        print("Regular user correctly denied access to experiments list")

    def test_get_experiment_variant(self, api_client, admin_headers):
        """Test getting variant assignment."""
        # First start the ui_theme_v1 experiment
        api_client.post(
            f"{BASE_URL}/api/v1/analytics/experiments/ui_theme_v1/start",
            headers=admin_headers
        )
        
        # Get variant
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/experiments/ui_theme_v1/variant",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "ui_theme_v1"
        assert "variant" in data
        assert data["variant"] in ["dark", "light"]
        print(f"Variant assigned: {data['variant']}")

    def test_track_conversion(self, api_client, admin_headers):
        """Test tracking a conversion for an experiment."""
        response = api_client.post(
            f"{BASE_URL}/api/v1/analytics/experiments/ui_theme_v1/convert?value=1.0",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["converted"] == True
        print("Conversion tracked successfully")

    def test_get_experiment_results(self, api_client, admin_headers):
        """Test getting experiment results."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/experiments/ui_theme_v1/results",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for result in data:
            print(f"  Variant {result['variant_name']}: {result['participants']} participants, "
                  f"{result['conversion_rate']} conversion rate")


# ===========================================
# Dashboard Tests (Phase 6)
# ===========================================
class TestDashboard:
    """Test dashboard and metrics endpoints."""

    def test_get_dashboard_widgets(self, api_client, admin_headers):
        """Test getting dashboard widgets."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/dashboard/widgets",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"Dashboard widgets: {len(data)}")

    def test_get_system_metrics(self, api_client, admin_headers):
        """Test getting system metrics."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/metrics/system",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data or "memory_percent" in data or "timestamp" in data
        print(f"System metrics retrieved: {list(data.keys())[:5]}...")

    def test_get_business_metrics(self, api_client, admin_headers):
        """Test getting business metrics."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/metrics/business",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_tracks" in data or "total_users" in data
        print(f"Business metrics: {data}")

    def test_get_dashboard_unauthorized(self, api_client, auth_headers):
        """Test dashboard access with regular user (should fail)."""
        response = api_client.get(
            f"{BASE_URL}/api/v1/analytics/dashboard",
            headers=auth_headers
        )
        # Should be 403 Forbidden for non-moderator users
        assert response.status_code == 403
        print("Regular user correctly denied access to dashboard")


# ===========================================
# Root Endpoint Test
# ===========================================
class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self, api_client):
        """Test the root endpoint."""
        response = api_client.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "SonicForge"
        assert "version" in data
        assert "status" in data
        assert "phases" in data
        print(f"SonicForge v{data['version']}: {data['status']}")
        print(f"Phases: {data['phases']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
