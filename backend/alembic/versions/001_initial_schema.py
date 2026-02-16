"""Initial schema with all tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-02-16 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create tracks table
    op.create_table(
        'tracks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('genre', sa.String(50), nullable=False, index=True),
        sa.Column('subgenre', sa.String(50), nullable=True),
        sa.Column('bpm', sa.Integer(), nullable=False, index=True),
        sa.Column('key', sa.String(10), nullable=False, index=True),
        sa.Column('energy', sa.Integer(), nullable=False),
        sa.Column('mood', sa.String(100), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=False),
        sa.Column('instruments', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('prompt_used', sa.Text(), nullable=False),
        sa.Column('generation_engine', sa.String(50), nullable=False),
        sa.Column('generation_params', postgresql.JSONB(), nullable=True),
        sa.Column('variant_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('concept_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('critic_score', sa.Float(), nullable=False, index=True),
        sa.Column('critic_feedback', sa.Text(), nullable=True),
        sa.Column('spectral_analysis', postgresql.JSONB(), nullable=True),
        sa.Column('has_artifacts', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('approved', sa.Boolean(), nullable=False, server_default='false', index=True),
        sa.Column('s3_key_wav', sa.String(500), nullable=True),
        sa.Column('s3_key_mp3', sa.String(500), nullable=True),
        sa.Column('s3_key_flac', sa.String(500), nullable=True),
        sa.Column('visual_s3_key', sa.String(500), nullable=True),
        sa.Column('play_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_played_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_listeners', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('popularity_score', sa.Float(), nullable=False, server_default='0.0', index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column(
            'search_vector',
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', coalesce(title, '') || ' ' || coalesce(genre, '') || ' ' || coalesce(mood, ''))"),
            nullable=True
        )
    )

    # Create indexes for tracks table
    op.create_index('idx_genre_score', 'tracks', ['genre', 'critic_score'])
    op.create_index('idx_approved_created', 'tracks', ['approved', 'created_at'])
    op.create_index('idx_genre_approved_score', 'tracks', ['genre', 'approved', 'critic_score'])
    op.create_index('idx_bpm_key', 'tracks', ['bpm', 'key'])
    op.create_index('idx_popularity', 'tracks', ['popularity_score', 'play_count'])
    op.create_index('idx_generation_params_gin', 'tracks', ['generation_params'], postgresql_using='gin')
    op.create_index('idx_spectral_analysis_gin', 'tracks', ['spectral_analysis'], postgresql_using='gin')
    op.create_index('idx_track_search_vector', 'tracks', ['search_vector'], postgresql_using='gin')
    op.create_index(
        'idx_approved_tracks_only',
        'tracks',
        ['created_at', 'critic_score'],
        postgresql_where=sa.text('approved = true')
    )

    # Create track_concepts table
    op.create_table(
        'track_concepts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('genre', sa.String(50), nullable=False),
        sa.Column('subgenre', sa.String(50), nullable=True),
        sa.Column('target_bpm', sa.Integer(), nullable=False),
        sa.Column('target_key', sa.String(10), nullable=False),
        sa.Column('target_energy', sa.Integer(), nullable=False),
        sa.Column('mood_description', sa.Text(), nullable=False),
        sa.Column('structure', sa.Text(), nullable=False),
        sa.Column('reference_tracks', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('trend_data', postgresql.JSONB(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('variants_generated', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('best_variant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create play_history table
    op.create_table(
        'play_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('concurrent_viewers', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chat_reactions', postgresql.JSONB(), nullable=True),
        sa.Column('skip_requested', sa.Boolean(), nullable=False, server_default='false')
    )

    # Create stream_sessions table
    op.create_table(
        'stream_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('platform', sa.String(50), nullable=False),
        sa.Column('stream_id', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='offline'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_viewers_peak', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_watch_hours', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('error_log', sa.Text(), nullable=True),
        sa.Column('config', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create stream_health_checks table
    op.create_table(
        'stream_health_checks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('is_healthy', sa.Boolean(), nullable=False),
        sa.Column('ffmpeg_pid', sa.Integer(), nullable=True),
        sa.Column('cpu_usage', sa.Float(), nullable=True),
        sa.Column('memory_usage', sa.Float(), nullable=True),
        sa.Column('bitrate_actual', sa.Float(), nullable=True),
        sa.Column('dropped_frames', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_track_id', postgresql.UUID(as_uuid=True), nullable=True)
    )

    # Create composite index for health checks
    op.create_index('idx_health_session_timestamp', 'stream_health_checks', ['session_id', 'timestamp'])

    # Create schedule_slots table
    op.create_table(
        'schedule_slots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('position', sa.Integer(), nullable=False, index=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('genre', sa.String(50), nullable=False),
        sa.Column('energy', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='queued'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create analytics_snapshots table
    op.create_table(
        'analytics_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('concurrent_viewers', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_views', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chat_messages_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('likes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('new_subscribers', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_genre', sa.String(50), nullable=True),
        sa.Column('current_track_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('platform', sa.String(50), nullable=False, server_default='youtube'),
        sa.Column('extra_data', postgresql.JSONB(), nullable=True)
    )

    # Create genre_performance table
    op.create_table(
        'genre_performance',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('genre', sa.String(50), nullable=False, index=True),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('hour', sa.Integer(), nullable=False),
        sa.Column('avg_viewers', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_retention', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('chat_engagement', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('tracks_played', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_track_score', sa.Float(), nullable=False, server_default='0.0')
    )

    # Create composite index for genre performance queries
    op.create_index('idx_genre_performance_date_hour', 'genre_performance', ['genre', 'date', 'hour'])

    # Create listener_requests table
    op.create_table(
        'listener_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('request_type', sa.String(50), nullable=False),
        sa.Column('value', sa.String(255), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('username', sa.String(255), nullable=True),
        sa.Column('fulfilled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create composite index for unfulfilled requests
    op.create_index('idx_listener_requests_fulfilled', 'listener_requests', ['fulfilled', 'created_at'])


def downgrade():
    op.drop_table('listener_requests')
    op.drop_table('genre_performance')
    op.drop_table('analytics_snapshots')
    op.drop_table('schedule_slots')
    op.drop_table('stream_health_checks')
    op.drop_table('stream_sessions')
    op.drop_table('play_history')
    op.drop_table('track_concepts')
    op.drop_table('tracks')
