-- Application store (Integration plan v0.3, section 8). UTC timestamps: YYYY-MM-DDTHH:MM:SS.ffffffZ.
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    owner_code TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('answer','tutor','quiz')),
    topic_id TEXT,
    topic_text TEXT,
    pending_item_id TEXT,
    item_version TEXT,
    pending_question TEXT,
    pending_kind TEXT CHECK(pending_kind IS NULL OR pending_kind IN ('quiz','tutor')),
    state TEXT NOT NULL CHECK(state IN ('idle','awaiting_response','paused','closed')),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    expires_at TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    kb_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK ((state = 'awaiting_response') = (pending_item_id IS NOT NULL))
);
CREATE INDEX sessions_owner ON sessions(owner_code, course_id);

CREATE TABLE notice_acceptance (
    owner_code TEXT NOT NULL,
    course_id TEXT NOT NULL,
    notice_version TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    PRIMARY KEY (owner_code, course_id, notice_version)
);

CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    owner_code TEXT NOT NULL,
    course_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    body_fingerprint TEXT NOT NULL,
    session_id TEXT,
    expected_revision INTEGER,
    pinned_versions TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('admitted','running','authorized','failed','cancelled')),
    content_type TEXT,
    response_code TEXT,
    fixed_payload TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (owner_code, course_id, idempotency_key)
);

CREATE TABLE delivery_authorizations (
    request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
    payload_digest TEXT NOT NULL,
    response_code TEXT NOT NULL,
    content_type TEXT NOT NULL,
    registry_epoch INTEGER NOT NULL,
    deployment_revision INTEGER NOT NULL,
    pinned_versions TEXT NOT NULL,
    authorized_at TEXT NOT NULL
);

CREATE TABLE learning_events (
    request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    expected_revision INTEGER NOT NULL CHECK(expected_revision >= 0),
    item_id TEXT NOT NULL,
    item_version TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK(correct IN (0, 1)),
    authorized_at TEXT NOT NULL,
    UNIQUE(session_id, expected_revision, item_id, item_version)
);

CREATE TABLE item_review_state (
    owner_code TEXT NOT NULL,
    course_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    item_version TEXT NOT NULL,
    streak INTEGER NOT NULL CHECK(streak BETWEEN 0 AND 3),
    due_at TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK(attempts >= 0),
    correct INTEGER NOT NULL CHECK(correct >= 0),
    PRIMARY KEY (owner_code, course_id, item_id, item_version)
);

CREATE TABLE topic_practice (
    owner_code TEXT NOT NULL,
    course_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK(attempts >= 0),
    correct INTEGER NOT NULL CHECK(correct >= 0 AND correct <= attempts),
    PRIMARY KEY (owner_code, course_id, topic_id)
);
