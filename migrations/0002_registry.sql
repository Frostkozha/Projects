-- Live registry (section 4, 8): current rights/status/epoch outside the immutable index.
CREATE TABLE registry_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    registry_version TEXT NOT NULL,
    epoch INTEGER NOT NULL CHECK(epoch >= 0),
    updated_at TEXT NOT NULL
);
CREATE TABLE source_live_state (
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('live','archived','blocked','pending_review','under_review','revoked')),
    rights_expires_at TEXT,
    review_due_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, source_version)
);
CREATE TABLE topic_blocks (
    course_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    blocked_at TEXT NOT NULL,
    PRIMARY KEY (course_id, topic_id)
);
CREATE TABLE deployment_state (
    course_id TEXT PRIMARY KEY,
    state TEXT NOT NULL CHECK(state IN ('active','suspended','exam_shutdown','disallowed')),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);
