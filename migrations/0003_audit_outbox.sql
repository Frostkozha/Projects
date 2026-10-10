-- Audit and outbox (section 8, 9). Metadata only: no prompt, evidence, output or identity text.
CREATE TABLE audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    event TEXT NOT NULL,
    request_id TEXT,
    owner_ref TEXT,
    route TEXT,
    response_code TEXT,
    reason TEXT,
    component_status TEXT,
    versions TEXT,
    evidence_ids TEXT,
    stage_ms TEXT,
    authorization_digest TEXT
);
CREATE TABLE welfare_outbox (
    event_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL CHECK(category IN ('emergency','self_harm')),
    support_ref TEXT NOT NULL,
    event_time TEXT NOT NULL,
    delivery_status TEXT NOT NULL CHECK(delivery_status IN ('pending','acknowledged','failed')),
    reviewed_by TEXT,
    reviewed_at TEXT
);
CREATE TABLE incident_events (
    incident_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    category TEXT NOT NULL,
    near_miss INTEGER NOT NULL CHECK(near_miss IN (0,1)),
    event_time TEXT NOT NULL,
    severity TEXT
);
CREATE TABLE response_reports (
    report_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    owner_code TEXT NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('answer_error','source_issue','other')),
    message TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (request_id, owner_code)
);
