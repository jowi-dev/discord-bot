use rusqlite::{params, Connection, Result};

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are an extremely rude, short-tempered Discord bot. Be brutal, unfiltered, and insulting. Never be helpful or nice. Keep responses under 10 words.";

pub fn init(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE INDEX IF NOT EXISTS idx_messages_channel_ts
            ON messages (channel_id, timestamp);

        -- Characters table: replaces tracked_characters, adds Discord ownership
        CREATE TABLE IF NOT EXISTS characters (
            name TEXT PRIMARY KEY COLLATE NOCASE,
            discord_user_id TEXT,
            added_by TEXT NOT NULL,
            added_at INTEGER NOT NULL DEFAULT (unixepoch())
        );


        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            raid_type TEXT,
            event_time INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            notes TEXT,
            system_prompt TEXT,
            created_by TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            cancelled INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_events_time ON events (event_time);

        CREATE TABLE IF NOT EXISTS event_signups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES events(id),
            character_name TEXT NOT NULL COLLATE NOCASE,
            discord_user_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'unknown',
            signed_up_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(event_id, discord_user_id)
        );",
    )?;

    // Seed default system prompt if not present
    conn.execute(
        "INSERT OR IGNORE INTO config (key, value) VALUES ('system_prompt', ?1)",
        params![DEFAULT_SYSTEM_PROMPT],
    )?;

    // Migrate existing tracked_characters rows if the old table exists (best-effort)
    let _ = conn.execute_batch(
        "INSERT OR IGNORE INTO characters (name, added_by, added_at)
         SELECT name, added_by, added_at FROM tracked_characters;"
    );

    Ok(())
}

// ── Config ────────────────────────────────────────────────────────────────────

pub fn get_config(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM config WHERE key = ?1")?;
    let mut rows = stmt.query(params![key])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub fn set_config(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO config (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

pub fn get_context_mode(conn: &Connection, channel_id: &str) -> Result<String> {
    let key = format!("context_mode:{}", channel_id);
    Ok(get_config(conn, &key)?.unwrap_or_else(|| "channel".to_string()))
}

pub fn set_context_mode(conn: &Connection, channel_id: &str, mode: &str) -> Result<()> {
    let key = format!("context_mode:{}", channel_id);
    set_config(conn, &key, mode)
}

// ── Messages ─────────────────────────────────────────────────────────────────

pub fn clear_messages(conn: &Connection, channel_id: &str) -> Result<usize> {
    conn.execute(
        "DELETE FROM messages WHERE channel_id = ?1",
        params![channel_id],
    )
}

pub fn store_message(conn: &Connection, channel_id: &str, role: &str, content: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO messages (channel_id, role, content) VALUES (?1, ?2, ?3)",
        params![channel_id, role, content],
    )?;
    Ok(())
}

pub struct StoredMessage {
    pub role: String,
    pub content: String,
}

pub fn get_recent_messages(
    conn: &Connection,
    channel_id: &str,
    limit: usize,
) -> Result<Vec<StoredMessage>> {
    let mut stmt = conn.prepare(
        "SELECT role, content FROM messages
         WHERE channel_id = ?1
         ORDER BY timestamp DESC, id DESC
         LIMIT ?2",
    )?;
    let mut messages: Vec<StoredMessage> = stmt
        .query_map(params![channel_id, limit as i64], |row| {
            Ok(StoredMessage {
                role: row.get(0)?,
                content: row.get(1)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;

    // Reverse so oldest is first (we fetched newest-first for LIMIT)
    messages.reverse();
    Ok(messages)
}

// ── Characters ────────────────────────────────────────────────────────────────

pub fn add_character(conn: &Connection, name: &str, added_by: &str) -> Result<bool> {
    let rows = conn.execute(
        "INSERT OR IGNORE INTO characters (name, added_by) VALUES (?1, ?2)",
        params![name, added_by],
    )?;
    Ok(rows > 0)
}

pub fn remove_character(conn: &Connection, name: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM characters WHERE name = ?1",
        params![name],
    )?;
    Ok(rows > 0)
}

/// Claim a character: associate it with a Discord user.
/// Inserts the character if it doesn't exist yet, then sets discord_user_id.
/// Returns Err if the character is already claimed by a different user.
pub fn claim_character(
    conn: &Connection,
    name: &str,
    discord_user_id: &str,
) -> Result<ClaimResult> {
    // Upsert the character row
    conn.execute(
        "INSERT OR IGNORE INTO characters (name, added_by) VALUES (?1, ?2)",
        params![name, discord_user_id],
    )?;

    // Check current claim state
    let existing: Option<String> = conn
        .query_row(
            "SELECT discord_user_id FROM characters WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    match existing {
        Some(ref uid) if uid == discord_user_id => return Ok(ClaimResult::AlreadyYours),
        Some(_) => return Ok(ClaimResult::TakenByOther),
        None => {}
    }

    conn.execute(
        "UPDATE characters SET discord_user_id = ?1 WHERE name = ?2",
        params![discord_user_id, name],
    )?;
    Ok(ClaimResult::Claimed)
}

pub enum ClaimResult {
    Claimed,
    AlreadyYours,
    TakenByOther,
}

pub fn unclaim_character(conn: &Connection, name: &str, discord_user_id: &str) -> Result<bool> {
    let rows = conn.execute(
        "UPDATE characters SET discord_user_id = NULL
         WHERE name = ?1 AND discord_user_id = ?2",
        params![name, discord_user_id],
    )?;
    Ok(rows > 0)
}

pub fn get_user_characters(conn: &Connection, discord_user_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT name FROM characters WHERE discord_user_id = ?1 ORDER BY name",
    )?;
    let names = stmt
        .query_map(params![discord_user_id], |row| row.get(0))?
        .collect::<Result<Vec<String>>>()?;
    Ok(names)
}

/// Look up the Discord user ID for a character name (case-insensitive).
#[allow(dead_code)]
pub fn get_character_owner(conn: &Connection, name: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT discord_user_id FROM characters WHERE name = ?1",
        params![name],
        |row| row.get(0),
    )
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    })
}

// kept for backward-compat with existing callers in main.rs
pub fn add_tracked_character(conn: &Connection, name: &str, added_by: &str) -> Result<bool> {
    add_character(conn, name, added_by)
}

pub fn remove_tracked_character(conn: &Connection, name: &str) -> Result<bool> {
    remove_character(conn, name)
}

pub fn get_tracked_characters(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT name FROM characters ORDER BY name")?;
    let names = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>>>()?;
    Ok(names)
}

// ── Events ────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct Event {
    pub id: i64,
    pub name: String,
    pub raid_type: Option<String>,
    pub event_time: i64,
    pub channel_id: String,
    pub notes: Option<String>,
    pub system_prompt: Option<String>,
    pub created_by: String,
}

pub fn create_event(
    conn: &Connection,
    name: &str,
    raid_type: Option<&str>,
    event_time: i64,
    channel_id: &str,
    notes: Option<&str>,
    system_prompt: Option<&str>,
    created_by: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO events (name, raid_type, event_time, channel_id, notes, system_prompt, created_by)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![name, raid_type, event_time, channel_id, notes, system_prompt, created_by],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_event(conn: &Connection, id: i64) -> Result<Option<Event>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, raid_type, event_time, channel_id, notes, system_prompt, created_by
         FROM events WHERE id = ?1 AND cancelled = 0",
    )?;
    let mut rows = stmt.query(params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(row_to_event(row)?)),
        None => Ok(None),
    }
}

pub fn get_upcoming_events(conn: &Connection, from_ts: i64) -> Result<Vec<Event>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, raid_type, event_time, channel_id, notes, system_prompt, created_by
         FROM events
         WHERE cancelled = 0 AND event_time >= ?1
         ORDER BY event_time ASC",
    )?;
    let events = stmt
        .query_map(params![from_ts], |row| row_to_event(row))?
        .collect::<Result<Vec<_>>>()?;
    Ok(events)
}

/// Events starting between `from_ts` and `to_ts` (exclusive).
pub fn get_events_in_window(conn: &Connection, from_ts: i64, to_ts: i64) -> Result<Vec<Event>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, raid_type, event_time, channel_id, notes, system_prompt, created_by
         FROM events
         WHERE cancelled = 0 AND event_time >= ?1 AND event_time < ?2
         ORDER BY event_time ASC",
    )?;
    let events = stmt
        .query_map(params![from_ts, to_ts], |row| row_to_event(row))?
        .collect::<Result<Vec<_>>>()?;
    Ok(events)
}

pub fn cancel_event(conn: &Connection, id: i64) -> Result<bool> {
    let rows = conn.execute(
        "UPDATE events SET cancelled = 1 WHERE id = ?1 AND cancelled = 0",
        params![id],
    )?;
    Ok(rows > 0)
}

pub fn set_event_system_prompt(conn: &Connection, id: i64, prompt: &str) -> Result<bool> {
    let rows = conn.execute(
        "UPDATE events SET system_prompt = ?1 WHERE id = ?2 AND cancelled = 0",
        params![prompt, id],
    )?;
    Ok(rows > 0)
}

fn row_to_event(row: &rusqlite::Row) -> rusqlite::Result<Event> {
    Ok(Event {
        id: row.get(0)?,
        name: row.get(1)?,
        raid_type: row.get(2)?,
        event_time: row.get(3)?,
        channel_id: row.get(4)?,
        notes: row.get(5)?,
        system_prompt: row.get(6)?,
        created_by: row.get(7)?,
    })
}

// ── Event signups ─────────────────────────────────────────────────────────────

pub struct Signup {
    pub character_name: String,
    pub discord_user_id: String,
    pub role: String,
}

pub enum SignupResult {
    Added,
    Updated,
    EventNotFound,
}

pub fn signup_for_event(
    conn: &Connection,
    event_id: i64,
    character_name: &str,
    discord_user_id: &str,
    role: &str,
) -> Result<SignupResult> {
    // Verify event exists and is not cancelled
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM events WHERE id = ?1 AND cancelled = 0",
            params![event_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !exists {
        return Ok(SignupResult::EventNotFound);
    }

    let existing: bool = conn
        .query_row(
            "SELECT 1 FROM event_signups WHERE event_id = ?1 AND discord_user_id = ?2",
            params![event_id, discord_user_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if existing {
        conn.execute(
            "UPDATE event_signups SET character_name = ?1, role = ?2
             WHERE event_id = ?3 AND discord_user_id = ?4",
            params![character_name, role, event_id, discord_user_id],
        )?;
        Ok(SignupResult::Updated)
    } else {
        conn.execute(
            "INSERT INTO event_signups (event_id, character_name, discord_user_id, role)
             VALUES (?1, ?2, ?3, ?4)",
            params![event_id, character_name, discord_user_id, role],
        )?;
        Ok(SignupResult::Added)
    }
}

pub fn remove_signup(conn: &Connection, event_id: i64, discord_user_id: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM event_signups WHERE event_id = ?1 AND discord_user_id = ?2",
        params![event_id, discord_user_id],
    )?;
    Ok(rows > 0)
}

pub fn get_signups(conn: &Connection, event_id: i64) -> Result<Vec<Signup>> {
    let mut stmt = conn.prepare(
        "SELECT character_name, discord_user_id, role
         FROM event_signups WHERE event_id = ?1
         ORDER BY signed_up_at ASC",
    )?;
    let signups = stmt
        .query_map(params![event_id], |row| {
            Ok(Signup {
                character_name: row.get(0)?,
                discord_user_id: row.get(1)?,
                role: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;
    Ok(signups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init(&conn).unwrap();
        conn
    }

    #[test]
    fn test_init_creates_schema() {
        let conn = setup();
        conn.prepare("SELECT * FROM config").unwrap();
        conn.prepare("SELECT * FROM messages").unwrap();
        conn.prepare("SELECT * FROM characters").unwrap();
        conn.prepare("SELECT * FROM events").unwrap();
        conn.prepare("SELECT * FROM event_signups").unwrap();
    }

    #[test]
    fn test_default_system_prompt() {
        let conn = setup();
        let prompt = get_config(&conn, "system_prompt").unwrap().unwrap();
        assert_eq!(prompt, DEFAULT_SYSTEM_PROMPT);
    }

    #[test]
    fn test_set_and_get_config() {
        let conn = setup();
        set_config(&conn, "test_key", "test_value").unwrap();
        assert_eq!(
            get_config(&conn, "test_key").unwrap(),
            Some("test_value".to_string())
        );

        set_config(&conn, "test_key", "new_value").unwrap();
        assert_eq!(
            get_config(&conn, "test_key").unwrap(),
            Some("new_value".to_string())
        );
    }

    #[test]
    fn test_store_and_retrieve_messages() {
        let conn = setup();
        store_message(&conn, "chan1", "user", "hello").unwrap();
        store_message(&conn, "chan1", "assistant", "hi there").unwrap();

        let msgs = get_recent_messages(&conn, "chan1", 10).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content, "hi there");
    }

    #[test]
    fn test_message_history_limit() {
        let conn = setup();
        for i in 0..20 {
            store_message(&conn, "chan1", "user", &format!("msg {}", i)).unwrap();
        }

        let msgs = get_recent_messages(&conn, "chan1", 5).unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0].content, "msg 15");
        assert_eq!(msgs[4].content, "msg 19");
    }

    #[test]
    fn test_messages_scoped_to_channel() {
        let conn = setup();
        store_message(&conn, "chan_a", "user", "message in A").unwrap();
        store_message(&conn, "chan_b", "user", "message in B").unwrap();

        let msgs_a = get_recent_messages(&conn, "chan_a", 10).unwrap();
        assert_eq!(msgs_a.len(), 1);
        assert_eq!(msgs_a[0].content, "message in A");

        let msgs_b = get_recent_messages(&conn, "chan_b", 10).unwrap();
        assert_eq!(msgs_b.len(), 1);
        assert_eq!(msgs_b[0].content, "message in B");
    }

    #[test]
    fn test_add_character() {
        let conn = setup();
        assert!(add_character(&conn, "Pyuul", "user123").unwrap());
        assert!(!add_character(&conn, "Pyuul", "user456").unwrap());
        assert!(!add_character(&conn, "pyuul", "user789").unwrap());
    }

    #[test]
    fn test_claim_character() {
        let conn = setup();
        add_character(&conn, "Pyuul", "user123").unwrap();
        // user123 claims it
        conn.execute(
            "UPDATE characters SET discord_user_id = ?1 WHERE name = ?2",
            params!["user123", "Pyuul"],
        ).unwrap();
        let owner = get_character_owner(&conn, "Pyuul").unwrap();
        assert_eq!(owner, Some("user123".to_string()));
    }

    #[test]
    fn test_get_user_characters() {
        let conn = setup();
        add_character(&conn, "Pyuul", "user123").unwrap();
        add_character(&conn, "Zara", "user123").unwrap();
        conn.execute("UPDATE characters SET discord_user_id = 'user123' WHERE name IN ('Pyuul', 'Zara')", []).unwrap();
        let chars = get_user_characters(&conn, "user123").unwrap();
        assert_eq!(chars, vec!["Pyuul", "Zara"]);
    }

    #[test]
    fn test_create_and_get_event() {
        let conn = setup();
        let id = create_event(&conn, "Kara Tuesday", Some("Karazhan"), 1000, "chan1", None, None, "user1").unwrap();
        let event = get_event(&conn, id).unwrap().unwrap();
        assert_eq!(event.name, "Kara Tuesday");
        assert_eq!(event.raid_type, Some("Karazhan".to_string()));
        assert_eq!(event.event_time, 1000);
    }

    #[test]
    fn test_cancel_event() {
        let conn = setup();
        let id = create_event(&conn, "Kara", None, 1000, "chan1", None, None, "user1").unwrap();
        assert!(cancel_event(&conn, id).unwrap());
        assert!(get_event(&conn, id).unwrap().is_none());
    }

    #[test]
    fn test_signup_and_remove() {
        let conn = setup();
        let event_id = create_event(&conn, "Kara", None, 1000, "chan1", None, None, "user1").unwrap();

        match signup_for_event(&conn, event_id, "Pyuul", "user1", "tank").unwrap() {
            SignupResult::Added => {}
            _ => panic!("expected Added"),
        }

        let signups = get_signups(&conn, event_id).unwrap();
        assert_eq!(signups.len(), 1);
        assert_eq!(signups[0].character_name, "Pyuul");
        assert_eq!(signups[0].role, "tank");

        // Second signup by same user updates
        match signup_for_event(&conn, event_id, "Zara", "user1", "healer").unwrap() {
            SignupResult::Updated => {}
            _ => panic!("expected Updated"),
        }
        let signups = get_signups(&conn, event_id).unwrap();
        assert_eq!(signups.len(), 1);
        assert_eq!(signups[0].character_name, "Zara");

        assert!(remove_signup(&conn, event_id, "user1").unwrap());
        assert_eq!(get_signups(&conn, event_id).unwrap().len(), 0);
    }

    #[test]
    fn test_get_upcoming_events() {
        let conn = setup();
        create_event(&conn, "Past", None, 100, "chan1", None, None, "u1").unwrap();
        create_event(&conn, "Future", None, 9999999999, "chan1", None, None, "u1").unwrap();
        let events = get_upcoming_events(&conn, 200).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "Future");
    }
}
