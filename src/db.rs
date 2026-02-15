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

        CREATE TABLE IF NOT EXISTS tracked_characters (
            name TEXT PRIMARY KEY COLLATE NOCASE,
            added_by TEXT NOT NULL,
            added_at INTEGER NOT NULL DEFAULT (unixepoch())
        );",
    )?;

    // Seed default system prompt if not present
    conn.execute(
        "INSERT OR IGNORE INTO config (key, value) VALUES ('system_prompt', ?1)",
        params![DEFAULT_SYSTEM_PROMPT],
    )?;

    Ok(())
}

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

pub fn add_tracked_character(conn: &Connection, name: &str, added_by: &str) -> Result<bool> {
    let rows = conn.execute(
        "INSERT OR IGNORE INTO tracked_characters (name, added_by) VALUES (?1, ?2)",
        params![name, added_by],
    )?;
    Ok(rows > 0)
}

pub fn remove_tracked_character(conn: &Connection, name: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM tracked_characters WHERE name = ?1",
        params![name],
    )?;
    Ok(rows > 0)
}

pub fn get_tracked_characters(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT name FROM tracked_characters ORDER BY name")?;
    let names = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>>>()?;
    Ok(names)
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
        // Verify tables exist by querying them
        conn.prepare("SELECT * FROM config").unwrap();
        conn.prepare("SELECT * FROM messages").unwrap();
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

        // Overwrite
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
        // Should be the last 5 messages
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
    fn test_add_tracked_character() {
        let conn = setup();
        assert!(add_tracked_character(&conn, "Pyuul", "user123").unwrap());
        // Duplicate insert returns false
        assert!(!add_tracked_character(&conn, "Pyuul", "user456").unwrap());
        // Case-insensitive duplicate
        assert!(!add_tracked_character(&conn, "pyuul", "user789").unwrap());
    }

    #[test]
    fn test_remove_tracked_character() {
        let conn = setup();
        add_tracked_character(&conn, "Pyuul", "user123").unwrap();
        assert!(remove_tracked_character(&conn, "Pyuul").unwrap());
        // Already removed
        assert!(!remove_tracked_character(&conn, "Pyuul").unwrap());
    }

    #[test]
    fn test_get_tracked_characters() {
        let conn = setup();
        add_tracked_character(&conn, "Zara", "user1").unwrap();
        add_tracked_character(&conn, "Alpha", "user2").unwrap();
        add_tracked_character(&conn, "Miko", "user3").unwrap();

        let chars = get_tracked_characters(&conn).unwrap();
        assert_eq!(chars, vec!["Alpha", "Miko", "Zara"]);
    }
}
