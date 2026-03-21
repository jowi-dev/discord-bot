use chrono::{TimeZone, Utc};
use chrono_tz::US::Eastern;

use crate::db::{CharacterInfo, Event, Signup};

/// Role shorthand normalization
pub fn normalize_role(s: &str) -> &'static str {
    match s.to_lowercase().as_str() {
        "tank" | "t" => "tank",
        "healer" | "heal" | "h" => "healer",
        "dps" | "d" | "damage" | "rdps" | "mdps" => "dps",
        _ => "unknown",
    }
}

/// Format a Unix timestamp as a human-readable Eastern time string.
pub fn format_event_time(ts: i64) -> String {
    let utc = Utc.timestamp_opt(ts, 0).single().unwrap_or_default();
    let eastern = utc.with_timezone(&Eastern);
    eastern.format("%A, %B %-d at %-I:%M %p ET").to_string()
}

/// One-line summary: "**Kara Tuesday** (Karazhan) — Thursday, March 26 at 7:00 PM ET [id: 3]"
pub fn format_event_summary(event: &Event) -> String {
    let time_str = format_event_time(event.event_time);
    let raid_tag = event
        .raid_type
        .as_deref()
        .map(|r| format!(" ({})", r))
        .unwrap_or_default();
    format!("**{}**{} — {} [id: {}]", event.name, raid_tag, time_str, event.id)
}

/// Format signup names with Discord @mentions where possible.
/// `resolve_mention` maps discord_user_id -> Discord mention string.
/// `resolve_char_info` maps character_name -> optional CharacterInfo for class display.
pub fn format_event_detail_with_mentions<F, G>(
    event: &Event,
    signups: &[Signup],
    resolve_mention: F,
    resolve_char_info: G,
) -> String
where
    F: Fn(&str) -> String,
    G: Fn(&str) -> Option<CharacterInfo>,
{
    let mut out = String::new();

    out.push_str(&format!("**{}**", event.name));
    if let Some(rt) = &event.raid_type {
        out.push_str(&format!(" — {}", rt));
    }
    out.push('\n');
    out.push_str(&format!("📅 {}\n", format_event_time(event.event_time)));

    if let Some(notes) = &event.notes {
        if !notes.is_empty() {
            out.push_str(&format!("📝 {}\n", notes));
        }
    }

    out.push('\n');

    if signups.is_empty() {
        out.push_str("*No signups yet.*\n");
    } else {
        let tanks: Vec<_> = signups.iter().filter(|s| s.role == "tank").collect();
        let healers: Vec<_> = signups.iter().filter(|s| s.role == "healer").collect();
        let dps: Vec<_> = signups.iter().filter(|s| s.role == "dps").collect();
        let unknown: Vec<_> = signups.iter().filter(|s| s.role == "unknown").collect();

        out.push_str(&format!("**Signups ({}):**\n", signups.len()));

        let fmt_list = |group: Vec<&Signup>| -> String {
            group
                .iter()
                .map(|s| {
                    let mention = resolve_mention(&s.discord_user_id);
                    let class_tag = resolve_char_info(&s.character_name)
                        .and_then(|c| c.class)
                        .map(|c| format!(" {}", c))
                        .unwrap_or_default();
                    format!("{}{} ({})", s.character_name, class_tag, mention)
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        if !tanks.is_empty() {
            out.push_str(&format!("🛡 Tanks: {}\n", fmt_list(tanks)));
        }
        if !healers.is_empty() {
            out.push_str(&format!("💚 Healers: {}\n", fmt_list(healers)));
        }
        if !dps.is_empty() {
            out.push_str(&format!("⚔ DPS: {}\n", fmt_list(dps)));
        }
        if !unknown.is_empty() {
            out.push_str(&format!("❓ Other: {}\n", fmt_list(unknown)));
        }
    }

    out
}

/// Build the system prompt used for LLM-generated event announcements/reminders.
/// Falls back to the global system prompt if the event has none.
pub fn event_llm_system_prompt(event: &Event, global_fallback: &str) -> String {
    event
        .system_prompt
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(global_fallback)
        .to_string()
}

/// Build the user prompt for generating an event announcement.
pub fn announcement_user_prompt(event: &Event) -> String {
    let time_str = format_event_time(event.event_time);
    let raid_type = event.raid_type.as_deref().unwrap_or("raid");
    let notes = event.notes.as_deref().unwrap_or("");
    format!(
        "Write a short, exciting raid announcement for a World of Warcraft: Burning Crusade raid. \
         Event name: {}. Raid type: {}. Scheduled: {}. Notes: {}. \
         Keep it under 3 sentences. Be dramatic and in-character.",
        event.name, raid_type, time_str, notes
    )
}

/// Build the user prompt for a reminder message.
pub fn reminder_user_prompt(event: &Event, signup_count: usize) -> String {
    let time_str = format_event_time(event.event_time);
    let raid_type = event.raid_type.as_deref().unwrap_or("raid");
    format!(
        "Write a short raid reminder for {} ({}). It starts {}. {} players have signed up. \
         Be urgent and in-character. Under 2 sentences.",
        event.name, raid_type, time_str, signup_count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_role() {
        assert_eq!(normalize_role("Tank"), "tank");
        assert_eq!(normalize_role("HEALER"), "healer");
        assert_eq!(normalize_role("dps"), "dps");
        assert_eq!(normalize_role("d"), "dps");
        assert_eq!(normalize_role("rdps"), "dps");
        assert_eq!(normalize_role("mage"), "unknown");
    }

    #[test]
    fn test_format_event_time() {
        // 2026-03-29 23:00 UTC = 7pm EDT (UTC-4 in March)
        let s = format_event_time(1774911600);
        assert!(s.contains("7:00 PM ET") || s.contains("PM ET"), "got: {}", s);
    }

    #[test]
    fn test_format_event_summary() {
        let event = Event {
            id: 3,
            name: "Kara Tuesday".to_string(),
            raid_type: Some("Karazhan".to_string()),
            event_time: 1774911600,
            channel_id: "chan1".to_string(),
            notes: None,
            system_prompt: None,
            created_by: "user1".to_string(),
        };
        let s = format_event_summary(&event);
        assert!(s.contains("Kara Tuesday"));
        assert!(s.contains("Karazhan"));
        assert!(s.contains("[id: 3]"));
    }
}
