use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Utc};
use chrono_tz::US::Eastern;

/// Parse a flexible date/time string into a UTC timestamp (Eastern time input).
///
/// Supports formats like:
///   "3/29/2026 at 7pm" / "3/29/2026 at 7:30pm" / "3/29/2026 at 19:00"
///   "march 29 at 7pm" / "march 29 2026 at 7pm"
///   "3/29 at 7pm" (year inferred from current date, rolls to next year if past)
pub fn parse_event_time(input: &str) -> Option<DateTime<Utc>> {
    // Normalize "at" separator and collapse whitespace
    let normalized = input
        .trim()
        .replace(" at ", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    // Expand bare hour+meridiem to include :00 so NaiveDateTime has all fields.
    // e.g. "7pm" -> "7:00pm", "19" -> "19:00"  (but "7:30pm" stays as-is)
    let expanded = expand_bare_hour(&normalized);

    // Build candidates: original casing + title-cased first word (for month names)
    let title = title_case_first(&expanded);
    let lower = expanded.to_lowercase();
    let lower_title = title_case_first(&lower);

    // Dedup while preserving order
    let mut candidates: Vec<&str> = vec![expanded.as_str(), title.as_str(), lower.as_str(), lower_title.as_str()];
    candidates.dedup();

    // All formats require minutes (we've already normalized bare hours above)
    let formats_with_year: &[&str] = &[
        "%m/%d/%Y %I:%M%P",
        "%m/%d/%Y %I:%M%p",
        "%m/%d/%Y %H:%M",
        "%B %d %Y %I:%M%P",
        "%B %d %Y %I:%M%p",
        "%B %d %Y %H:%M",
        "%B %d, %Y %I:%M%P",
        "%B %d, %Y %I:%M%p",
        "%B %d, %Y %H:%M",
    ];

    let formats_without_year: &[&str] = &[
        "%m/%d %I:%M%P",
        "%m/%d %I:%M%p",
        "%m/%d %H:%M",
        "%B %d %I:%M%P",
        "%B %d %I:%M%p",
        "%B %d %H:%M",
    ];

    for s in &candidates {
        for fmt in formats_with_year {
            if let Ok(naive) = NaiveDateTime::parse_from_str(s, fmt) {
                return naive_eastern_to_utc(naive);
            }
        }
    }

    // Try formats without year — inject current and next year
    let now = Utc::now();
    for s in &candidates {
        for fmt in formats_without_year {
            let fmt_with_year = format!("{} %Y", fmt);
            let with_year = format!("{} {}", s, now.year());
            if let Ok(naive) = NaiveDateTime::parse_from_str(&with_year, &fmt_with_year) {
                let dt = naive_eastern_to_utc(naive)?;
                if dt >= now {
                    return Some(dt);
                }
                // Roll to next year
                let with_next = format!("{} {}", s, now.year() + 1);
                if let Ok(naive2) = NaiveDateTime::parse_from_str(&with_next, &fmt_with_year) {
                    return naive_eastern_to_utc(naive2);
                }
                return Some(dt);
            }
        }
    }

    None
}

/// Insert ":00" between a bare hour and its meridiem suffix (or just append ":00" for 24h hours).
/// "7pm" → "7:00pm", "19" → "19:00", "7:30pm" → "7:30pm" (no change)
fn expand_bare_hour(s: &str) -> String {
    // Regex-free: find the last token and expand it if it looks like a bare hour
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if let Some(last) = tokens.last() {
        let expanded = expand_time_token(last);
        if expanded != *last {
            let prefix = &tokens[..tokens.len() - 1];
            let mut out = prefix.join(" ");
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&expanded);
            return out;
        }
    }
    s.to_string()
}

fn expand_time_token(tok: &str) -> String {
    let lower = tok.to_lowercase();
    // Has meridiem suffix (am/pm) but no colon: "7pm", "11am"
    for suffix in &["pm", "am"] {
        if lower.ends_with(suffix) {
            let hour_part = &tok[..tok.len() - 2];
            if !hour_part.is_empty() && !hour_part.contains(':') {
                return format!("{}:00{}", hour_part, suffix);
            }
        }
    }
    // Pure numeric, no meridiem, no colon: "19" or "7" → "19:00" / "7:00"
    if !tok.contains(':') && tok.chars().all(|c| c.is_ascii_digit()) {
        return format!("{}:00", tok);
    }
    tok.to_string()
}

fn title_case_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

fn naive_eastern_to_utc(naive: NaiveDateTime) -> Option<DateTime<Utc>> {
    Eastern
        .from_local_datetime(&naive)
        .single()
        .map(|dt| dt.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_time_token() {
        assert_eq!(expand_time_token("7pm"), "7:00pm");
        assert_eq!(expand_time_token("11AM"), "11:00am");
        assert_eq!(expand_time_token("7:30pm"), "7:30pm");
        assert_eq!(expand_time_token("19"), "19:00");
        assert_eq!(expand_time_token("19:00"), "19:00");
    }

    #[test]
    fn test_slash_date_with_year_pm() {
        let dt = parse_event_time("3/29/2026 at 7pm").unwrap();
        // 7pm ET in late March = EDT (UTC-4), so 23:00 UTC
        assert_eq!(dt.format("%Y-%m-%d %H:%M UTC").to_string(), "2026-03-29 23:00 UTC");
    }

    #[test]
    fn test_slash_date_with_year_24h() {
        let dt = parse_event_time("3/29/2026 at 19:00").unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M UTC").to_string(), "2026-03-29 23:00 UTC");
    }

    #[test]
    fn test_named_month_with_year() {
        let dt = parse_event_time("march 29 2026 at 7pm").unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M UTC").to_string(), "2026-03-29 23:00 UTC");
    }

    #[test]
    fn test_named_month_no_year() {
        let dt = parse_event_time("march 29 at 7pm").unwrap();
        assert_eq!(dt.format("%m-%d %H:%M UTC").to_string(), "03-29 23:00 UTC");
    }

    #[test]
    fn test_with_minutes() {
        let dt = parse_event_time("3/29/2026 at 7:30pm").unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M UTC").to_string(), "2026-03-29 23:30 UTC");
    }

    #[test]
    fn test_slash_no_year() {
        let dt = parse_event_time("3/29 at 7pm").unwrap();
        assert_eq!(dt.format("%m-%d %H:%M UTC").to_string(), "03-29 23:00 UTC");
    }

    #[test]
    fn test_invalid_returns_none() {
        assert!(parse_event_time("not a date").is_none());
    }
}
