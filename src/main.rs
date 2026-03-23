mod datetime_parse;
mod db;
mod events;

use chrono::Utc;
use futures::future::join_all;
use reqwest::Client as HttpClient;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use serenity::builder::{CreateAttachment, CreateMessage};
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::model::id::ChannelId;
use serenity::prelude::*;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

// Download from CivitAI: https://civitai.com/models/257749
const COMFY_MODEL_NAME: &str = "juggernautXL_ragnarokBy.safetensors";

// No content restrictions — this is intentional.
const COMFY_NEGATIVE_PROMPT: &str =
    "worst quality, bad quality, low quality, lowres, blurry, jpeg artifacts, \
     bad anatomy, bad hands, extra fingers, missing fingers, deformed, mutation, extra limbs, \
     watermark, signature, text, logo, ugly, poorly drawn face, out of frame";

const HISTORY_LIMIT: usize = 10;

struct BattleNetAuth {
    client_id: String,
    client_secret: String,
    token: Option<String>,
    expires_at: Option<Instant>,
}

impl BattleNetAuth {
    fn new(client_id: String, client_secret: String) -> Self {
        Self {
            client_id,
            client_secret,
            token: None,
            expires_at: None,
        }
    }

    fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => Instant::now() >= exp,
            None => true,
        }
    }
}

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: u64,
}

#[derive(Deserialize)]
struct WowCharacter {
    name: String,
    level: u32,
    race: WowEnum,
    character_class: WowEnum,
}

#[derive(Deserialize)]
struct WowEnum {
    name: String,
}

struct Handler {
    http_client: HttpClient,
    llama_api_url: Option<String>,
    llama_hosts: Vec<(String, String)>, // (name, url) pairs for health checks
    comfy_api_url: Option<String>,
    comfy_hosts: Vec<(String, String)>, // (name, url) pairs for health checks
    battlenet_auth: Option<Arc<Mutex<BattleNetAuth>>>,
    db: Arc<Mutex<Connection>>,
}

#[derive(Deserialize)]
struct LlamaHealthResponse {
    status: String,
}

#[derive(Deserialize)]
struct ComfyPromptResponse {
    prompt_id: String,
}

#[derive(Serialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    temperature: f32,
    stop: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

impl Handler {
    async fn get_battlenet_token(&self) -> Result<String, String> {
        let auth_lock = self
            .battlenet_auth
            .as_ref()
            .ok_or("Battle.net not configured")?;
        let mut auth = auth_lock.lock().await;

        if !auth.is_expired() {
            return Ok(auth.token.clone().unwrap());
        }

        let resp = self
            .http_client
            .post("https://oauth.battle.net/token")
            .basic_auth(&auth.client_id, Some(&auth.client_secret))
            .form(&[("grant_type", "client_credentials")])
            .send()
            .await
            .map_err(|e| format!("OAuth request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("OAuth returned status {}", resp.status()));
        }

        let token_resp: OAuthTokenResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse OAuth response: {}", e))?;

        // Expire 60s early to avoid edge cases
        let expires_at = Instant::now()
            + std::time::Duration::from_secs(token_resp.expires_in.saturating_sub(60));
        auth.token = Some(token_resp.access_token.clone());
        auth.expires_at = Some(expires_at);

        Ok(token_resp.access_token)
    }

    async fn fetch_wow_character(&self, name: &str) -> Result<WowCharacter, String> {
        let token = self.get_battlenet_token().await?;
        let url = format!(
            "https://us.api.blizzard.com/profile/wow/character/nightslayer/{}?namespace=profile-classicann-us&locale=en_US",
            name.to_lowercase()
        );

        let resp = self
            .http_client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| format!("API request failed: {}", e))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(format!("Character **{}** not found on Nightslayer.", name));
        }

        if !resp.status().is_success() {
            return Err(format!("Blizzard API returned status {}", resp.status()));
        }

        resp.json::<WowCharacter>()
            .await
            .map_err(|e| format!("Failed to parse character data: {}", e))
    }

    async fn check_llama_health(&self, url: &str) -> String {
        let result = self
            .http_client
            .get(format!("{}/health", url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await;

        match result {
            Err(_) => "❌ unreachable".to_string(),
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<LlamaHealthResponse>().await {
                    Ok(h) if h.status == "ok" => "✅ ok".to_string(),
                    Ok(h) => format!("⚠ {}", h.status),
                    Err(_) => "✅ reachable".to_string(),
                }
            }
            Ok(resp) if resp.status() == 503 => {
                match resp.json::<LlamaHealthResponse>().await {
                    Ok(h) => format!("⏳ {}", h.status),
                    Err(_) => "⏳ unavailable (503)".to_string(),
                }
            }
            Ok(resp) => format!("❌ HTTP {}", resp.status()),
        }
    }

    async fn check_comfy_health(&self, url: &str) -> String {
        let result = self
            .http_client
            .get(format!("{}/system_stats", url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await;

        match result {
            Err(_) => "❌ unreachable".to_string(),
            Ok(resp) if resp.status().is_success() => "✅ ok".to_string(),
            Ok(resp) => format!("❌ HTTP {}", resp.status()),
        }
    }

    async fn health_check(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        // LLM (text gen)
        lines.push("**LLM (text gen)**".to_string());
        if self.llama_hosts.is_empty() && self.llama_api_url.is_none() {
            lines.push("  not configured".to_string());
        } else {
            let host_futs: Vec<_> = self
                .llama_hosts
                .iter()
                .map(|(name, url)| async move { (name.as_str(), self.check_llama_health(url).await) })
                .collect();
            let host_results = join_all(host_futs).await;
            for (name, status) in host_results {
                lines.push(format!("  {}: {}", name, status));
            }
            if let Some(router_url) = &self.llama_api_url {
                let status = self.check_llama_health(router_url).await;
                lines.push(format!("  router: {}", status));
            }
        }

        lines.push(String::new());

        // Image gen (ComfyUI)
        lines.push("**Image gen (ComfyUI)**".to_string());
        if self.comfy_hosts.is_empty() && self.comfy_api_url.is_none() {
            lines.push("  not configured".to_string());
        } else {
            let host_futs: Vec<_> = self
                .comfy_hosts
                .iter()
                .map(|(name, url)| async move { (name.as_str(), self.check_comfy_health(url).await) })
                .collect();
            let host_results = join_all(host_futs).await;
            for (name, status) in host_results {
                lines.push(format!("  {}: {}", name, status));
            }
            if let Some(comfy_url) = &self.comfy_api_url {
                // Only show fallback router entry if no named hosts cover it
                if self.comfy_hosts.is_empty() {
                    let status = self.check_comfy_health(comfy_url).await;
                    lines.push(format!("  comfyui: {}", status));
                }
            }
        }

        lines.join("\n")
    }

    async fn ask_llama(&self, context_key: &str, user_message: &str) -> Result<String, String> {
        let api_url = self
            .llama_api_url
            .as_ref()
            .ok_or("LLAMA_API_URL not configured")?;

        // Build messages array with system prompt and history
        let messages = {
            let conn = self.db.lock().await;

            // Store the user message
            db::store_message(&conn, context_key, "user", user_message)
                .map_err(|e| format!("DB error storing user message: {}", e))?;

            let system_prompt = db::get_config(&conn, "system_prompt")
                .map_err(|e| format!("DB error: {}", e))?
                .unwrap_or_default();

            let history = db::get_recent_messages(&conn, context_key, HISTORY_LIMIT)
                .map_err(|e| format!("DB error: {}", e))?;

            let mut msgs = Vec::with_capacity(history.len() + 1);

            if !system_prompt.is_empty() {
                msgs.push(ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                });
            }

            for m in history {
                msgs.push(ChatMessage {
                    role: m.role,
                    content: m.content,
                });
            }

            // Append a reminder suffix to the last user message
            if let Some(last) = msgs.last_mut() {
                if last.role == "user" {
                    let cap = db::get_config(&conn, "response_cap")
                        .ok()
                        .flatten()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(10);
                    last.content.push_str(&format!(
                        "\n(Reply in {} words or less. Stay in character.)",
                        cap
                    ));
                }
            }

            msgs
        };

        let request = ChatRequest {
            messages,
            temperature: 0.4,
            stop: vec![
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
                "</s>".to_string(),
                "[INST]".to_string(),
            ],
        };

        let response = self
            .http_client
            .post(format!("{}/v1/chat/completions", api_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to reach llama.cpp: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("llama.cpp returned status {}", response.status()));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let reply = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from model".to_string())?;

        // Store the assistant response
        {
            let conn = self.db.lock().await;
            if let Err(e) = db::store_message(&conn, context_key, "assistant", &reply) {
                error!("Failed to store assistant message: {}", e);
            }
        }

        Ok(reply)
    }

    async fn query_llm_oneshot(
        &self,
        system_prompt: String,
        user_message: String,
    ) -> Result<String, String> {
        let api_url = self
            .llama_api_url
            .as_ref()
            .ok_or("LLAMA_API_URL not configured")?;

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_message,
            },
        ];

        let request = ChatRequest {
            messages,
            temperature: 0.4,
            stop: vec![
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
                "</s>".to_string(),
                "[INST]".to_string(),
            ],
        };

        let response = self
            .http_client
            .post(format!("{}/v1/chat/completions", api_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to reach llama.cpp: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("llama.cpp returned status {}", response.status()));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from model".to_string())
    }

    // Expands a natural-language image request into a detailed prompt for epiCRealism.
    // Falls back to the original prompt if the LLM is unavailable.
    async fn expand_image_prompt(&self, user_prompt: &str) -> String {
        let server_context = {
            let conn = self.db.lock().await;
            db::get_config(&conn, "image_prompt")
                .ok()
                .flatten()
                .unwrap_or_default()
        };

        let system = format!(
            "You are a prompt engineer for Pony Diffusion XL, a Stable Diffusion XL model. \
             Convert the user's image request into a list of comma-separated descriptive tags. \
             Do NOT write sentences or prose. Do NOT use words like 'a', 'the', 'with', 'its', 'and', 'is', 'are'. \
             Include tags for: subject, physical details, clothing, pose, lighting, setting, camera angle, art style. \
             Reply with ONLY the comma-separated tags, nothing else.\n\
             \n\
             Example:\n\
             Input: a knight fighting a dragon at sunset\n\
             Output: armored knight, sword raised, dragon, scales, fire breath, golden sunset, rocky cliff, epic battle, low angle shot, cinematic lighting, detailed illustration\n\
             \n\
             Server context — use this to interpret references correctly: {}",
            server_context
        );

        match self.query_llm_oneshot(system, user_prompt.to_string()).await {
            Ok(tags) => {
                info!("Expanded image prompt: {} -> {}", user_prompt, tags.trim());
                tags.trim().to_string()
            }
            Err(e) => {
                warn!("LLM prompt expansion failed, using raw prompt: {}", e);
                user_prompt.to_string()
            }
        }
    }

    /// Returns true if the message author has the ColumbianDrugLords role.
    async fn is_raid_admin(&self, ctx: &Context, msg: &Message) -> bool {
        let guild_id = match msg.guild_id {
            Some(g) => g,
            None => return false,
        };
        let member = match guild_id.member(&ctx.http, msg.author.id).await {
            Ok(m) => m,
            Err(_) => return false,
        };
        let roles = match guild_id.roles(&ctx.http).await {
            Ok(r) => r,
            Err(_) => return false,
        };
        member.roles.iter().any(|rid| {
            roles.get(rid).map(|r| r.name == "ColumbianDrugLords").unwrap_or(false)
        })
    }

    async fn handle_raid_command(&self, ctx: &Context, msg: &Message, args: &str) {
        let parts: Vec<&str> = args.splitn(2, ' ').collect();
        let subcommand = parts.first().copied().unwrap_or("").to_lowercase();
        let rest = parts.get(1).copied().unwrap_or("").trim();

        match subcommand.as_str() {
            "list" => {
                let now = Utc::now().timestamp();
                let event_list = {
                    let conn = self.db.lock().await;
                    db::get_upcoming_events(&conn, now).unwrap_or_default()
                };
                if event_list.is_empty() {
                    if let Err(why) = msg.channel_id.say(&ctx.http, "No upcoming raids scheduled.").await {
                        error!("{:?}", why);
                    }
                    return;
                }
                let mut response = "**Upcoming Raids:**\n".to_string();
                for e in &event_list {
                    response.push_str(&format!("  {}\n", events::format_event_summary(e)));
                }
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("{:?}", why);
                }
            }

            "info" => {
                let id: i64 = match rest.parse() {
                    Ok(n) => n,
                    Err(_) => {
                        let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid info <id>`").await;
                        return;
                    }
                };
                let (event_opt, signups) = {
                    let conn = self.db.lock().await;
                    let e = db::get_event(&conn, id).unwrap_or(None);
                    let s = db::get_signups(&conn, id).unwrap_or_default();
                    (e, s)
                };
                match event_opt {
                    None => { let _ = msg.channel_id.say(&ctx.http, "Raid not found.").await; }
                    Some(event) => {
                        let db_ref = Arc::clone(&self.db);
                        let text = events::format_event_detail_with_mentions(
                            &event,
                            &signups,
                            |uid| format!("<@{}>", uid),
                            |char_name| {
                                let conn = db_ref.try_lock().ok()?;
                                db::get_character_info(&conn, char_name).ok().flatten()
                            },
                        );
                        if let Err(why) = msg.channel_id.say(&ctx.http, &text).await {
                            error!("{:?}", why);
                        }
                    }
                }
            }

            "create" => {
                if !self.is_raid_admin(ctx, msg).await {
                    let _ = msg.channel_id.say(&ctx.http, "You need the **ColumbianDrugLords** role to create raids.").await;
                    return;
                }
                // Format: <name> | <date/time> [| raid_type] [| notes]
                let segments: Vec<&str> = rest.splitn(4, '|').map(str::trim).collect();
                if segments.len() < 2 {
                    let _ = msg.channel_id.say(&ctx.http,
                        "Usage: `!raid create <name> | <date/time> [| raid_type] [| notes]`\n\
                         Example: `!raid create Kara Tuesday | 3/29 at 7pm | Karazhan | Sign up by Monday!`"
                    ).await;
                    return;
                }
                let name = segments[0];
                let date_str = segments[1];
                let raid_type = segments.get(2).copied().filter(|s| !s.is_empty());
                let notes = segments.get(3).copied().filter(|s| !s.is_empty());

                let event_time = match datetime_parse::parse_event_time(date_str) {
                    Some(dt) => dt.timestamp(),
                    None => {
                        let _ = msg.channel_id.say(&ctx.http,
                            "Couldn't parse that date. Try: `3/29 at 7pm`, `march 29 2026 at 7pm`, `3/29/2026 19:00`"
                        ).await;
                        return;
                    }
                };

                let channel_id = msg.channel_id.to_string();
                let created_by = msg.author.id.to_string();

                // Check for per-event system prompt in config or leave blank for now
                let event_id = {
                    let conn = self.db.lock().await;
                    db::create_event(&conn, name, raid_type, event_time, &channel_id, notes, None, &created_by)
                        .map_err(|e| { error!("DB error creating event: {}", e); e })
                        .ok()
                };

                let event_id = match event_id {
                    Some(id) => id,
                    None => {
                        let _ = msg.channel_id.say(&ctx.http, "Failed to create event.").await;
                        return;
                    }
                };

                // Fetch back the full event for formatting
                let event = {
                    let conn = self.db.lock().await;
                    db::get_event(&conn, event_id).unwrap_or(None)
                };

                let event = match event {
                    Some(e) => e,
                    None => {
                        let _ = msg.channel_id.say(&ctx.http, "Event created but couldn't retrieve it.").await;
                        return;
                    }
                };

                // Attempt AI-generated announcement
                let global_prompt = {
                    let conn = self.db.lock().await;
                    db::get_config(&conn, "system_prompt").ok().flatten().unwrap_or_default()
                };
                let system = events::event_llm_system_prompt(&event, &global_prompt);
                let user_prompt = events::announcement_user_prompt(&event);

                let ai_text = match self.query_llm_oneshot(system, user_prompt).await {
                    Ok(text) => format!("\n\n> {}", text.trim()),
                    Err(_) => String::new(),
                };

                let summary = events::format_event_summary(&event);
                let response = format!(
                    "Raid scheduled! {}{}\n\nSign up with `!raid signup {} <character> [tank|healer|dps]`",
                    summary, ai_text, event_id
                );
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("{:?}", why);
                }
            }

            "signup" => {
                // !raid signup <id> <character> [role]
                let parts: Vec<&str> = rest.splitn(3, ' ').collect();
                if parts.len() < 2 {
                    let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid signup <id> <character> [tank|healer|dps]`").await;
                    return;
                }
                let id: i64 = match parts[0].parse() {
                    Ok(n) => n,
                    Err(_) => { let _ = msg.channel_id.say(&ctx.http, "Invalid event id.").await; return; }
                };
                let character = parts[1];
                let role = events::normalize_role(parts.get(2).copied().unwrap_or("unknown"));
                let user_id = msg.author.id.to_string();

                let result = {
                    let conn = self.db.lock().await;
                    db::signup_for_event(&conn, id, character, &user_id, role)
                };
                match result {
                    Ok(db::SignupResult::Added) => {
                        let _ = msg.channel_id.say(&ctx.http,
                            &format!("**{}** signed up as {} for raid #{}!", character, role, id)
                        ).await;
                    }
                    Ok(db::SignupResult::Updated) => {
                        let _ = msg.channel_id.say(&ctx.http,
                            &format!("Updated your signup to **{}** ({}) for raid #{}.", character, role, id)
                        ).await;
                    }
                    Ok(db::SignupResult::EventNotFound) => {
                        let _ = msg.channel_id.say(&ctx.http, "Raid not found or cancelled.").await;
                    }
                    Err(e) => {
                        error!("DB error signing up: {}", e);
                        let _ = msg.channel_id.say(&ctx.http, "Failed to sign up.").await;
                    }
                }
            }

            "drop" => {
                let id: i64 = match rest.parse() {
                    Ok(n) => n,
                    Err(_) => { let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid drop <id>`").await; return; }
                };
                let user_id = msg.author.id.to_string();
                let result = {
                    let conn = self.db.lock().await;
                    db::remove_signup(&conn, id, &user_id)
                };
                match result {
                    Ok(true) => { let _ = msg.channel_id.say(&ctx.http, &format!("Removed your signup from raid #{}.", id)).await; }
                    Ok(false) => { let _ = msg.channel_id.say(&ctx.http, "You weren't signed up for that raid.").await; }
                    Err(e) => { error!("{}", e); let _ = msg.channel_id.say(&ctx.http, "Failed to remove signup.").await; }
                }
            }

            "cancel" => {
                if !self.is_raid_admin(ctx, msg).await {
                    let _ = msg.channel_id.say(&ctx.http, "You need the **ColumbianDrugLords** role to cancel raids.").await;
                    return;
                }
                let id: i64 = match rest.parse() {
                    Ok(n) => n,
                    Err(_) => { let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid cancel <id>`").await; return; }
                };
                let result = {
                    let conn = self.db.lock().await;
                    db::cancel_event(&conn, id)
                };
                match result {
                    Ok(true) => { let _ = msg.channel_id.say(&ctx.http, &format!("Raid #{} has been cancelled.", id)).await; }
                    Ok(false) => { let _ = msg.channel_id.say(&ctx.http, "Raid not found.").await; }
                    Err(e) => { error!("{}", e); let _ = msg.channel_id.say(&ctx.http, "Failed to cancel raid.").await; }
                }
            }

            "setprompt" => {
                if !self.is_raid_admin(ctx, msg).await {
                    let _ = msg.channel_id.say(&ctx.http, "You need the **ColumbianDrugLords** role.").await;
                    return;
                }
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid setprompt <id> <prompt>`").await;
                    return;
                }
                let id: i64 = match parts[0].parse() {
                    Ok(n) => n,
                    Err(_) => { let _ = msg.channel_id.say(&ctx.http, "Invalid event id.").await; return; }
                };
                let prompt = parts[1];
                let result = {
                    let conn = self.db.lock().await;
                    db::set_event_system_prompt(&conn, id, prompt)
                };
                match result {
                    Ok(true) => { let _ = msg.channel_id.say(&ctx.http, &format!("AI prompt updated for raid #{}.", id)).await; }
                    Ok(false) => { let _ = msg.channel_id.say(&ctx.http, "Raid not found.").await; }
                    Err(e) => { error!("{}", e); let _ = msg.channel_id.say(&ctx.http, "Failed to update prompt.").await; }
                }
            }

            "remind" => {
                if !self.is_raid_admin(ctx, msg).await {
                    let _ = msg.channel_id.say(&ctx.http, "You need the **ColumbianDrugLords** role.").await;
                    return;
                }
                let id: i64 = match rest.parse() {
                    Ok(n) => n,
                    Err(_) => { let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid remind <id>`").await; return; }
                };
                self.post_raid_reminder(ctx, msg.channel_id, id).await;
            }

            "setchannels" => {
                if !self.is_raid_admin(ctx, msg).await {
                    let _ = msg.channel_id.say(&ctx.http, "You need the **ColumbianDrugLords** role.").await;
                    return;
                }
                // Parse channel mentions: <#123456789> <#987654321>
                let channel_ids: Vec<u64> = rest
                    .split_whitespace()
                    .filter_map(|tok| {
                        tok.trim_start_matches("<#")
                            .trim_end_matches('>')
                            .parse::<u64>()
                            .ok()
                    })
                    .collect();
                if channel_ids.is_empty() {
                    let _ = msg.channel_id.say(&ctx.http, "Usage: `!raid setchannels #chan1 #chan2`").await;
                    return;
                }
                {
                    let conn = self.db.lock().await;
                    for (i, id) in channel_ids.iter().enumerate() {
                        let key = format!("reminder_channel_{}", i);
                        let _ = db::set_config(&conn, &key, &id.to_string());
                    }
                    // Clear any extra slots beyond what was just set
                    for i in channel_ids.len()..5 {
                        let key = format!("reminder_channel_{}", i);
                        let _ = db::set_config(&conn, &key, "");
                    }
                }
                let names: Vec<String> = channel_ids.iter().map(|id| format!("<#{}>", id)).collect();
                let _ = msg.channel_id.say(&ctx.http,
                    &format!("Daily reminders will be posted to: {}", names.join(", "))
                ).await;
            }

            _ => {
                let _ = msg.channel_id.say(&ctx.http,
                    "Unknown raid subcommand. Use `!help` to see available commands."
                ).await;
            }
        }
    }

    /// Post an AI-generated reminder for a specific raid to a channel.
    async fn post_raid_reminder(&self, ctx: &Context, channel_id: ChannelId, event_id: i64) {
        let (event_opt, signups) = {
            let conn = self.db.lock().await;
            let e = db::get_event(&conn, event_id).unwrap_or(None);
            let s = db::get_signups(&conn, event_id).unwrap_or_default();
            (e, s)
        };
        let event = match event_opt {
            Some(e) => e,
            None => {
                let _ = channel_id.say(&ctx.http, "Raid not found.").await;
                return;
            }
        };

        let global_prompt = {
            let conn = self.db.lock().await;
            db::get_config(&conn, "system_prompt").ok().flatten().unwrap_or_default()
        };

        let system = events::event_llm_system_prompt(&event, &global_prompt);
        let user_prompt = events::reminder_user_prompt(&event, signups.len());
        let ai_text = match self.query_llm_oneshot(system, user_prompt).await {
            Ok(text) => text.trim().to_string(),
            Err(_) => String::new(),
        };

        let summary = events::format_event_summary(&event);
        let signup_text = if signups.is_empty() {
            "No signups yet!".to_string()
        } else {
            let mentions: Vec<String> = signups
                .iter()
                .map(|s| format!("{} (<@{}>)", s.character_name, s.discord_user_id))
                .collect();
            format!("**Signed up ({}):** {}", signups.len(), mentions.join(", "))
        };

        let mut response = format!("**RAID REMINDER**\n{}\n{}", summary, signup_text);
        if !ai_text.is_empty() {
            response.push_str(&format!("\n\n> {}", ai_text));
        }
        response.push_str(&format!("\n\nSign up: `!raid signup {} <character> [tank|healer|dps]`", event.id));

        if let Err(why) = channel_id.say(&ctx.http, &response).await {
            error!("Error posting reminder: {:?}", why);
        }
    }

    async fn generate_image(&self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let api_url = self
            .comfy_api_url
            .as_ref()
            .ok_or("COMFY_API_URL not configured")?;

        // Use nanoseconds of current time as a seed for variety
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;

        let positive_prompt = prompt.to_string();

        let workflow = serde_json::json!({
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": { "ckpt_name": COMFY_MODEL_NAME }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": { "width": width, "height": height, "batch_size": 1 }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": { "text": positive_prompt, "clip": ["4", 1] }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": COMFY_NEGATIVE_PROMPT,
                    "clip": ["4", 1]
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 14,
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": { "samples": ["3", 0], "vae": ["4", 2] }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": { "filename_prefix": "discord", "images": ["8", 0] }
            }
        });

        let resp = self
            .http_client
            .post(format!("{}/prompt", api_url))
            .json(&serde_json::json!({ "prompt": workflow }))
            .send()
            .await
            .map_err(|e| format!("Failed to reach ComfyUI: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("ComfyUI /prompt returned {}", resp.status()));
        }

        let prompt_resp: ComfyPromptResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse ComfyUI response: {}", e))?;

        let prompt_id = prompt_resp.prompt_id;

        // Poll /history until generation completes (timeout 120s)
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let filename = loop {
            if std::time::Instant::now() > deadline {
                return Err("Image generation timed out".to_string());
            }

            tokio::time::sleep(std::time::Duration::from_secs(2)).await;

            let history: serde_json::Value = self
                .http_client
                .get(format!("{}/history/{}", api_url, prompt_id))
                .send()
                .await
                .map_err(|e| format!("Failed to poll ComfyUI history: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Failed to parse history: {}", e))?;

            if let Some(filename) = history
                .get(&prompt_id)
                .and_then(|e| e.pointer("/outputs/9/images/0/filename"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
            {
                break filename;
            }
        };

        let image_resp = self
            .http_client
            .get(format!("{}/view", api_url))
            .query(&[("filename", filename.as_str()), ("subfolder", ""), ("type", "output")])
            .send()
            .await
            .map_err(|e| format!("Failed to fetch generated image: {}", e))?;

        if !image_resp.status().is_success() {
            return Err(format!("ComfyUI /view returned {}", image_resp.status()));
        }

        let bytes = image_resp
            .bytes()
            .await
            .map_err(|e| format!("Failed to read image bytes: {}", e))?;

        Ok(bytes.to_vec())
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Ignore messages from bots (including ourselves)
        if msg.author.bot {
            return;
        }

        // Respond to direct commands
        if msg.content.starts_with("!help") {
            let cap = {
                let conn = self.db.lock().await;
                db::get_config(&conn, "response_cap")
                    .ok()
                    .flatten()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(10)
            };
            let response = format!(
                "**Commands:**\n\
                 `!help` — Show this message\n\
                 `!ping` — Pong!\n\
                 `!hello` — Greet the bot\n\
                 `!imagine <prompt>` — Generate an image\n\
                 `!imagine portrait: <prompt>` — Generate a portrait (512×768)\n\
                 `!imagine landscape: <prompt>` — Generate a landscape (768×512)\n\
                 `!imagine debug: <prompt>` — Show expanded prompt without generating\n\
                 `!systemprompt [text]` — View or set the system prompt\n\
                 `!imageprompt [text]` — View or set the image generation context\n\
                 `!cap <1-500>` — Set response word cap (currently **{}**)\n\
                 `!health` — Show LLM and image gen server status\n\
                 `!clear` — Clear conversation history\n\
                 `!contextchannel` — Shared history per channel\n\
                 `!contextuser` — Separate history per user\n\
                 `!addcharacter <name>` — Track a WoW character\n\
                 `!removecharacter <name>` — Stop tracking a character\n\
                 `!levelcheck` — Check levels of tracked characters (with insults)\n\
                 `!levelcheckraw` — Check levels without insults\n\
                 \n\
                 **Raid Scheduling:**\n\
                 `!raid create <name> | <date/time> [| raid_type] [| notes]` — Schedule a raid *(ColumbianDrugLords only)*\n\
                 `!raid list` — List upcoming raids\n\
                 `!raid info <id>` — Show raid details and signups\n\
                 `!raid signup <id> <character> [tank|healer|dps]` — Sign up for a raid\n\
                 `!raid drop <id>` — Remove yourself from a raid\n\
                 `!raid cancel <id>` — Cancel a raid *(ColumbianDrugLords only)*\n\
                 `!raid setprompt <id> <prompt>` — Set AI flavor prompt for a raid *(ColumbianDrugLords only)*\n\
                 `!raid remind <id>` — Post an AI-generated reminder *(ColumbianDrugLords only)*\n\
                 `!raid setchannels <#chan1> [#chan2]` — Set daily reminder channels *(ColumbianDrugLords only)*\n\
                 \n\
                 **Characters:**\n\
                 `!claim <character>` — Link a character to your Discord account\n\
                 `!unclaim <character>` — Unlink a character\n\
                 `!mycharacters` — Show your claimed characters\n\
                 \n\
                 Mention me to chat!",
                cap
            );
            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!ping") {
            if let Err(why) = msg.channel_id.say(&ctx.http, "Pong! 🏓").await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!hello") {
            let response = "IT'S CHRISTINITH! ARE YOU STUPID OR ARE YOU DEAF?!";
            if let Err(why) = msg.channel_id.say(&ctx.http, response).await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!health") {
            let typing = msg.channel_id.start_typing(&ctx.http);
            let response = self.health_check().await;
            drop(typing);
            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!systemprompt") {
            let new_prompt = msg.content.trim_start_matches("!systemprompt").trim();
            if new_prompt.is_empty() {
                // Show current prompt
                let conn = self.db.lock().await;
                let current = db::get_config(&conn, "system_prompt")
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                let response = format!("**Current system prompt:**\n{}", current);
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("Error sending message: {:?}", why);
                }
            } else {
                let conn = self.db.lock().await;
                match db::set_config(&conn, "system_prompt", new_prompt) {
                    Ok(_) => {
                        info!("{} updated system prompt to: {}", msg.author.name, new_prompt);
                        if let Err(why) = msg.channel_id.say(&ctx.http, "System prompt updated!").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                    Err(e) => {
                        error!("Failed to update system prompt: {}", e);
                        if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to update system prompt.").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!imageprompt") {
            let new_prompt = msg.content.trim_start_matches("!imageprompt").trim();
            if new_prompt.is_empty() {
                let conn = self.db.lock().await;
                let current = db::get_config(&conn, "image_prompt")
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                let response = format!("**Current image prompt context:**\n{}", current);
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("Error sending message: {:?}", why);
                }
            } else {
                let conn = self.db.lock().await;
                match db::set_config(&conn, "image_prompt", new_prompt) {
                    Ok(_) => {
                        info!("{} updated image prompt to: {}", msg.author.name, new_prompt);
                        if let Err(why) = msg.channel_id.say(&ctx.http, "Image prompt context updated!").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                    Err(e) => {
                        error!("Failed to update image prompt: {}", e);
                        if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to update image prompt context.").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!cap") {
            let arg = msg.content.trim_start_matches("!cap").trim();
            if arg.is_empty() {
                let cap = {
                    let conn = self.db.lock().await;
                    db::get_config(&conn, "response_cap")
                        .ok()
                        .flatten()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(10)
                };
                let response = format!("Response word cap is currently **{}**. Usage: `!cap <1-500>`", cap);
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("Error sending message: {:?}", why);
                }
            } else {
                match arg.parse::<u32>() {
                    Ok(n) if (1..=500).contains(&n) => {
                        let conn = self.db.lock().await;
                        match db::set_config(&conn, "response_cap", &n.to_string()) {
                            Ok(_) => {
                                info!("{} set response cap to {}", msg.author.name, n);
                                let response = format!("Response word cap set to **{}**.", n);
                                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                                    error!("Error sending message: {:?}", why);
                                }
                            }
                            Err(e) => {
                                error!("Failed to set response cap: {}", e);
                                if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to save cap.").await {
                                    error!("Error sending message: {:?}", why);
                                }
                            }
                        }
                    }
                    _ => {
                        if let Err(why) = msg.channel_id.say(&ctx.http, "Cap must be a number between 1 and 500.").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!clear") {
            let conn = self.db.lock().await;
            let channel_id = msg.channel_id.to_string();
            let mode = db::get_context_mode(&conn, &channel_id).unwrap_or_else(|_| "channel".to_string());
            let context_key = match mode.as_str() {
                "user" => format!("{}:{}", channel_id, msg.author.id),
                _ => channel_id,
            };
            match db::clear_messages(&conn, &context_key) {
                Ok(n) => {
                    let response = format!("Cleared {} messages.", n);
                    if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("Failed to clear messages: {}", e);
                }
            }
            return;
        }

        if msg.content.starts_with("!contextchannel") {
            let conn = self.db.lock().await;
            let channel_id = msg.channel_id.to_string();
            match db::set_context_mode(&conn, &channel_id, "channel") {
                Ok(_) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Context mode set to **channel** — everyone shares history here.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("Failed to set context mode: {}", e);
                }
            }
            return;
        }

        if msg.content.starts_with("!contextuser") {
            let conn = self.db.lock().await;
            let channel_id = msg.channel_id.to_string();
            match db::set_context_mode(&conn, &channel_id, "user") {
                Ok(_) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Context mode set to **user** — everyone gets their own history here.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("Failed to set context mode: {}", e);
                }
            }
            return;
        }

        if msg.content.starts_with("!addcharacter") {
            let name = msg.content.trim_start_matches("!addcharacter").trim();
            if name.is_empty() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "Usage: `!addcharacter <name>`").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            if self.battlenet_auth.is_none() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "Battle.net API not configured.").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let typing = msg.channel_id.start_typing(&ctx.http);
            match self.fetch_wow_character(name).await {
                Ok(character) => {
                    let conn = self.db.lock().await;
                    let added_by = msg.author.id.to_string();
                    match db::add_tracked_character(&conn, &character.name, &added_by) {
                        Ok(true) => {
                            let response = format!(
                                "Now tracking **{}** — Level {} {} {}",
                                character.name, character.level, character.race.name, character.character_class.name
                            );
                            drop(typing);
                            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                                error!("Error sending message: {:?}", why);
                            }
                        }
                        Ok(false) => {
                            let response = format!(
                                "**{}** is already tracked — Level {} {} {}",
                                character.name, character.level, character.race.name, character.character_class.name
                            );
                            drop(typing);
                            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                                error!("Error sending message: {:?}", why);
                            }
                        }
                        Err(e) => {
                            error!("DB error adding character: {}", e);
                            drop(typing);
                            if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to save character.").await {
                                error!("Error sending message: {:?}", why);
                            }
                        }
                    }
                }
                Err(e) => {
                    drop(typing);
                    if let Err(why) = msg.channel_id.say(&ctx.http, &e).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!removecharacter") {
            let name = msg.content.trim_start_matches("!removecharacter").trim();
            if name.is_empty() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "Usage: `!removecharacter <name>`").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let conn = self.db.lock().await;
            match db::remove_tracked_character(&conn, name) {
                Ok(true) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("Removed **{}** from tracking.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Ok(false) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("**{}** is not being tracked.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("DB error removing character: {}", e);
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to remove character.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!mycharacters") {
            let user_id = msg.author.id.to_string();
            let chars = {
                let conn = self.db.lock().await;
                db::get_user_characters(&conn, &user_id).unwrap_or_default()
            };
            let response = if chars.is_empty() {
                "You have no claimed characters. Use `!claim <name>` to link one.".to_string()
            } else {
                let lines: Vec<String> = chars.iter().map(|c| {
                    match (&c.class, &c.race, &c.level) {
                        (Some(cls), Some(race), Some(lvl)) => format!("**{}** — Level {} {} {}", c.name, lvl, race, cls),
                        (Some(cls), _, _) => format!("**{}** — {}", c.name, cls),
                        _ => format!("**{}**", c.name),
                    }
                }).collect();
                lines.join("\n")
            };
            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!unclaim ") {
            let name = msg.content.trim_start_matches("!unclaim").trim();
            let user_id = msg.author.id.to_string();
            let conn = self.db.lock().await;
            match db::unclaim_character(&conn, name, &user_id) {
                Ok(true) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("Unlinked **{}** from your account.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Ok(false) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("**{}** is not claimed by you.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("DB error unclaiming character: {}", e);
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to unclaim character.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!claim ") {
            let name = msg.content.trim_start_matches("!claim").trim();
            if name.is_empty() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "Usage: `!claim <character>`").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            // Optionally fetch WoW data to store class/race/level
            let (class, race, level) = if self.battlenet_auth.is_some() {
                let typing = msg.channel_id.start_typing(&ctx.http);
                let result = match self.fetch_wow_character(name).await {
                    Ok(c) => (Some(c.character_class.name), Some(c.race.name), Some(c.level as i64)),
                    Err(_) => (None, None, None),
                };
                drop(typing);
                result
            } else {
                (None, None, None)
            };

            let user_id = msg.author.id.to_string();
            let conn = self.db.lock().await;
            match db::claim_character(&conn, name, &user_id, class.as_deref(), race.as_deref(), level) {
                Ok(db::ClaimResult::Claimed) => {
                    let detail = match (&class, &race, &level) {
                        (Some(c), Some(r), Some(l)) => format!(" — Level {} {} {}", l, r, c),
                        _ => String::new(),
                    };
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("**{}** is now linked to your account{}.", name, detail)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Ok(db::ClaimResult::AlreadyYours) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("**{}** is already linked to your account.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Ok(db::ClaimResult::TakenByOther) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, &format!("**{}** is already claimed by someone else.", name)).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("DB error claiming character: {}", e);
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Failed to claim character.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
            }
            return;
        }

        if msg.content.starts_with("!raid") {
            let args = msg.content.trim_start_matches("!raid").trim();
            self.handle_raid_command(&ctx, &msg, args).await;
            return;
        }

        if msg.content.starts_with("!levelcheck") {
            let use_insults = !msg.content.starts_with("!levelcheckraw");

            if self.battlenet_auth.is_none() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "Battle.net API not configured.").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let names = {
                let conn = self.db.lock().await;
                db::get_tracked_characters(&conn).unwrap_or_default()
            };

            if names.is_empty() {
                if let Err(why) = msg.channel_id.say(&ctx.http, "No characters tracked. Use `!addcharacter <name>` to add one.").await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let typing = msg.channel_id.start_typing(&ctx.http);
            let futures: Vec<_> = names
                .iter()
                .map(|name| self.fetch_wow_character(name))
                .collect();
            let results = join_all(futures).await;

            let mut entries: Vec<(String, u32, String)> = Vec::new();
            let mut errors: Vec<String> = Vec::new();

            for (name, result) in names.iter().zip(results) {
                match result {
                    Ok(c) => entries.push((
                        c.name,
                        c.level,
                        format!("{} {}", c.race.name, c.character_class.name),
                    )),
                    Err(e) => errors.push(format!("{}: {}", name, e)),
                }
            }

            entries.sort_by(|a, b| b.1.cmp(&a.1));

            // Fetch insults in parallel if LLM is configured and this isn't !levelcheckraw
            let insults: Vec<Option<String>> = if use_insults && self.llama_api_url.is_some() {
                let system_prompt = {
                    let conn = self.db.lock().await;
                    db::get_config(&conn, "system_prompt")
                        .ok()
                        .flatten()
                        .unwrap_or_default()
                };

                let insult_futures: Vec<_> = entries
                    .iter()
                    .map(|(name, level, desc)| {
                        let sys = system_prompt.clone();
                        let prompt = format!(
                            "Give a 1-5 word insult for a level {} {} named {}. Reply with ONLY the insult, nothing else.",
                            level, desc, name
                        );
                        self.query_llm_oneshot(sys, prompt)
                    })
                    .collect();

                join_all(insult_futures)
                    .await
                    .into_iter()
                    .map(|r| r.ok())
                    .collect()
            } else {
                entries.iter().map(|_| None).collect()
            };

            let mut response = String::from("**Level Check — Nightslayer**\n");
            for ((name, level, desc), insult) in entries.iter().zip(insults.iter()) {
                match insult {
                    Some(text) => response.push_str(&format!(
                        "  {} — Level {} {} — *{}*\n", name, level, desc, text.trim()
                    )),
                    None => response.push_str(&format!(
                        "  {} — Level {} {}\n", name, level, desc
                    )),
                }
            }
            for err in &errors {
                response.push_str(&format!("  ⚠ {}\n", err));
            }

            drop(typing);
            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                error!("Error sending message: {:?}", why);
            }
            return;
        }

        if msg.content.starts_with("!imagine") {
            let raw = msg.content.trim_start_matches("!imagine").trim();

            if raw.is_empty() {
                if let Err(why) = msg.channel_id.say(
                    &ctx.http,
                    "Usage: `!imagine [portrait:|landscape:|debug:] <prompt>`",
                ).await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            // debug: mode — show expanded prompt without generating
            if let Some(p) = raw.strip_prefix("debug:") {
                let prompt = p.trim();
                let typing = msg.channel_id.start_typing(&ctx.http);
                let expanded = self.expand_image_prompt(prompt).await;
                drop(typing);
                let response = format!("**Expanded prompt:**\n```\n{}\n```", expanded);
                if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let (prompt, width, height) = if let Some(p) = raw.strip_prefix("portrait:") {
                (p.trim(), 512u32, 768u32)
            } else if let Some(p) = raw.strip_prefix("landscape:") {
                (p.trim(), 768u32, 512u32)
            } else {
                (raw, 512u32, 512u32)
            };

            let typing = msg.channel_id.start_typing(&ctx.http);

            // Expand prompt using LLM, falling back to raw prompt on failure
            let expanded = self.expand_image_prompt(prompt).await;

            match self.generate_image(&expanded, width, height).await {
                Ok(image_bytes) => {
                    drop(typing);
                    let attachment = CreateAttachment::bytes(image_bytes, "generated.png");
                    if let Err(why) = msg
                        .channel_id
                        .send_message(&ctx.http, CreateMessage::new().add_file(attachment))
                        .await
                    {
                        error!("Error sending image: {:?}", why);
                        if let Err(why) = msg.channel_id.say(&ctx.http, "Sorry, failed to send the generated image.").await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                }
                Err(e) => {
                    drop(typing);
                    error!("Image generation error: {}", e);
                    if let Err(why) = msg.channel_id.say(&ctx.http, "Sorry, no available image generator at this time.").await {
                        error!("Error sending message: {:?}", why);
                    }
                }
            }
            return;
        }

        // When mentioned, send the message to llama.cpp
        if msg.mentions_me(&ctx.http).await.unwrap_or(false) {
            info!("Received message from {}: {}", msg.author.name, msg.content);

            // Show typing indicator while waiting for LLM
            let typing = msg.channel_id.start_typing(&ctx.http);

            // Strip the bot mention from the message to get the actual question
            let content = msg
                .content
                .split_once('>')
                .map(|(_, rest)| rest.trim())
                .unwrap_or(&msg.content);

            if content.is_empty() {
                if let Err(why) = msg
                    .channel_id
                    .say(&ctx.http, "You mentioned me but didn't say anything!")
                    .await
                {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            let channel_id = msg.channel_id.to_string();
            let context_key = {
                let conn = self.db.lock().await;
                let mode = db::get_context_mode(&conn, &channel_id).unwrap_or_else(|_| "channel".to_string());
                match mode.as_str() {
                    "user" => format!("{}:{}", channel_id, msg.author.id),
                    _ => channel_id.clone(),
                }
            };
            let response = match self.ask_llama(&context_key, content).await {
                Ok(reply) => reply,
                Err(e) => {
                    error!("LLM error: {}", e);
                    format!("Sorry, I couldn't get a response: {}", e)
                }
            };

            drop(typing);

            // Discord has a 2000 char limit - truncate if needed
            let response = if response.len() > 1990 {
                format!("{}...", &response[..1990])
            } else {
                response
            };

            if let Err(why) = msg.channel_id.say(&ctx.http, &response).await {
                error!("Error sending message: {:?}", why);
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        info!("{} is connected and ready!", ready.user.name);

        let db = Arc::clone(&self.db);
        let http_client = self.http_client.clone();
        let llama_api_url = self.llama_api_url.clone();

        tokio::spawn(async move {
            // Pseudo-handler for LLM calls in background task
            let handler = BackgroundHandler { db: Arc::clone(&db), http_client, llama_api_url };
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));

            loop {
                interval.tick().await;
                handler.run_reminder_check(&ctx).await;
            }
        });
    }
}

struct BackgroundHandler {
    db: Arc<Mutex<Connection>>,
    http_client: HttpClient,
    llama_api_url: Option<String>,
}

impl BackgroundHandler {
    async fn query_llm_oneshot(&self, system_prompt: String, user_message: String) -> Result<String, String> {
        let api_url = self
            .llama_api_url
            .as_ref()
            .ok_or("LLAMA_API_URL not configured")?;

        let messages = vec![
            ChatMessage { role: "system".to_string(), content: system_prompt },
            ChatMessage { role: "user".to_string(), content: user_message },
        ];
        let request = ChatRequest {
            messages,
            temperature: 0.4,
            stop: vec![
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
                "</s>".to_string(),
                "[INST]".to_string(),
            ],
        };
        let response = self
            .http_client
            .post(format!("{}/v1/chat/completions", api_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("LLM request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("LLM returned {}", response.status()));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from model".to_string())
    }

    async fn run_reminder_check(&self, ctx: &Context) {
        let now = Utc::now().timestamp();
        // Window: events starting within the next hour
        let one_hour = now + 3600;
        // Window: events starting within the next 7 days (daily digest)
        let seven_days = now + (7 * 24 * 3600);

        let reminder_channels: Vec<ChannelId> = {
            let conn = self.db.lock().await;
            (0..5)
                .filter_map(|i| {
                    let key = format!("reminder_channel_{}", i);
                    db::get_config(&conn, &key)
                        .ok()
                        .flatten()
                        .filter(|s| !s.is_empty())
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(ChannelId::new)
                })
                .collect()
        };

        if reminder_channels.is_empty() {
            return;
        }

        // Imminent reminders: events within the next hour
        let imminent = {
            let conn = self.db.lock().await;
            db::get_events_in_window(&conn, now, one_hour).unwrap_or_default()
        };

        for event in &imminent {
            let signups = {
                let conn = self.db.lock().await;
                db::get_signups(&conn, event.id).unwrap_or_default()
            };

            let global_prompt = {
                let conn = self.db.lock().await;
                db::get_config(&conn, "system_prompt").ok().flatten().unwrap_or_default()
            };
            let system = events::event_llm_system_prompt(event, &global_prompt);
            let user_prompt = events::reminder_user_prompt(event, signups.len());
            let ai_text = match self.query_llm_oneshot(system, user_prompt).await {
                Ok(t) => t.trim().to_string(),
                Err(_) => String::new(),
            };

            let signup_text = if signups.is_empty() {
                "No signups yet!".to_string()
            } else {
                let mentions: Vec<String> = signups
                    .iter()
                    .map(|s| format!("{} (<@{}>)", s.character_name, s.discord_user_id))
                    .collect();
                format!("**Signed up ({}):** {}", signups.len(), mentions.join(", "))
            };

            let summary = events::format_event_summary(event);
            let mut msg_text = format!("🔔 **RAID STARTING SOON**\n{}\n{}", summary, signup_text);
            if !ai_text.is_empty() {
                msg_text.push_str(&format!("\n\n> {}", ai_text));
            }

            for channel in &reminder_channels {
                if let Err(e) = channel.say(&ctx.http, &msg_text).await {
                    error!("Failed to send imminent reminder to {}: {:?}", channel, e);
                }
            }
        }

        // Daily digest: fire once per calendar day (UTC), not every hour
        let today_utc = Utc::now().format("%Y-%m-%d").to_string();
        let last_digest = {
            let conn = self.db.lock().await;
            db::get_config(&conn, "last_digest_date").ok().flatten().unwrap_or_default()
        };

        if last_digest != today_utc {
            let daily_events = {
                let conn = self.db.lock().await;
                db::get_events_in_window(&conn, now, seven_days).unwrap_or_default()
            };

            if !daily_events.is_empty() {
                let mut digest = "📅 **Upcoming Raids (next 7 days):**\n".to_string();
                for event in &daily_events {
                    let signup_count = {
                        let conn = self.db.lock().await;
                        db::get_signups(&conn, event.id).map(|s| s.len()).unwrap_or(0)
                    };
                    digest.push_str(&format!(
                        "  {} — {} signed up\n",
                        events::format_event_summary(event),
                        signup_count
                    ));
                }
                digest.push_str("\nSign up with `!raid signup <id> <character>`");

                for channel in &reminder_channels {
                    if let Err(e) = channel.say(&ctx.http, &digest).await {
                        error!("Failed to send daily digest to {}: {:?}", channel, e);
                    }
                }
            }

            // Mark digest as sent for today even if no events (avoids re-checking every hour)
            let conn = self.db.lock().await;
            let _ = db::set_config(&conn, "last_digest_date", &today_utc);
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get Discord token from environment
    let token = env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment");

    // Get llama.cpp API URL (optional - bot works without it but can't answer LLM questions)
    let llama_api_url = env::var("LLAMA_API_URL").ok();
    if llama_api_url.is_some() {
        info!("LLAMA_API_URL configured: {}", llama_api_url.as_ref().unwrap());
    } else {
        warn!("LLAMA_API_URL not set - LLM features disabled");
    }

    // Parse named LLM hosts for health checks (e.g. "reflect=http://reflect.local:8080,yoga=http://yoga.local:8080")
    let llama_hosts: Vec<(String, String)> = env::var("LLAMA_HOSTS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|entry| {
            let mut parts = entry.splitn(2, '=');
            let name = parts.next()?.trim().to_string();
            let url = parts.next()?.trim().to_string();
            if name.is_empty() || url.is_empty() { None } else { Some((name, url)) }
        })
        .collect();
    if !llama_hosts.is_empty() {
        info!("LLAMA_HOSTS configured: {:?}", llama_hosts.iter().map(|(n, _)| n).collect::<Vec<_>>());
    }

    // Get ComfyUI API URL (optional - bot works without it but !imagine will fail gracefully)
    let comfy_api_url = env::var("COMFY_API_URL").ok();
    if comfy_api_url.is_some() {
        info!("COMFY_API_URL configured: {}", comfy_api_url.as_ref().unwrap());
    } else {
        warn!("COMFY_API_URL not set - image generation disabled");
    }

    // Parse named ComfyUI hosts for health checks (e.g. "reflect=http://reflect.local:8188")
    let comfy_hosts: Vec<(String, String)> = env::var("COMFY_HOSTS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|entry| {
            let mut parts = entry.splitn(2, '=');
            let name = parts.next()?.trim().to_string();
            let url = parts.next()?.trim().to_string();
            if name.is_empty() || url.is_empty() { None } else { Some((name, url)) }
        })
        .collect();
    if !comfy_hosts.is_empty() {
        info!("COMFY_HOSTS configured: {:?}", comfy_hosts.iter().map(|(n, _)| n).collect::<Vec<_>>());
    }

    // Get Battle.net credentials (optional)
    let battlenet_auth = match (
        env::var("BATTLENET_CLIENT_ID"),
        env::var("BATTLENET_CLIENT_SECRET"),
    ) {
        (Ok(id), Ok(secret)) => {
            info!("Battle.net API configured");
            Some(Arc::new(Mutex::new(BattleNetAuth::new(id, secret))))
        }
        _ => {
            warn!("BATTLENET_CLIENT_ID/SECRET not set — WoW features disabled");
            None
        }
    };

    // Initialize database
    let db_path = env::var("DATABASE_PATH").unwrap_or_else(|_| "./discord-bot.db".to_string());
    info!("Opening database at {}", db_path);
    let conn = Connection::open(&db_path).expect("Failed to open database");
    db::init(&conn).expect("Failed to initialize database schema");
    let db = Arc::new(Mutex::new(conn));

    // Set gateway intents
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Create client
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler {
            http_client: HttpClient::new(),
            llama_api_url,
            llama_hosts,
            comfy_api_url,
            comfy_hosts,
            battlenet_auth,
            db,
        })
        .await
        .expect("Error creating client");

    info!("Starting Discord bot...");

    // Start the client
    if let Err(why) = client.start().await {
        error!("Client error: {:?}", why);
    }
}
