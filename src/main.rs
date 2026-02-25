mod db;

use futures::future::join_all;
use reqwest::Client as HttpClient;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use serenity::builder::{CreateAttachment, CreateMessage};
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

const COMFY_MODEL_NAME: &str = "pony-diffusion-xl.safetensors";

// Quality negatives for Pony Diffusion XL. No content restrictions — this is intentional.
const COMFY_NEGATIVE_PROMPT: &str =
    "score_4, score_5, score_6, worst quality, bad quality, low quality, lowres, \
     blurry, jpeg artifacts, compression artifacts, \
     bad anatomy, bad hands, extra fingers, missing fingers, deformed, mutation, extra limbs, \
     watermark, signature, text, logo, ugly";

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
    comfy_api_url: Option<String>,
    battlenet_auth: Option<Arc<Mutex<BattleNetAuth>>>,
    db: Arc<Mutex<Connection>>,
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

    // Rewrites a natural-language image request into booru-style tags for Pony Diffusion XL.
    // Falls back to the original prompt if the LLM is unavailable.
    async fn expand_image_prompt(&self, user_prompt: &str) -> String {
        let server_context = {
            let conn = self.db.lock().await;
            db::get_config(&conn, "system_prompt")
                .ok()
                .flatten()
                .unwrap_or_default()
        };

        let system = format!(
            "You are a prompt engineer for Pony Diffusion XL, a Stable Diffusion model trained \
             on booru-style image tags. Convert the image request into a comma-separated list of \
             tags. Include: subject details, species/race, clothing/armor, art style, lighting, \
             setting, and mood. Do NOT include score tags (added separately). \
             Reply with ONLY the tags, nothing else.\n\
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

        // Pony Diffusion XL uses score tags in positive prompt for quality control
        let positive_prompt = format!("score_9, score_8_up, score_7_up, {}", prompt);

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
                    "steps": 25,
                    "cfg": 6.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
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
                 `!imagine portrait: <prompt>` — Generate a portrait (832×1216)\n\
                 `!imagine landscape: <prompt>` — Generate a landscape (1216×832)\n\
                 `!systemprompt [text]` — View or set the system prompt\n\
                 `!cap <1-500>` — Set response word cap (currently **{}**)\n\
                 `!clear` — Clear conversation history\n\
                 `!contextchannel` — Shared history per channel\n\
                 `!contextuser` — Separate history per user\n\
                 `!addcharacter <name>` — Track a WoW character\n\
                 `!removecharacter <name>` — Stop tracking a character\n\
                 `!levelcheck` — Check levels of tracked characters (with insults)\n\
                 `!levelcheckraw` — Check levels without insults\n\
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
                    "Usage: `!imagine [portrait:|landscape:] <prompt>`",
                ).await {
                    error!("Error sending message: {:?}", why);
                }
                return;
            }

            // Parse optional aspect ratio prefix
            let (prompt, width, height) = if let Some(p) = raw.strip_prefix("portrait:") {
                (p.trim(), 832u32, 1216u32)
            } else if let Some(p) = raw.strip_prefix("landscape:") {
                (p.trim(), 1216u32, 832u32)
            } else {
                (raw, 1024u32, 1024u32)
            };

            let typing = msg.channel_id.start_typing(&ctx.http);

            // Expand to booru-style tags using LLM, falling back to raw prompt on failure
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

    async fn ready(&self, _: Context, ready: Ready) {
        info!("{} is connected and ready!", ready.user.name);
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

    // Get ComfyUI API URL (optional - bot works without it but !imagine will fail gracefully)
    let comfy_api_url = env::var("COMFY_API_URL").ok();
    if comfy_api_url.is_some() {
        info!("COMFY_API_URL configured: {}", comfy_api_url.as_ref().unwrap());
    } else {
        warn!("COMFY_API_URL not set - image generation disabled");
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
            comfy_api_url,
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
