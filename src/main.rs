mod db;

use reqwest::Client as HttpClient;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

const HISTORY_LIMIT: usize = 10;

struct Handler {
    http_client: HttpClient,
    llama_api_url: Option<String>,
    db: Arc<Mutex<Connection>>,
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
                    last.content.push_str("\n(Reply in 10 words or less. Stay in character.)");
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
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Ignore messages from bots (including ourselves)
        if msg.author.bot {
            return;
        }

        // Respond to direct commands
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
