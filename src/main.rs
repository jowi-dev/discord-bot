use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use std::env;
use tracing::{error, info, warn};

struct Handler {
    http_client: HttpClient,
    llama_api_url: Option<String>,
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
    async fn ask_llama(&self, user_message: &str) -> Result<String, String> {
        let api_url = self
            .llama_api_url
            .as_ref()
            .ok_or("LLAMA_API_URL not configured")?;

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
            }],
            temperature: 0.7,
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

            let response = match self.ask_llama(content).await {
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

    // Set gateway intents
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Create client
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler {
            http_client: HttpClient::new(),
            llama_api_url,
        })
        .await
        .expect("Error creating client");

    info!("Starting Discord bot...");

    // Start the client
    if let Err(why) = client.start().await {
        error!("Client error: {:?}", why);
    }
}
