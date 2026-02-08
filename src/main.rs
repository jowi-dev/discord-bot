use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use std::env;
use tracing::{error, info};

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Ignore messages from bots (including ourselves)
        if msg.author.bot {
            return;
        }

        // Respond to messages that mention the bot
        if msg.mentions_me(&ctx.http).await.unwrap_or(false) {
            info!("Received message from {}: {}", msg.author.name, msg.content);

            // Simple response for now
            let response = format!(
                "Hello {}! I'm a Rust bot running on NixOS. You said: {}",
                msg.author.name,
                msg.content
            );

            if let Err(why) = msg.channel_id.say(&ctx.http, response).await {
                error!("Error sending message: {:?}", why);
            }
        }

        // Also respond to direct commands
        if msg.content.starts_with("!ping") {
            if let Err(why) = msg.channel_id.say(&ctx.http, "Pong! 🏓").await {
                error!("Error sending message: {:?}", why);
            }
        }

        if msg.content.starts_with("!hello") {
            let response = "IT'S CHRISTINITH! ARE YOU STUPID OR ARE YOU DEAF?!";
            if let Err(why) = msg.channel_id.say(&ctx.http, response).await {
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

    // Set gateway intents
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Create client
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler)
        .await
        .expect("Error creating client");

    info!("Starting Discord bot...");

    // Start the client
    if let Err(why) = client.start().await {
        error!("Client error: {:?}", why);
    }
}
