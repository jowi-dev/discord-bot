# Discord Bot

A simple, resource-efficient Discord bot written in Rust.

## Features

- Responds to mentions
- Responds to commands:
  - `!ping` - Returns "Pong!"
  - `!hello` - Returns a greeting message
- Uses ~10-20MB RAM
- Deployed as a systemd service on NixOS

## Setup

### 1. Create Discord Bot

1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Go to "Bot" tab
4. Click "Add Bot"
5. Under "Privileged Gateway Intents", enable:
   - MESSAGE CONTENT INTENT
6. Click "Reset Token" to get your bot token
7. Copy the token

### 2. Invite Bot to Server

1. Go to "OAuth2" > "URL Generator"
2. Select scopes: `bot`
3. Select permissions:
   - Read Messages/View Channels
   - Send Messages
   - Read Message History
4. Copy the generated URL and open it to invite the bot

### 3. Deploy to NixOS

On the alien laptop, create the token file:

```bash
sudo mkdir -p /etc/discord-bot
sudo nano /etc/discord-bot/token.env
```

Add this content (replace with your actual token):
```
DISCORD_TOKEN=your_bot_token_here
```

Save and set permissions:
```bash
sudo chmod 600 /etc/discord-bot/token.env
```

Then deploy from your Mac:
```bash
j remote deploy alien
```

### 4. Check Status

SSH into alien and check the bot:
```bash
j remote ssh alien
sudo systemctl status discord-bot
sudo journalctl -u discord-bot -f  # Follow logs
```

## Testing

In your Discord server:
- Mention the bot: `@YourBot hello`
- Try commands: `!ping` or `!hello`

The bot should respond!

## Troubleshooting

Check logs:
```bash
sudo journalctl -u discord-bot -n 50
```

Restart the bot:
```bash
sudo systemctl restart discord-bot
```
