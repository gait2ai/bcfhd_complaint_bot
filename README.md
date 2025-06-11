# BCFHD Complaint Management Bot

![BCFHD Logo](https://bcfhd.org/wp-content/uploads/2023/11/logo.png)

The BCFHD Complaint Management Bot is a Telegram-based application designed to help the Bena Charity for Human Development (BCFHD) efficiently manage and process beneficiary complaints and suggestions.

## Overview

This application provides:
- Automated complaint collection and classification
- Critical complaint identification and escalation
- Beneficiary data management
- Multilingual support (primarily Arabic and English)
- Integration with Google Sheets for data storage
- Email notifications for critical cases

## Features

- **Telegram Bot Interface**: Easy-to-use interface for beneficiaries to submit complaints
- **AI-Powered Processing**: Uses advanced AI models for complaint classification and criticality assessment
- **Google Sheets Integration**: Stores all complaint data in organized spreadsheets
- **Critical Case Handling**: Immediate email notifications for urgent cases
- **Beneficiary Profiles**: Maintains beneficiary information for repeat users
- **Multilingual Support**: Handles Arabic and English inputs with appropriate responses

## Technical Details

### System Architecture

```
bcfhd_complaint_bot/
├── app/                            
│   ├── bot/                      
│   │   ├── bot_core_logic.py     
│   │   └── bot_telegram_handlers.py 
│   ├── core/                     
│   │   ├── ai_handler.py         
│   │   ├── cache_manager.py      
│   │   └── prompt_builder.py     
│   └── config/                   
│       ├── config.yaml           
│       ├── bcfhd_system_prompt.txt 
│       ├── service_account.json    
│       ├── gmail_credentials.json  
│       └── token.json              
│
├── .env                            
├── requirements.txt                
└── main.py                         
```

### Core Components

1. **AI Handler**: Manages communication with AI models (OpenRouter and Hugging Face)
2. **Cache Manager**: Provides intelligent caching for performance optimization
3. **Prompt Builder**: Constructs dynamic prompts for the AI models
4. **Telegram Handlers**: Manages all user interactions and conversation flows

## Installation

### Prerequisites

- Python 3.9+
- Telegram bot token
- Google Service Account credentials
- OpenRouter API key (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/bcfhd-complaint-bot.git
   cd bcfhd-complaint-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the application:
   - Copy `default_config.yaml` to `app/config/config.yaml` and modify as needed
   - Place your Google Service Account credentials in `app/config/service_account.json`
   - Set up Gmail API credentials if using email notifications
   - Create a `.env` file with your Telegram bot token:
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token_here
     ```

4. Run the application:
   ```bash
   python main.py
   ```

## Configuration

The main configuration file (`app/config/config.yaml`) includes settings for:
- AI model selection and fallback options
- Cache behavior
- Google Sheets integration
- Logging preferences
- BCFHD-specific parameters

## Usage

Beneficiaries can interact with the bot through these commands:
- `/start` - Begin interaction with the bot
- `/complaint` - Submit a new complaint
- `/suggestion` - Submit a suggestion
- `/contact` - Get BCFHD contact information
- `/help` - Get help with using the bot

## Monitoring

The application provides comprehensive logging in `logs/bcfhd_bot.log` with rotation and retention settings configurable in the YAML file.

## License

This application is proprietary software developed for Bena Charity for Human Development (BCFHD).

## Credits

**Produced by:** AI Gate for Artificial Intelligence Applications  
**For:** Bena Charity for Human Development (BCFHD)  
**Contact:** [AI Gate](abuamr.dubai@gmail.com)

---

For support or additional information, please contact AI Gate  directly by abuamr.dubai@gmail.com