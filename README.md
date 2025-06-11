# BCFHD Complaint Management Telegram Bot

## 🚩 Project Overview

The BCFHD Complaint Management Bot is an advanced Telegram bot designed to assist Bena Charity for Human Development (BCFHD) in efficiently receiving, processing, and managing complaints and suggestions from beneficiaries. The bot leverages Large Language Models (LLMs) for tasks such as complaint classification, summarization, translation, and critical issue identification, ensuring a responsive and intelligent interaction.

This system aims to streamline the feedback mechanism, improve data logging accuracy to Google Sheets, and enable timely intervention for urgent cases through automated email notifications.

## ✨ Key Features

*   **Telegram Integration:** User-friendly interaction via a dedicated Telegram bot.
*   **AI-Powered Complaint Processing:**
    *   Natural Language Understanding for user intent.
    *   Automated classification of complaints (Type, Category, Sensitivity) based on predefined keys.
    *   Summarization and translation of Arabic complaints into English for internal logging.
    *   Identification of critical complaints requiring immediate attention.
*   **Google Sheets Integration:**
    *   Automated logging of detailed complaint information into a structured Google Sheet.
    *   Management of a beneficiary database for recognizing returning users and pre-filling data.
*   **Critical Complaint Alerts:** Automated email notifications to designated personnel for urgent/critical cases.
*   **Multi-LLM Provider Support:** Utilizes an `AIHandler` module capable of interfacing with multiple LLM providers (e.g., OpenRouter, Hugging Face Inference API) with fallback and retry logic.
*   **Dynamic Prompt Engineering:** Employs a `PromptBuilder` to construct context-aware and task-specific prompts for LLMs.
*   **Intelligent Caching:** Implements a `CacheManager` for LLM responses and frequently accessed data to enhance performance and reduce API costs.
*   **Configurable:** Highly configurable behavior through a central `config.yaml` file.
*   **Bilingual Support:** Primarily designed for Arabic-speaking beneficiaries, with English support.
*   **Robust Orchestration:** A `main.py` script manages the initialization and lifecycle of all application components.

## 🛠️ Project Structure

bcfhd_complaint_bot/
├── app/
│ ├── bot/
│ │ ├── bot_core_logic.py
│ │ └── bot_telegram_handlers.py
│ ├── core/
│ │ ├── ai_handler.py
│ │ ├── cache_manager.py
│ │ └── prompt_builder.py
│ └── config/
│ ├── config.yaml
│ ├── bcfhd_system_prompt.txt
│ ├── service_account.json (gitignored)
│ ├── gmail_credentials.json (gitignored)
│ └── token.json (generated, gitignored)
├── .env (gitignored)
├── .gitignore
├── requirements.txt
└── main.py


## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Access to Google Sheets API and Gmail API (with `service_account.json` and `gmail_credentials.json` configured)
*   API keys for LLM providers (OpenRouter, Hugging Face)
*   A Telegram Bot Token

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd bcfhd_complaint_bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up configuration files:**
    *   Place your `service_account.json` (for Google Sheets) and `gmail_credentials.json` (for Gmail OAuth) into the `app/config/` directory.
    *   Rename `default_config.yaml` (if provided as such) to `config.yaml` in `app/config/` and customize it with BCFHD-specific information, Google Sheet ID, critical email, LLM model preferences, etc.
    *   Customize `app/config/bcfhd_system_prompt.txt` if needed.

5.  **Set up environment variables:**
    *   Create a `.env` file in the project root directory (`bcfhd_complaint_bot/`).
    *   Add the following environment variables with your actual keys/tokens and paths:
        ```env
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY" # If using OpenRouter
        HF_API_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"   # If using Hugging Face

        # Paths to Google Credentials Files (these are defaults, confirm they match your setup)
        GOOGLE_SHEETS_CREDENTIALS_PATH="app/config/service_account.json"
        GMAIL_API_CLIENT_SECRET_PATH="app/config/gmail_credentials.json"
        GMAIL_API_TOKEN_PATH="app/config/token.json" 
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file.

### Running the Bot

1.  **Initial Gmail API Authorization (First Run Only):**
    *   When you run the bot for the first time, the Gmail API integration will likely require an OAuth 2.0 authorization flow. This usually involves opening a URL in your browser, logging into the Google account that will send emails, and granting permissions. A `token.json` file will be created in `app/config/` upon successful authorization.

2.  **Start the bot:**
    ```bash
    python main.py
    ```

## 🔧 Configuration

The main behavior of the bot and its components is configured through `app/config/config.yaml`. Key sections include:
*   `institution`: BCFHD-specific details.
*   `bcfhd_bot_settings`: Google Sheet ID, critical email, sheet names.
*   `ai_models`: LLM provider settings, model preferences, API keys (via env vars).
*   `cache`: Caching behavior and storage.
*   `prompts`: System prompt template file and other prompt-related settings.
*   `logging`: Logging levels and output.

Refer to the comments within `config.yaml` for detailed explanations of each parameter.

## 📄 License

This project is licensed under the terms specified in the LICENSE file. Please refer to it for more details. *(See section below for LICENSE file content)*

## 🤝 Contributing

[Optional: Add guidelines for contributing if this were an open project, or state that contributions are not open at this time.]

## 📧 Contact

For inquiries related to BCFHD, please contact: [BCFHD's General Contact Email, e.g., bena@bcfhd.org]
For technical inquiries regarding this bot application: [Your contact or relevant technical contact]

---