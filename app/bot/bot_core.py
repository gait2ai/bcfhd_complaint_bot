"""
BCFHD Telegram Bot Core Logic
Contains the primary BCFHDBot class for complaint management system.
"""

import os
import logging
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pytz
import yaml

# Google libraries
import gspread
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Telegram libraries
from telegram import Update
from telegram.ext import Application, ContextTypes

# Core modules
from app.core.ai_handler import AIHandler
from app.core.cache_manager import CacheManager
from app.core.prompt_builder import PromptBuilder


@dataclass
class ComplaintData:
    """Data structure for complaint information"""
    user_id: int
    name: str = ""
    sex: str = ""
    phone: str = ""
    residence_status: str = ""
    governorate: str = ""
    directorate: str = ""
    village: str = ""
    complaint_details: str = ""  # English summary
    is_critical: bool = False
    original_complaint_text: str = ""


class BCFHDBot:
    """
    Primary bot class for Bena Charity for Human Development's 
    Telegram-based complaint management system.
    """
    
    def __init__(self, config: Dict, ai_handler: AIHandler, 
                 cache_manager: Optional[CacheManager], 
                 prompt_builder: PromptBuilder, telegram_token: str):
        """Initialize BCFHDBot with required services and configuration"""
        
        # Store core dependencies
        self.config = config
        self.ai_handler = ai_handler
        self.cache_manager = cache_manager
        self.prompt_builder = prompt_builder
        self.telegram_token = telegram_token
        
        # Extract BCFHD-specific settings
        bcfhd_settings = self.config['bcfhd_bot_settings']
        self.spreadsheet_id = bcfhd_settings['google_sheet_id']
        self.critical_email = bcfhd_settings['critical_complaint_email']
        self.complaints_sheet_name = bcfhd_settings['complaints_sheet_name']
        self.beneficiary_data_sheet_name = bcfhd_settings['beneficiary_data_sheet_name']
        self.classification_key_sheet_name = bcfhd_settings['classification_key_sheet_name']
        
        # Set timezone
        self.local_tz = pytz.timezone(self.config['institution']['timezone'])
        
        # Initialize data structures
        self.user_data: Dict[int, ComplaintData] = {}
        self.complaint_classification_keys: List[Dict] = []
        
        # Initialize service clients
        self.gspread_client = None
        self.spreadsheet = None
        self.gmail_service = None
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize_internal_services(self):
        """Initialize Google Sheets, Gmail, and load classification keys"""
        
        await self._initialize_google_sheets()
        await self._initialize_gmail()
        await self._load_bcfhd_classification_keys()
    
    async def _initialize_google_sheets(self):
        """Initialize Google Sheets client and open spreadsheet"""
        try:
            # Construct path to service account credentials
            service_account_path = Path("app/config/service_account.json")
            
            if not service_account_path.exists():
                self.logger.error(f"Service account file not found: {service_account_path}")
                return
            
            # Initialize gspread client
            self.gspread_client = gspread.service_account(filename=str(service_account_path))
            
            # Open spreadsheet
            self.spreadsheet = self.gspread_client.open_by_key(self.spreadsheet_id)
            
            self.logger.info("Google Sheets client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Sheets: {e}")
    
    async def _initialize_gmail(self):
        """Initialize Gmail service for sending critical complaint emails"""
        try:
            credentials_path = Path("app/config/gmail_credentials.json")
            token_path = Path("app/config/token.json")
            
            if not credentials_path.exists():
                self.logger.error(f"Gmail credentials file not found: {credentials_path}")
                return
            
            SCOPES = ['https://www.googleapis.com/auth/gmail.send']
            creds = None
            
            # Load existing token
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            
            # If no valid credentials available, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_path), SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build Gmail service
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            
            self.logger.info("Gmail service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gmail service: {e}")
    
    async def _load_bcfhd_classification_keys(self):
        """Load complaint classification keys from Google Sheets"""
        try:
            if self.spreadsheet is None:
                self.logger.warning("Spreadsheet not initialized, cannot load classification keys")
                return
            
            # Try to get from cache first
            if self.cache_manager:
                cached_keys = await self.cache_manager.get("classification_keys")
                if cached_keys:
                    self.complaint_classification_keys = cached_keys
                    self.logger.info(f"Loaded {len(cached_keys)} classification keys from cache")
                    return
            
            # Load from sheet
            worksheet = self.spreadsheet.worksheet(self.classification_key_sheet_name)
            records = worksheet.get_all_records()
            
            self.complaint_classification_keys = records
            
            # Cache the keys
            if self.cache_manager:
                await self.cache_manager.set(
                    "classification_keys", 
                    records, 
                    ttl=3600,  # 1 hour
                    category="classification_keys"
                )
            
            self.logger.info(f"Loaded {len(records)} classification keys from Google Sheets")
            
        except Exception as e:
            self.logger.error(f"Failed to load classification keys: {e}")
            self.complaint_classification_keys = []
    
    # LLM Interaction Helper Methods
    
    async def _get_llm_response(self, task_specific_instruction: str, 
                               user_input: str, context_data: Optional[Any] = None,
                               output_format_instruction: Optional[str] = None,
                               user_language_code: str = 'ar') -> Optional[str]:
        """Generic helper to get LLM response using prompt builder"""
        try:
            system_prompt = await self.prompt_builder.build_bcfhd_task_prompt(
                task_specific_instruction=task_specific_instruction,
                user_input_text=user_input,
                context_data=context_data,
                output_format_instruction=output_format_instruction,
                user_language_code=user_language_code
            )
            
            response = await self.ai_handler.generate_response(
                user_message=user_input,
                system_prompt=system_prompt,
                context=context_data
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM response error: {e}")
            return None
    
    async def _is_critical_complaint_llm(self, text: str) -> bool:
        """Determine if complaint is critical using LLM"""
        try:
            task_instruction = """Determine if this complaint is CRITICAL or NON_CRITICAL.
            CRITICAL complaints involve: immediate danger, severe health issues, 
            urgent humanitarian needs, life-threatening situations, or severe violations of rights.
            NON_CRITICAL complaints are general feedback, suggestions, or non-urgent issues."""
            
            response = await self._get_llm_response(
                task_specific_instruction=task_instruction,
                user_input=text,
                output_format_instruction="Respond ONLY with CRITICAL or NON_CRITICAL."
            )
            
            if response:
                return response.strip().upper() == "CRITICAL"
            return False
            
        except Exception as e:
            self.logger.error(f"Critical complaint check error: {e}")
            return False
    
    async def _classify_complaint_llm(self, complaint_text: str) -> Tuple[str, str, str]:
        """Classify complaint using LLM and classification keys"""
        try:
            task_instruction = """Classify this complaint based on the provided classification keys.
            Analyze the complaint text and match it to the most appropriate type, category, and sensitivity level."""
            
            output_format = 'Respond ONLY with JSON: {"type": "TypeValue", "category": "CategoryValue", "sensitivity": "SensitivityValue"}'
            
            response = await self._get_llm_response(
                task_specific_instruction=task_instruction,
                user_input=complaint_text,
                context_data=self.complaint_classification_keys,
                output_format_instruction=output_format
            )
            
            if response:
                try:
                    result = json.loads(response.strip())
                    return (
                        result.get("type", "General"),
                        result.get("category", "Other"),
                        result.get("sensitivity", "Normal")
                    )
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse classification JSON: {response}")
            
            return ("General", "Other", "Normal")
            
        except Exception as e:
            self.logger.error(f"Complaint classification error: {e}")
            return ("General", "Other", "Normal")
    
    async def _summarize_and_translate_complaint_llm(self, arabic_text: str) -> str:
        """Summarize and translate Arabic complaint to English"""
        try:
            task_instruction = """Summarize and translate this Arabic complaint text to English.
            Provide a clear, concise summary that captures the main points and concerns."""
            
            response = await self._get_llm_response(
                task_specific_instruction=task_instruction,
                user_input=arabic_text,
                output_format_instruction="Provide ONLY the English summary."
            )
            
            return response.strip() if response else arabic_text
            
        except Exception as e:
            self.logger.error(f"Translation/summary error: {e}")
            return arabic_text
    
    async def _determine_user_intent_llm(self, text: str) -> str:
        """Determine user intent from message text"""
        try:
            task_instruction = f"""User sent: '{text}'. Analyze the intent of this message.
            Determine if this is a complaint, suggestion, contact request, or off-topic message."""
            
            output_format = "Respond ONLY with one of: COMPLAINT_INTENT, SUGGESTION_INTENT, CONTACT_INTENT, OFF_TOPIC"
            
            response = await self._get_llm_response(
                task_specific_instruction=task_instruction,
                user_input=text,
                output_format_instruction=output_format
            )
            
            if response and response.strip() in ["COMPLAINT_INTENT", "SUGGESTION_INTENT", "CONTACT_INTENT", "OFF_TOPIC"]:
                return response.strip()
            
            return "OFF_TOPIC"
            
        except Exception as e:
            self.logger.error(f"Intent determination error: {e}")
            return "OFF_TOPIC"
    
    # Python-based Helper Methods
    
    def _transliterate_yemeni_location_py(self, arabic_text: str) -> str:
        """Transliterate Yemeni Arabic location names to English script"""
        if not arabic_text or not self._is_arabic_text(arabic_text):
            return arabic_text
        
        # Basic Arabic to English transliteration mapping
        transliteration_map = {
            'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
            'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's',
            'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a',
            'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
            'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ى': 'a', 'ة': 'a',
            'أ': 'a', 'إ': 'i', 'آ': 'aa', 'ء': '', ' ': ' '
        }
        
        # Common Yemeni prefixes and locations
        common_locations = {
            'صنعاء': 'Sana\'a',
            'عدن': 'Aden',
            'تعز': 'Taiz',
            'الحديدة': 'Al-Hudaydah',
            'إب': 'Ibb',
            'ذمار': 'Dhamar',
            'مأرب': 'Marib',
            'لحج': 'Lahij',
            'أبين': 'Abyan',
            'شبوة': 'Shabwah',
            'حضرموت': 'Hadramawt',
            'المهرة': 'Al-Mahrah',
            'سقطرى': 'Soqotra'
        }
        
        # Check for exact matches first
        arabic_text_clean = arabic_text.strip()
        if arabic_text_clean in common_locations:
            return common_locations[arabic_text_clean]
        
        # Transliterate character by character
        result = ""
        for char in arabic_text:
            result += transliteration_map.get(char, char)
        
        # Clean up result
        result = re.sub(r'\s+', ' ', result.strip())
        result = result.title()  # Capitalize first letter of each word
        
        return result if result else arabic_text
    
    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        if not text:
            return False
        
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                    arabic_chars += 1
        
        return total_chars > 0 and (arabic_chars / total_chars) > 0.5
    
    # Google Sheets Interaction Methods
    
    async def _check_existing_beneficiary(self, user_id: int) -> Optional[Dict]:
        """Check if beneficiary exists in the database"""
        try:
            # Check cache first
            if self.cache_manager:
                cache_key = f"beneficiary_{user_id}"
                cached_data = await self.cache_manager.get(cache_key)
                if cached_data:
                    return cached_data
            
            if self.spreadsheet is None:
                return None
            
            worksheet = self.spreadsheet.worksheet(self.beneficiary_data_sheet_name)
            records = worksheet.get_all_records()
            
            for record in records:
                if str(record.get('user_id', '')) == str(user_id):
                    # Cache the result
                    if self.cache_manager:
                        await self.cache_manager.set(
                            f"beneficiary_{user_id}",
                            record,
                            ttl=1800,  # 30 minutes
                            category="beneficiary_profiles"
                        )
                    return record
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking existing beneficiary: {e}")
            return None
    
    async def _save_beneficiary_data(self, data: ComplaintData) -> bool:
        """Save or update beneficiary data in Google Sheets"""
        try:
            if self.spreadsheet is None:
                return False
            
            worksheet = self.spreadsheet.worksheet(self.beneficiary_data_sheet_name)
            timestamp = datetime.now(self.local_tz).strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare row data
            row_data = [
                data.user_id,
                data.name,  # Arabic name
                data.phone,  # English
                data.residence_status,  # English
                self._transliterate_yemeni_location_py(data.governorate),
                self._transliterate_yemeni_location_py(data.directorate),
                self._transliterate_yemeni_location_py(data.village),
                timestamp
            ]
            
            # Check if user exists and update or append
            existing = await self._check_existing_beneficiary(data.user_id)
            if existing:
                # Find and update existing row
                records = worksheet.get_all_records()
                for i, record in enumerate(records, start=2):  # Start from row 2 (after headers)
                    if str(record.get('user_id', '')) == str(data.user_id):
                        worksheet.update(f'A{i}:H{i}', [row_data])
                        break
            else:
                # Append new row
                worksheet.append_row(row_data)
            
            # Clear cache for this user
            if self.cache_manager:
                await self.cache_manager.set(
                    f"beneficiary_{data.user_id}",
                    None,
                    ttl=0,  # Delete
                    category="beneficiary_profiles"
                )
            
            self.logger.info(f"Beneficiary data saved for user {data.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving beneficiary data: {e}")
            return False
    
    async def _log_complaint_to_sheet(self, data: ComplaintData) -> bool:
        """Log complaint to Google Sheets with full processing"""
        try:
            if self.spreadsheet is None:
                return False
            
            # Get classification
            complaint_type, category, sensitivity = await self._classify_complaint_llm(
                data.original_complaint_text
            )
            
            # Get English summary if needed
            if self._is_arabic_text(data.original_complaint_text):
                data.complaint_details = await self._summarize_and_translate_complaint_llm(
                    data.original_complaint_text
                )
            else:
                data.complaint_details = data.original_complaint_text
            
            # Prepare complaint row data (23 columns as specified)
            timestamp = datetime.now(self.local_tz)
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            row_data = [
                data.user_id,                                                    # 1
                data.name,                                                       # 2
                data.sex,                                                        # 3
                data.phone,                                                      # 4
                data.residence_status,                                           # 5
                self._transliterate_yemeni_location_py(data.governorate),       # 6
                self._transliterate_yemeni_location_py(data.directorate),       # 7
                self._transliterate_yemeni_location_py(data.village),           # 8
                data.original_complaint_text,                                    # 9
                data.complaint_details,                                          # 10 (English summary)
                complaint_type,                                                  # 11
                category,                                                        # 12
                sensitivity,                                                     # 13
                "CRITICAL" if data.is_critical else "NON_CRITICAL",            # 14
                "PENDING",                                                       # 15 (Status)
                "",                                                             # 16 (Assigned to)
                "",                                                             # 17 (Resolution notes)
                formatted_timestamp,                                            # 18 (Submitted at)
                "",                                                             # 19 (Updated at)
                "",                                                             # 20 (Resolved at)
                "TELEGRAM",                                                     # 21 (Source)
                "",                                                             # 22 (Internal notes)
                ""                                                              # 23 (Follow-up required)
            ]
            
            # Log to complaints sheet
            worksheet = self.spreadsheet.worksheet(self.complaints_sheet_name)
            worksheet.append_row(row_data)
            
            # Save beneficiary data
            await self._save_beneficiary_data(data)
            
            self.logger.info(f"Complaint logged for user {data.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging complaint to sheet: {e}")
            return False
    
    # Gmail Method
    
    async def _send_critical_complaint_email(self, data: ComplaintData) -> bool:
        """Send email notification for critical complaints"""
        try:
            if self.gmail_service is None:
                self.logger.warning("Gmail service not initialized")
                return False
            
            timestamp = datetime.now(self.local_tz).strftime("%Y-%m-%d %H:%M:%S")
            
            # Construct email content
            subject = f"🚨 CRITICAL COMPLAINT - BCFHD Bot Alert"
            
            body = f"""
CRITICAL COMPLAINT ALERT

Name: {data.name}
Phone: {data.phone}
User ID: {data.user_id}
Location: {data.governorate}, {data.directorate}, {data.village}
Timestamp: {timestamp}

ORIGINAL COMPLAINT:
{data.original_complaint_text}

ENGLISH SUMMARY:
{data.complaint_details}

This complaint has been automatically flagged as CRITICAL and requires immediate attention.

---
BCFHD Telegram Bot System
            """.strip()
            
            # Create email message
            import base64
            from email.mime.text import MIMEText
            
            message = MIMEText(body)
            message['to'] = self.critical_email
            message['subject'] = subject
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send email
            self.gmail_service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            self.logger.info(f"Critical complaint email sent for user {data.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending critical complaint email: {e}")
            return False
    
    async def run(self):
        """Main run method to start the Telegram bot"""
        try:
            # Create Telegram application
            application = Application.builder().token(self.telegram_token).build()
            
            # Set post-init to initialize internal services
            application.post_init = self.initialize_internal_services
            
            # Import and setup Telegram handlers
            from app.bot.bot_telegram_handlers import setup_telegram_handlers
            setup_telegram_handlers(application, self)
            
            self.logger.info("BCFHD Telegram Bot starting polling...")
            
            # Start polling
            await application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            raise
