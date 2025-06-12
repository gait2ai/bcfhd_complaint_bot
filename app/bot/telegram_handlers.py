"""
Telegram handlers module for BCFHD complaint management bot.
Handles all Telegram-specific interactions and conversation flows.
"""

import logging
from typing import Optional, Dict, Any

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ContextTypes, ConversationHandler
)

from app.bot.bot_core_logic import BCFHDBot, ComplaintData

# State constants for ConversationHandler
(START_COMPLAINT_FLOW, CONFIRM_EXISTING, COLLECTING_NAME, COLLECTING_SEX, 
 COLLECTING_PHONE, COLLECTING_RESIDENCE, COLLECTING_GOVERNORATE, 
 COLLECTING_DIRECTORATE, COLLECTING_VILLAGE, COLLECTING_COMPLAINT, 
 CONFIRMING_SUBMISSION, CRITICAL_NAME, CRITICAL_PHONE) = range(13)

# Logger for this module
logger = logging.getLogger(__name__)

# Bilingual messages
MESSAGES = {
    'ar': {
        'welcome': "مرحباً بك في بوت جمعية بينة للتنمية البشرية\nيمكنك تقديم شكوى أو اقتراح. استخدم /complaint لتقديم شكوى أو /suggestion لتقديم اقتراح.",
        'welcome_existing': "مرحباً بك مرة أخرى! بياناتك محفوظة لدينا.",
        'use_existing_data': "هل تريد استخدام بياناتك المحفوظة؟",
        'enter_name': "الرجاء إدخال اسمك الكامل:",
        'enter_sex': "الرجاء تحديد جنسك:",
        'enter_phone': "الرجاء إدخال رقم هاتفك:",
        'enter_residence': "الرجاء تحديد وضعك السكني:",
        'enter_governorate': "الرجاء إدخال المحافظة:",
        'enter_directorate': "الرجاء إدخال المديرية:",
        'enter_village': "الرجاء إدخال القرية/المنطقة:",
        'enter_complaint': "الرجاء إدخال تفاصيل شكواك:",
        'confirm_submission': "هل البيانات صحيحة؟",
        'critical_intro': "تم تصنيف حالتك كحالة حرجة. سيتم التعامل معها بأولوية عالية.",
        'critical_name': "الرجاء إدخال اسمك للحالة الحرجة:",
        'critical_phone': "الرجاء إدخال رقم هاتفك للحالة الحرجة:",
        'critical_registered': "تم تسجيل الحالة الحرجة. سيتصل بك مسؤول قريباً.",
        'complaint_success': "تم تسجيل شكواك بنجاح. شكراً لك.",
        'suggestion_success': "تم تسجيل اقتراحك بنجاح. شكراً لك.",
        'restart': "لنبدأ من جديد. الرجاء إدخال اسمك الكامل:",
        'cancelled': "تم إلغاء العملية.",
        'error': "حدث خطأ. الرجاء المحاولة مرة أخرى.",
        'help': "هذا البوت لتسجيل الشكاوى والاقتراحات لجمعية بينة للتنمية البشرية.\n\nالأوامر المتاحة:\n/complaint - تقديم شكوى\n/suggestion - تقديم اقتراح\n/contact - معلومات التواصل\n/help - المساعدة\n/cancel - إلغاء العملية الحالية",
        'suggestion_ack': "شكراً لاقتراحك. سيتم دراسته والاستفادة منه.",
        'contact_info': "للتواصل مع جمعية بينة للتنمية البشرية:\nالهاتف: [رقم الهاتف]\nالبريد الإلكتروني: [البريد الإلكتروني]\nالعنوان: [العنوان]",
        'complaint_intent': "يبدو أنك تريد تقديم شكوى. استخدم الأمر /complaint",
        'suggestion_intent': "يبدو أنك تريد تقديم اقتراح. استخدم الأمر /suggestion",
        'off_topic': "هذا البوت مخصص لتسجيل الشكاوى والاقتراحات فقط. استخدم /help للمساعدة.",
        'invalid_name': "الرجاء إدخال اسم صحيح (3 كلمات على الأقل).",
        'invalid_phone': "الرجاء إدخال رقم هاتف صحيح.",
        'yes': 'نعم',
        'no': 'لا',
        'male': 'ذكر',
        'female': 'أنثى',
        'resident': 'مقيم',
        'idp': 'نازح',
        'returnee': 'عائد',
        'residence_explanation': "مقيم: تعيش في منطقتك الأصلية\nنازح: انتقلت من منطقة لأخرى\nعائد: عدت إلى منطقتك بعد نزوح"
    },
    'en': {
        'welcome': "Welcome to Bena Charity for Human Development bot\nYou can submit a complaint or suggestion. Use /complaint to submit a complaint or /suggestion to submit a suggestion.",
        'welcome_existing': "Welcome back! Your data is saved with us.",
        'use_existing_data': "Would you like to use your saved data?",
        'enter_name': "Please enter your full name:",
        'enter_sex': "Please select your gender:",
        'enter_phone': "Please enter your phone number:",
        'enter_residence': "Please select your residence status:",
        'enter_governorate': "Please enter your governorate:",
        'enter_directorate': "Please enter your directorate:",
        'enter_village': "Please enter your village/area:",
        'enter_complaint': "Please enter your complaint details:",
        'confirm_submission': "Is the information correct?",
        'critical_intro': "Your case has been classified as critical. It will be handled with high priority.",
        'critical_name': "Please enter your name for the critical case:",
        'critical_phone': "Please enter your phone number for the critical case:",
        'critical_registered': "Critical case has been registered. An officer will contact you soon.",
        'complaint_success': "Your complaint has been registered successfully. Thank you.",
        'suggestion_success': "Your suggestion has been registered successfully. Thank you.",
        'restart': "Let's start over. Please enter your full name:",
        'cancelled': "Operation cancelled.",
        'error': "An error occurred. Please try again.",
        'help': "This bot is for registering complaints and suggestions for Bena Charity for Human Development.\n\nAvailable commands:\n/complaint - Submit a complaint\n/suggestion - Submit a suggestion\n/contact - Contact information\n/help - Help\n/cancel - Cancel current operation",
        'suggestion_ack': "Thank you for your suggestion. It will be reviewed and considered.",
        'contact_info': "To contact Bena Charity for Human Development:\nPhone: [Phone Number]\nEmail: [Email Address]\nAddress: [Address]",
        'complaint_intent': "It seems you want to submit a complaint. Use the /complaint command",
        'suggestion_intent': "It seems you want to submit a suggestion. Use the /suggestion command",
        'off_topic': "This bot is only for registering complaints and suggestions. Use /help for assistance.",
        'invalid_name': "Please enter a valid name (at least 3 words).",
        'invalid_phone': "Please enter a valid phone number.",
        'yes': 'Yes',
        'no': 'No',
        'male': 'Male',
        'female': 'Female',
        'resident': 'Resident',
        'idp': 'IDP',
        'returnee': 'Returnee',
        'residence_explanation': "Resident: You live in your original area\nIDP: You moved from one area to another\nReturnee: You returned to your area after displacement"
    }
}

def get_message(text: str, is_arabic: bool = True) -> str:
    """Get message in appropriate language."""
    lang = 'ar' if is_arabic else 'en'
    return MESSAGES[lang].get(text, text)

def get_yes_no_keyboard(is_arabic: bool = True) -> ReplyKeyboardMarkup:
    """Get Yes/No keyboard in appropriate language."""
    lang = 'ar' if is_arabic else 'en'
    keyboard = [[MESSAGES[lang]['yes'], MESSAGES[lang]['no']]]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

def get_sex_keyboard(is_arabic: bool = True) -> ReplyKeyboardMarkup:
    """Get Male/Female keyboard in appropriate language."""
    lang = 'ar' if is_arabic else 'en'
    keyboard = [[MESSAGES[lang]['male'], MESSAGES[lang]['female']]]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

def get_residence_keyboard(is_arabic: bool = True) -> ReplyKeyboardMarkup:
    """Get residence status keyboard in appropriate language."""
    lang = 'ar' if is_arabic else 'en'
    keyboard = [[MESSAGES[lang]['resident'], MESSAGES[lang]['idp'], MESSAGES[lang]['returnee']]]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

async def start_complaint_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Start the complaint collection flow."""
    user_id = update.effective_user.id
    
    # Initialize user data
    bot_instance.user_data[user_id] = ComplaintData(user_id=user_id)
    
    # Check if this is a critical complaint
    is_critical = await bot_instance._is_critical_complaint_llm(update.message.text)
    
    if is_critical:
        bot_instance.user_data[user_id].is_critical = True
        bot_instance.user_data[user_id].original_complaint_text = update.message.text
        
        is_arabic = bot_instance._is_arabic_text(update.message.text)
        await update.message.reply_text(
            get_message('critical_intro', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
        await update.message.reply_text(get_message('critical_name', is_arabic))
        return CRITICAL_NAME
    else:
        # Check for existing beneficiary data
        existing_data = await bot_instance._check_existing_beneficiary(user_id)
        
        if existing_data:
            # Store existing data
            complaint_data = bot_instance.user_data[user_id]
            complaint_data.name = existing_data.get('name', '')
            complaint_data.sex = existing_data.get('sex', '')
            complaint_data.phone = existing_data.get('phone', '')
            complaint_data.residence_status = existing_data.get('residence_status', '')
            complaint_data.governorate = existing_data.get('governorate', '')
            complaint_data.directorate = existing_data.get('directorate', '')
            complaint_data.village = existing_data.get('village', '')
            
            is_arabic = bot_instance._is_arabic_text(update.message.text)
            await update.message.reply_text(
                get_message('use_existing_data', is_arabic),
                reply_markup=get_yes_no_keyboard(is_arabic)
            )
            return CONFIRM_EXISTING
        else:
            # New user, non-critical
            is_arabic = bot_instance._is_arabic_text(update.message.text)
            await update.message.reply_text(
                get_message('enter_name', is_arabic),
                reply_markup=ReplyKeyboardRemove()
            )
            return COLLECTING_NAME

async def confirm_existing_data(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Confirm whether to use existing data."""
    user_id = update.effective_user.id
    response = update.message.text.lower().strip()
    
    is_arabic = bot_instance._is_arabic_text(response)
    yes_text = get_message('yes', is_arabic).lower()
    
    if response == yes_text or response in ['yes', 'نعم']:
        await update.message.reply_text(
            get_message('enter_complaint', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
        return COLLECTING_COMPLAINT
    else:
        await update.message.reply_text(
            get_message('enter_name', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
        return COLLECTING_NAME

async def collect_name(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's full name."""
    user_id = update.effective_user.id
    name = update.message.text.strip()
    
    # Basic validation - at least 3 words
    if len(name.split()) < 3:
        is_arabic = bot_instance._is_arabic_text(name)
        await update.message.reply_text(get_message('invalid_name', is_arabic))
        return COLLECTING_NAME
    
    bot_instance.user_data[user_id].name = name
    
    is_arabic = bot_instance._is_arabic_text(name)
    await update.message.reply_text(
        get_message('enter_sex', is_arabic),
        reply_markup=get_sex_keyboard(is_arabic)
    )
    return COLLECTING_SEX

async def collect_sex(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's gender."""
    user_id = update.effective_user.id
    sex_input = update.message.text.strip()
    
    # Normalize gender input
    if sex_input.lower() in ['male', 'ذكر', 'm']:
        sex = 'Male'
    elif sex_input.lower() in ['female', 'أنثى', 'f']:
        sex = 'Female'
    else:
        is_arabic = bot_instance._is_arabic_text(sex_input)
        await update.message.reply_text(
            get_message('enter_sex', is_arabic),
            reply_markup=get_sex_keyboard(is_arabic)
        )
        return COLLECTING_SEX
    
    bot_instance.user_data[user_id].sex = sex
    
    is_arabic = bot_instance._is_arabic_text(sex_input)
    await update.message.reply_text(
        get_message('enter_phone', is_arabic),
        reply_markup=ReplyKeyboardRemove()
    )
    return COLLECTING_PHONE

async def collect_phone(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's phone number."""
    user_id = update.effective_user.id
    phone = update.message.text.strip()
    
    # Basic phone validation
    if len(phone) < 8 or not any(char.isdigit() for char in phone):
        is_arabic = bot_instance._is_arabic_text(phone)
        await update.message.reply_text(get_message('invalid_phone', is_arabic))
        return COLLECTING_PHONE
    
    bot_instance.user_data[user_id].phone = phone
    
    is_arabic = bot_instance._is_arabic_text(phone)
    residence_msg = f"{get_message('enter_residence', is_arabic)}\n\n{get_message('residence_explanation', is_arabic)}"
    await update.message.reply_text(
        residence_msg,
        reply_markup=get_residence_keyboard(is_arabic)
    )
    return COLLECTING_RESIDENCE

async def collect_residence(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's residence status."""
    user_id = update.effective_user.id
    residence_input = update.message.text.strip()
    
    # Normalize residence input
    if residence_input.lower() in ['resident', 'مقيم']:
        residence = 'Resident'
    elif residence_input.lower() in ['idp', 'نازح']:
        residence = 'IDP'
    elif residence_input.lower() in ['returnee', 'عائد']:
        residence = 'Returnee'
    else:
        is_arabic = bot_instance._is_arabic_text(residence_input)
        await update.message.reply_text(
            get_message('enter_residence', is_arabic),
            reply_markup=get_residence_keyboard(is_arabic)
        )
        return COLLECTING_RESIDENCE
    
    bot_instance.user_data[user_id].residence_status = residence
    
    is_arabic = bot_instance._is_arabic_text(residence_input)
    await update.message.reply_text(
        get_message('enter_governorate', is_arabic),
        reply_markup=ReplyKeyboardRemove()
    )
    return COLLECTING_GOVERNORATE

async def collect_governorate(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's governorate."""
    user_id = update.effective_user.id
    governorate = update.message.text.strip()
    
    bot_instance.user_data[user_id].governorate = governorate
    
    is_arabic = bot_instance._is_arabic_text(governorate)
    await update.message.reply_text(get_message('enter_directorate', is_arabic))
    return COLLECTING_DIRECTORATE

async def collect_directorate(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's directorate."""
    user_id = update.effective_user.id
    directorate = update.message.text.strip()
    
    bot_instance.user_data[user_id].directorate = directorate
    
    is_arabic = bot_instance._is_arabic_text(directorate)
    await update.message.reply_text(get_message('enter_village', is_arabic))
    return COLLECTING_VILLAGE

async def collect_village(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect user's village."""
    user_id = update.effective_user.id
    village = update.message.text.strip()
    
    bot_instance.user_data[user_id].village = village
    
    is_arabic = bot_instance._is_arabic_text(village)
    await update.message.reply_text(get_message('enter_complaint', is_arabic))
    return COLLECTING_COMPLAINT

async def collect_complaint(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect complaint details and prepare for confirmation."""
    user_id = update.effective_user.id
    complaint_data = bot_instance.user_data[user_id]
    
    complaint_data.original_complaint_text = update.message.text
    
    # Summarize and translate complaint
    complaint_data.complaint_details = await bot_instance._summarize_and_translate_complaint_llm(
        complaint_data.original_complaint_text
    )
    
    # Create confirmation message with all collected data
    is_arabic = bot_instance._is_arabic_text(complaint_data.original_complaint_text)
    
    if is_arabic:
        confirmation_msg = f"""الرجاء مراجعة بياناتك:

الاسم: {complaint_data.name}
الجنس: {complaint_data.sex}
الهاتف: {complaint_data.phone}
الوضع السكني: {complaint_data.residence_status}
المحافظة: {complaint_data.governorate}
المديرية: {complaint_data.directorate}
القرية/المنطقة: {complaint_data.village}
الشكوى: {complaint_data.original_complaint_text}"""
    else:
        confirmation_msg = f"""Please review your information:

Name: {complaint_data.name}
Gender: {complaint_data.sex}
Phone: {complaint_data.phone}
Residence Status: {complaint_data.residence_status}
Governorate: {complaint_data.governorate}
Directorate: {complaint_data.directorate}
Village/Area: {complaint_data.village}
Complaint: {complaint_data.original_complaint_text}"""
    
    await update.message.reply_text(confirmation_msg)
    await update.message.reply_text(
        get_message('confirm_submission', is_arabic),
        reply_markup=get_yes_no_keyboard(is_arabic)
    )
    
    return CONFIRMING_SUBMISSION

async def confirm_submission(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Confirm and submit the complaint."""
    user_id = update.effective_user.id
    complaint_data = bot_instance.user_data[user_id]
    response = update.message.text.lower().strip()
    
    is_arabic = bot_instance._is_arabic_text(response)
    yes_text = get_message('yes', is_arabic).lower()
    
    if response == yes_text or response in ['yes', 'نعم']:
        try:
            # Classify complaint
            complaint_type, category, sensitivity = await bot_instance._classify_complaint_llm(
                complaint_data.original_complaint_text
            )
            
            # Log complaint to sheet
            success = await bot_instance._log_complaint_to_sheet(
                complaint_data, 
                classification_results=(complaint_type, category, sensitivity)
            )
            
            if success:
                # Determine if this is a complaint or suggestion based on classification
                if 'suggestion' in complaint_type.lower() or 'اقتراح' in complaint_type:
                    success_msg = get_message('suggestion_success', is_arabic)
                else:
                    success_msg = get_message('complaint_success', is_arabic)
                
                await update.message.reply_text(
                    success_msg,
                    reply_markup=ReplyKeyboardRemove()
                )
            else:
                await update.message.reply_text(
                    get_message('error', is_arabic),
                    reply_markup=ReplyKeyboardRemove()
                )
        except Exception as e:
            bot_instance.logger.error(f"Error submitting complaint: {e}")
            await update.message.reply_text(
                get_message('error', is_arabic),
                reply_markup=ReplyKeyboardRemove()
            )
        
        # Clean up user data
        del bot_instance.user_data[user_id]
        return ConversationHandler.END
    else:
        # Restart data collection
        await update.message.reply_text(
            get_message('restart', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
        return COLLECTING_NAME

async def collect_critical_name(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect name for critical complaint."""
    user_id = update.effective_user.id
    name = update.message.text.strip()
    
    bot_instance.user_data[user_id].name = name
    
    is_arabic = bot_instance._is_arabic_text(name)
    await update.message.reply_text(get_message('critical_phone', is_arabic))
    return CRITICAL_PHONE

async def collect_critical_phone(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Collect phone for critical complaint and finalize."""
    user_id = update.effective_user.id
    phone = update.message.text.strip()
    
    complaint_data = bot_instance.user_data[user_id]
    complaint_data.phone = phone
    
    try:
        # Log critical complaint
        log_success = await bot_instance._log_complaint_to_sheet(complaint_data)
        
        # Send critical complaint email
        email_success = await bot_instance._send_critical_complaint_email(complaint_data)
        
        is_arabic = bot_instance._is_arabic_text(phone)
        await update.message.reply_text(
            get_message('critical_registered', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        bot_instance.logger.error(f"Error handling critical complaint: {e}")
        is_arabic = bot_instance._is_arabic_text(phone)
        await update.message.reply_text(
            get_message('error', is_arabic),
            reply_markup=ReplyKeyboardRemove()
        )
    
    # Clean up user data
    del bot_instance.user_data[user_id]
    return ConversationHandler.END

# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle /start command."""
    user_id = update.effective_user.id
    
    # Check for existing beneficiary
    existing_data = await bot_instance._check_existing_beneficiary(user_id)
    
    if existing_data:
        # Arabic message for returning users
        welcome_msg = f"{get_message('welcome_existing', True)}\n\n{get_message('welcome', True)}"
    else:
        # Default Arabic welcome message
        welcome_msg = get_message('welcome', True)
    
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle /help command."""
    # Default to Arabic for help
    await update.message.reply_text(get_message('help', True))

async def suggestion_command(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle /suggestion command."""
    # Default to Arabic
    await update.message.reply_text(get_message('suggestion_ack', True))
    return ConversationHandler.END

async def contact_command(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle /contact command."""
    # Default to Arabic
    await update.message.reply_text(get_message('contact_info', True))
    return ConversationHandler.END

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot) -> int:
    """Cancel the current conversation."""
    user_id = update.effective_user.id
    
    # Clean up user data
    bot_instance.user_data.pop(user_id, None)
    
    # Default to Arabic for cancel message
    await update.message.reply_text(
        get_message('cancelled', True),
        reply_markup=ReplyKeyboardRemove()
    )
    
    return ConversationHandler.END

# General message handler
async def handle_general_message(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle general messages outside of conversations."""
    user_id = update.effective_user.id
    
    # Simple check if user is in active conversation
    # This could be enhanced with proper conversation state tracking
    if user_id not in bot_instance.user_data:
        try:
            intent = await bot_instance._determine_user_intent_llm(update.message.text)
            is_arabic = bot_instance._is_arabic_text(update.message.text)
            
            if intent == "COMPLAINT_INTENT":
                await update.message.reply_text(get_message('complaint_intent', is_arabic))
            elif intent == "SUGGESTION_INTENT":
                await update.message.reply_text(get_message('suggestion_intent', is_arabic))
            else:
                await update.message.reply_text(get_message('off_topic', is_arabic))
        except Exception as e:
            bot_instance.logger.error(f"Error determining user intent: {e}")
            is_arabic = bot_instance._is_arabic_text(update.message.text)
            await update.message.reply_text(get_message('off_topic', is_arabic))

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_instance: BCFHDBot):
    """Handle errors."""
    bot_instance.logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.effective_message:
        # Default to Arabic for error messages
        await update.effective_message.reply_text(get_message('error', True))

def setup_telegram_handlers(application: Application, bot_instance: BCFHDBot) -> None:
    """Set up all Telegram handlers for the bot."""
    
    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('complaint', lambda u, c: start_complaint_flow(u, c, bot_instance)),
            MessageHandler(
                filters.TEXT & ~filters.COMMAND & filters.Regex(r'.*(شكوى|complaint|مشكلة|problem).*'),
                lambda u, c: start_complaint_flow(u, c, bot_instance)
            )
        ],
        states={
            START_COMPLAINT_FLOW: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: start_complaint_flow(u, c, bot_instance))
            ],
            CONFIRM_EXISTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: confirm_existing_data(u, c, bot_instance))
            ],
            COLLECTING_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_name(u, c, bot_instance))
            ],
            COLLECTING_SEX: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_sex(u, c, bot_instance))
            ],
            COLLECTING_PHONE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_phone(u, c, bot_instance))
            ],
            COLLECTING_RESIDENCE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_residence(u, c, bot_instance))
            ],
            COLLECTING_GOVERNORATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_governorate(u, c, bot_instance))
            ],
            COLLECTING_DIRECTORATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_directorate(u, c, bot_instance))
            ],
            COLLECTING_VILLAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_village(u, c, bot_instance))
            ],
            COLLECTING_COMPLAINT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_complaint(u, c, bot_instance))
            ],
            CONFIRMING_SUBMISSION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: confirm_submission(u, c, bot_instance))
            ],
            CRITICAL_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_critical_name(u, c, bot_instance))
            ],
            CRITICAL_PHONE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: collect_critical_phone(u, c, bot_instance))
            ]
        },
        fallbacks=[
            CommandHandler('cancel', lambda u, c: cancel_conversation(u, c, bot_instance))
        ],
        allow_reentry=True
    )
    
    # Add conversation handler to application
    application.add_handler(conv_handler)
    
    # Add command handlers
    application.add_handler(CommandHandler("start", lambda u, c: start_command(u, c, bot_instance)))
    application.add_handler(CommandHandler("help", lambda u, c: help_command(u, c, bot_instance)))
    application.add_handler(CommandHandler("suggestion", lambda u, c: suggestion_command(u, c, bot_instance)))
    application.add_handler(CommandHandler("contact", lambda u, c: contact_command(u, c, bot_instance)))
    
    # Add general message handler (lower priority)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            lambda u, c: handle_general_message(u, c, bot_instance)
        ),
        group=1
    )
    
    # Add error handler
    application.add_error_handler(lambda u, c: error_handler(u, c, bot_instance))