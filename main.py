#!/usr/bin/env python3
"""
BCFHD Complaint Management Bot - Main Orchestration Script

This module serves as the central entry point for the Bena Charity for Human Development (BCFHD)
Telegram bot application. It coordinates the initialization and lifecycle management of all
core service modules and the main bot logic.

Author: Generated for BCFHD
License: Proprietary
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from yaml.loader import SafeLoader

# Import core modules
from app.core.ai_handler import AIHandler
from app.core.cache_manager import CacheManager
from app.core.prompt_builder import PromptBuilder
from app.bot.bcfhd_bot import BCFHDBot


# Global variables for cleanup
ai_handler_instance: Optional[AIHandler] = None
cache_manager_instance: Optional[CacheManager] = None
bcfhd_bot_instance: Optional[BCFHDBot] = None


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on the configuration file.
    
    Args:
        config: Global configuration dictionary
    """
    logging_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file_path = logging_config.get('log_file_path', 'logs/bcfhd_bot.log')
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging level
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = []
    
    # File handler with rotation
    if log_file_path:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=logging_config.get('max_file_size_mb', 10) * 1024 * 1024,
            backupCount=logging_config.get('backup_count', 5)
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Console handler
    if logging_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)


def load_configuration() -> Dict[str, Any]:
    """
    Load and validate the configuration file.
    
    Returns:
        Dict containing the loaded configuration
        
    Raises:
        SystemExit: If configuration loading fails
    """
    # Define path to configuration file
    APP_ROOT = Path(__file__).parent.absolute()
    config_path = APP_ROOT / "app" / "config" / "config.yaml"
    
    try:
        logging.info(f"Loading configuration from: {config_path}")
        
        if not config_path.exists():
            logging.error(f"Configuration file not found at: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.load(config_file, Loader=SafeLoader)
            
        if not config:
            logging.error("Configuration file is empty or invalid")
            sys.exit(1)
            
        # Validate required sections
        required_sections = ['institution', 'bcfhd_bot_settings', 'ai_models', 'cache', 'prompts']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logging.error(f"Missing required configuration sections: {missing_sections}")
            sys.exit(1)
            
        logging.info("Configuration loaded successfully")
        return config
        
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        sys.exit(1)


async def initialize_cache_manager(config: Dict[str, Any]) -> Optional[CacheManager]:
    """
    Initialize the CacheManager if caching is enabled.
    
    Args:
        config: Global configuration dictionary
        
    Returns:
        CacheManager instance or None if caching is disabled
    """
    cache_config = config.get('cache', {})
    
    if not cache_config.get('enabled', True):
        logging.info("Caching is disabled in configuration")
        return None
    
    try:
        # Resolve cache directory relative to project root
        APP_ROOT = Path(__file__).parent.absolute()
        cache_dir = APP_ROOT / cache_config.get('cache_dir', 'app_cache')
        
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initializing CacheManager with directory: {cache_dir}")
        
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            config=cache_config,
            loop=asyncio.get_event_loop()
        )
        
        logging.info("CacheManager initialized successfully")
        return cache_manager
        
    except Exception as e:
        logging.error(f"Failed to initialize CacheManager: {e}")
        logging.warning("Continuing without caching support")
        return None


async def initialize_prompt_builder(config: Dict[str, Any]) -> PromptBuilder:
    """
    Initialize the PromptBuilder.
    
    Args:
        config: Global configuration dictionary
        
    Returns:
        PromptBuilder instance
        
    Raises:
        SystemExit: If PromptBuilder initialization fails
    """
    try:
        # Define config directory
        APP_ROOT = Path(__file__).parent.absolute()
        config_dir = APP_ROOT / "app" / "config"
        
        # Extract required configuration sections
        institution_data = config.get('institution', {})
        templates = config.get('prompts', {})
        
        logging.info(f"Initializing PromptBuilder with config directory: {config_dir}")
        
        # Check if system prompt template exists
        template_file = templates.get('system_template_file', 'bcfhd_system_prompt.txt')
        template_path = config_dir / template_file
        
        if not template_path.exists():
            logging.error(f"System prompt template not found at: {template_path}")
            sys.exit(1)
        
        prompt_builder = PromptBuilder(
            config_dir=config_dir,
            institution_data=institution_data,
            templates=templates
        )
        
        logging.info("PromptBuilder initialized successfully")
        return prompt_builder
        
    except Exception as e:
        logging.error(f"Failed to initialize PromptBuilder: {e}")
        sys.exit(1)


async def initialize_ai_handler(config: Dict[str, Any], cache_manager: Optional[CacheManager]) -> AIHandler:
    """
    Initialize the AIHandler.
    
    Args:
        config: Global configuration dictionary
        cache_manager: CacheManager instance or None
        
    Returns:
        AIHandler instance
        
    Raises:
        SystemExit: If AIHandler initialization fails
    """
    try:
        ai_config = config.get('ai_models', {})
        
        logging.info("Initializing AIHandler")
        
        ai_handler = AIHandler(
            config=ai_config,
            cache_manager=cache_manager
        )
        
        # Ensure the HTTP session is ready
        await ai_handler._ensure_session()
        
        logging.info("AIHandler initialized successfully")
        return ai_handler
        
    except Exception as e:
        logging.error(f"Failed to initialize AIHandler: {e}")
        sys.exit(1)


async def initialize_bcfhd_bot(
    config: Dict[str, Any],
    ai_handler: AIHandler,
    cache_manager: Optional[CacheManager],
    prompt_builder: PromptBuilder
) -> BCFHDBot:
    """
    Initialize the BCFHDBot.
    
    Args:
        config: Global configuration dictionary
        ai_handler: AIHandler instance
        cache_manager: CacheManager instance or None
        prompt_builder: PromptBuilder instance
        
    Returns:
        BCFHDBot instance
        
    Raises:
        SystemExit: If BCFHDBot initialization fails
    """
    try:
        # Get Telegram bot token from environment
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not telegram_token:
            logging.error("TELEGRAM_BOT_TOKEN environment variable is not set")
            sys.exit(1)
        
        logging.info("Initializing BCFHDBot")
        
        bcfhd_bot = BCFHDBot(
            config=config,
            ai_handler=ai_handler,
            cache_manager=cache_manager,
            prompt_builder=prompt_builder,
            telegram_token=telegram_token
        )
        
        logging.info("BCFHDBot initialized successfully")
        return bcfhd_bot
        
    except Exception as e:
        logging.error(f"Failed to initialize BCFHDBot: {e}")
        sys.exit(1)


async def cleanup_resources():
    """
    Gracefully clean up all resources.
    """
    global ai_handler_instance, cache_manager_instance, bcfhd_bot_instance
    
    logging.info("Starting resource cleanup...")
    
    # Cleanup AIHandler
    if ai_handler_instance:
        try:
            await ai_handler_instance.cleanup()
            logging.info("AIHandler cleanup completed")
        except Exception as e:
            logging.error(f"Error during AIHandler cleanup: {e}")
    
    # Cleanup CacheManager
    if cache_manager_instance:
        try:
            await cache_manager_instance.cleanup()
            logging.info("CacheManager cleanup completed")
        except Exception as e:
            logging.error(f"Error during CacheManager cleanup: {e}")
    
    # Note: BCFHDBot cleanup is handled by the telegram library
    # when the application shuts down
    
    logging.info("Resource cleanup completed")


def setup_signal_handlers():
    """
    Setup signal handlers for graceful shutdown.
    """
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        # The cleanup will be handled in the finally block of run_application
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_application():
    """
    Main application runner that orchestrates all components.
    """
    global ai_handler_instance, cache_manager_instance, bcfhd_bot_instance
    
    config = None
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Setup logging with loaded configuration
        setup_logging(config)
        
        logging.info("=" * 60)
        logging.info("BCFHD Complaint Management Bot Starting Up")
        logging.info("=" * 60)
        
        # Initialize CacheManager
        logging.info("Step 1/4: Initializing CacheManager...")
        cache_manager_instance = await initialize_cache_manager(config)
        
        # Initialize PromptBuilder
        logging.info("Step 2/4: Initializing PromptBuilder...")
        prompt_builder = await initialize_prompt_builder(config)
        
        # Initialize AIHandler
        logging.info("Step 3/4: Initializing AIHandler...")
        ai_handler_instance = await initialize_ai_handler(config, cache_manager_instance)
        
        # Initialize BCFHDBot
        logging.info("Step 4/4: Initializing BCFHDBot...")
        bcfhd_bot_instance = await initialize_bcfhd_bot(
            config,
            ai_handler_instance,
            cache_manager_instance,
            prompt_builder
        )
        
        logging.info("=" * 60)
        logging.info("All components initialized successfully!")
        logging.info("Starting BCFHD Telegram Bot...")
        logging.info("=" * 60)
        
        # Run the bot application
        await bcfhd_bot_instance.run()
        
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down gracefully...")
        
    except Exception as e:
        logging.error(f"Unhandled exception in main application: {e}", exc_info=True)
        
    finally:
        # Ensure cleanup is always performed
        await cleanup_resources()
        logging.info("BCFHD Bot shutdown completed")


def main():
    """
    Main entry point for the application.
    """
    # Setup basic logging before configuration is loaded
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup signal handlers
    setup_signal_handlers()
    
    try:
        # Run the main application
        asyncio.run(run_application())
        
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()