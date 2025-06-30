import asyncio
import json
import logging
import os
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from telethon import TelegramClient, events
from telethon.tl.types import Channel, Chat, MessageMediaPhoto, MessageMediaDocument
from telethon.tl.types import DocumentAttributeVideo, DocumentAttributeAudio, DocumentAttributeFilename
import re
from ai import AIChatClient
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramChannelMonitor:
    def __init__(self, api_id: int, api_hash: str, phone: str,ai_api_key:str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.ai_client = AIChatClient(api_key=ai_api_key,system_prompt="Ты контент-креатор канала по мобильным играм и Аниме в Telegram , тебе будут приходить новости ,а ты должна их анализировать и публиковать в вовлекающем для читателя формате, прорабатывай текст форматируя его - не пиши свои мысли или действия, только новость,уберай теги и названия других каналов,пиши коротко но информативно,переводи на русский")
        self.client = TelegramClient('session', api_id, api_hash)
        self._channel_info_cache = {}
        # Хранилище для отслеживания уже обработанных сообщений
        self.processed_messages: Set[int] = set()
        
        # Конфигурация каналов (цели -> источники)
        self.channel_config: Dict[str, List[str]] = {}
        
        # Обратный маппинг для быстрого поиска (источник -> цели)
        self.source_to_targets: Dict[str, List[str]] = {}
        
        # Настройки фильтрации и обработки
        self.filters = {
            'min_length': 10,  # Минимальная длина сообщения
            'keywords_exclude': ['реклама', 'spam', 'промо'],  # Исключающие слова
            'keywords_include': [],  # Обязательные слова (если пусто - игнорируется)
            'delay_between_posts': 30,  # Задержка между постами в секундах
        }
        
        # Настройки медиа
        self.media_settings = {
            'download_media': True,  # Скачивать ли медиа
            'temp_dir': 'temp_media',  # Временная папка для медиа
            'max_file_size': 50 * 1024 * 1024,  # Максимальный размер файла (50MB)
            'allowed_extensions': ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webm', '.pdf', '.doc', '.docx'],
            'clean_temp_files': True,  # Удалять временные файлы после отправки
        }
        
        # Создаем временную директорию
        self.setup_temp_directory()
        
        # Загружаем конфигурацию
        self.load_config()

    def setup_temp_directory(self):
        """Создание временной директории для медиа файлов"""
        try:
            if not os.path.exists(self.media_settings['temp_dir']):
                os.makedirs(self.media_settings['temp_dir'])
                logger.info(f"Создана временная директория: {self.media_settings['temp_dir']}")
        except Exception as e:
            logger.error(f"Ошибка создания временной директории: {e}")
            # Используем системную временную директорию как fallback
            self.media_settings['temp_dir'] = tempfile.gettempdir()

    def load_config(self):
        """Загрузка конфигурации из файла"""
        try:
            with open('channel_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.channel_config = config.get('channels', {})
                self.filters.update(config.get('filters', {}))
                self.media_settings.update(config.get('media_settings', {}))
                
                # Создаем обратный маппинг
                self.build_source_to_targets_mapping()
                
                logger.info(f"Загружена конфигурация для {len(self.channel_config)} целевых каналов")
                logger.info(f"Отслеживается {len(self.source_to_targets)} источников")
        except FileNotFoundError:
            logger.warning("Файл конфигурации не найден. Создаю пример...")
            self.create_example_config()
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")

    def build_source_to_targets_mapping(self):
	    """Создание обратного маппинга источник -> цели с поддержкой топиков"""
	    self.source_to_targets = {}
	    
	    for target_channel_str, source_channels in self.channel_config.items():
	        target_data = self.normalize_channel_identifier(target_channel_str)
	        
	        for source_channel_str in source_channels:
	            source_data = self.normalize_channel_identifier(source_channel_str)
	            source_key = source_channel_str  # Используем исходную строку как ключ
	            
	            if source_key not in self.source_to_targets:
	                self.source_to_targets[source_key] = []
	            
	            self.source_to_targets[source_key].append({
	                "target": target_channel_str,
	                "target_data": target_data
	            })
	    
	    logger.info(f"Создан маппинг для {len(self.source_to_targets)} источников")
	    

    def create_example_config(self):
	    """Создание примера конфигурации с поддержкой топиков"""
	    example_config = {
	        "channels": {
	            "@target_channel_1": [
	                "@source_channel_1", 
	                "@source_channel_2",
	                {"channel": "@source_channel_3", "topic_id": 123},  # Источник с топиком
	                "https://t.me/source_channel_4"
	            ],
	            # Целевой канал с топиком
	            {"channel": "@target_channel_2", "topic_id": 456}: [
	                "@source_channel_1", 
	                {"channel": "@source_channel_5", "topic_id": 789}
	            ]
	        },
	        "filters": {
	            "min_length": 10,
	            "keywords_exclude": ["реклама", "spam", "промо"],
	            "keywords_include": [],
	            "delay_between_posts": 30
	        },
	        "media_settings": {
	            "download_media": True,
	            "temp_dir": "temp_media",
	            "max_file_size": 52428800,
	            "allowed_extensions": [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".webm", ".pdf", ".doc", ".docx"],
	            "clean_temp_files": True
	        }
	    }
	    
	    with open('channel_config.json', 'w', encoding='utf-8') as f:
	        json.dump(example_config, f, ensure_ascii=False, indent=2)
	    
	    logger.info("Создан файл channel_config.json с примером конфигурации")

    async def start(self):
        """Запуск мониторинга"""
        await self.client.start(phone=self.phone)
        logger.info("Клиент запущен")
        
        # Проверяем доступность каналов
        await self.validate_channels()
        
        # Регистрируем обработчик новых сообщений
        @self.client.on(events.NewMessage())
        async def handle_new_message(event):
            await self.process_new_message(event)
        
        logger.info("Мониторинг запущен. Ожидание новых сообщений...")
        await self.client.run_until_disconnected()

    async def validate_channels(self):
	    """Проверка доступности каналов и топиков"""
	    valid_sources = []
	    valid_targets = []
	    
	    # Проверяем источники
	    for source_key in self.source_to_targets.keys():
	        try:
	            source_data = self.normalize_channel_identifier(source_key)
	            await asyncio.sleep(random.uniform(1, 3))
	            entity = await self.client.get_entity(source_data["channel"])
	            
	            # Источники всегда обычные каналы (без топиков в данной задаче)
	            valid_sources.append(source_key)
	            logger.info(f"✓ Источник доступен: {entity.title} ({source_data['channel']})")
	                
	        except Exception as e:
	            logger.error(f"✗ Ошибка проверки источника {source_key}: {e}")
	    
	    # Проверяем целевые каналы
	    for target_channel_str in self.channel_config.keys():
	        try:
	            target_data = self.normalize_channel_identifier(target_channel_str)
	            await asyncio.sleep(random.uniform(1, 3))
	            entity = await self.client.get_entity(target_data["channel"])
	            
	            if target_data["topic_id"]:
	                # Проверяем топик
	                if hasattr(entity, 'forum') and entity.forum:
	                    try:
	                        async for message in self.client.iter_messages(entity, reply_to=target_data["topic_id"], limit=1):
	                            pass
	                        valid_targets.append(target_channel_str)
	                        logger.info(f"✓ Целевой канал с топиком доступен: {entity.title} (топик {target_data['topic_id']})")
	                    except Exception:
	                        logger.warning(f"✗ Топик {target_data['topic_id']} недоступен в {entity.title}")
	                else:
	                    logger.warning(f"✗ Канал {entity.title} не поддерживает топики")
	            else:
	                valid_targets.append(target_channel_str)
	                logger.info(f"✓ Целевой канал доступен: {entity.title}")
	                
	        except Exception as e:
	            logger.error(f"✗ Ошибка проверки целевого канала {target_channel_str}: {e}")
	    
	    # Обновляем конфигурацию только валидными каналами
	    updated_config = {}
	    for target_channel_str, source_channels in self.channel_config.items():
	        if target_channel_str in valid_targets:
	            valid_sources_for_target = [src for src in source_channels if src in valid_sources]
	            if valid_sources_for_target:
	                updated_config[target_channel_str] = valid_sources_for_target
	    
	    self.channel_config = updated_config
	    self.build_source_to_targets_mapping()
    
        
    def normalize_channel_identifier(self, channel_str):
	    """Нормализация идентификатора канала из строки"""
	    if '#' in channel_str:
	        channel, topic_id = channel_str.split('#', 1)
	        return {"channel": channel, "topic_id": int(topic_id)}
	    else:
	        return {"channel": channel_str, "topic_id": None}
    
    
    def get_channel_key(self, channel_data):
	    """Получение ключа для идентификации канала"""
	    normalized = self.normalize_channel_identifier(channel_data)
	    if normalized["topic_id"]:
	        return f"{normalized['channel']}#{normalized['topic_id']}"
	    return normalized["channel"]
    
    async def process_new_message(self, event):
	    """Обработка нового сообщения (источники только обычные каналы)"""
	    try:
	        sender = await event.get_sender()
	        if not sender:
	            return
	
	        source_key = None
	        
	        # Поиск источника среди обычных каналов
	        for configured_source_key in self.source_to_targets.keys():
	            try:
	                source_data = self.normalize_channel_identifier(configured_source_key)
	                await asyncio.sleep(random.uniform(1, 3))
	                configured_entity = await self.client.get_entity(source_data["channel"])
	                
	                if sender.id == configured_entity.id:
	                    source_key = configured_source_key
	                    break
	                        
	            except Exception as e:
	                logger.error(f"Ошибка проверки источника {configured_source_key}: {e}")
	                continue
	        
	        if not source_key:
	            return
	
	        if event.message.id in self.processed_messages:
	            return
	
	        if not self.should_process_message(event.message):
	            return
	
	        # Получаем целевые каналы
	        target_configs = self.source_to_targets[source_key]
	        
	        # Обрабатываем сообщение
	        processed_text = self.process_message_text(event.message.raw_text or "")
	        response_ai = await self.ai_client.get_text_response(processed_text)
	        if response_ai:
	            processed_text = response_ai
	        
	        # Скачиваем медиа
	        media_files = []
	        if event.message.media and self.media_settings['download_media']:
	            media_files = await self.download_media(event.message)
	        
	        # Публикуем в целевые каналы
	        await self.publish_to_targets_with_topics(processed_text, target_configs, media_files)
	        
	        # Очищаем временные файлы
	        if self.media_settings['clean_temp_files']:
	            await self.cleanup_media_files(media_files)
	        
	        self.processed_messages.add(event.message.id)
	        
	        if len(self.processed_messages) > 1000:
	            self.processed_messages = set(list(self.processed_messages)[-500:])
	        
	        logger.info(f"Сообщение из {source_key} обработано и опубликовано в {len(target_configs)} каналов")
	
	    except Exception as e:
	        logger.error(f"Ошибка при обработке сообщения: {e}")
        
    
    async def publish_to_targets_with_topics(self, text: str, target_configs: List[Dict], media_files: List[Dict]):
	    """Публикация сообщения в целевые каналы с поддержкой топиков"""
	    for target_config in target_configs:
	        try:
	            await asyncio.sleep(self.filters['delay_between_posts'])
	            
	            target_data = target_config["target_data"]
	            channel = target_data["channel"]
	            topic_id = target_data["topic_id"]
	            
	            # Отправляем с учетом топика
	            if media_files:
	                await self.send_media_to_channel_with_topic(channel, text, media_files, topic_id)
	            else:
	                if text:
	                    if topic_id:
	                        # Отправляем в топик
	                        await self.client.send_message(
	                            channel, 
	                            text, 
	                            reply_to=topic_id
	                        )
	                    else:
	                        # Обычная отправка
	                        await self.client.send_message(channel, text)
	            
	            target_display = f"{channel} (топик {topic_id})" if topic_id else channel
	            logger.info(f"Сообщение опубликовано в {target_display}")
	            
	        except Exception as e:
	            logger.error(f"Ошибка публикации в {target_config['target']}: {e}")
            

    def should_process_message(self, message) -> bool:
        """Проверка, нужно ли обрабатывать сообщение"""
        text = message.raw_text or ""
        
        # Проверка минимальной длины
        if len(text.strip()) < self.filters['min_length']:
            return False
        
        # Проверка исключающих слов
        text_lower = text.lower()
        for exclude_word in self.filters['keywords_exclude']:
            if exclude_word.lower() in text_lower:
                logger.info(f"Сообщение пропущено из-за исключающего слова: {exclude_word}")
                return False
        
        return True

    def process_message_text(self, text: str) -> str:
        """Обработка текста сообщения перед публикацией"""
        if not text:
            return ""
        
        # Удаляем лишние пробелы и переносы
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Можно добавить дополнительную обработку:
        # - удаление определенных символов
        # - замена ссылок
        # - добавление подписи и т.д.
        
        return text.strip()

    async def download_media(self, message) -> List[Dict]:
        """Скачивание медиа файлов из сообщения"""
        media_files = []
        
        try:
            if not message.media:
                return media_files
            
            # Проверяем размер файла
            file_size = getattr(message.media, 'size', 0)
            if hasattr(message.media, 'document') and message.media.document:
                file_size = message.media.document.size
            
            if file_size > self.media_settings['max_file_size']:
                logger.warning(f"Файл слишком большой ({file_size} байт), пропускаем")
                return media_files
            
            # Определяем тип медиа и расширение
            file_extension = self.get_media_extension(message.media)
            if file_extension and file_extension not in self.media_settings['allowed_extensions']:
                logger.info(f"Тип файла {file_extension} не разрешен, пропускаем")
                return media_files
            
            # Генерируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"media_{message.id}_{timestamp}"
            if file_extension:
                filename += file_extension
            
            filepath = os.path.join(self.media_settings['temp_dir'], filename)
            
            # Скачиваем файл
            logger.info(f"Скачиваем медиа файл: {filename}")
            await message.download_media(file=filepath)
            
            # Проверяем, что файл скачался
            if os.path.exists(filepath):
                media_files.append({
                    'path': filepath,
                    'filename': filename,
                    'type': self.get_media_type(message.media),
                    'size': os.path.getsize(filepath)
                })
                logger.info(f"Медиа файл скачан: {filename} ({os.path.getsize(filepath)} байт)")
            else:
                logger.error(f"Не удалось скачать медиа файл: {filename}")
                
        except Exception as e:
            logger.error(f"Ошибка при скачивании медиа: {e}")
        
        return media_files

    def get_media_extension(self, media) -> Optional[str]:
        """Определение расширения медиа файла"""
        try:
            if isinstance(media, MessageMediaPhoto):
                return '.jpg'
            elif isinstance(media, MessageMediaDocument) and media.document:
                # Ищем имя файла в атрибутах
                for attr in media.document.attributes:
                    if isinstance(attr, DocumentAttributeFilename):
                        filename = attr.file_name
                        return os.path.splitext(filename)[1].lower()
                
                # Определяем по MIME типу
                mime_type = media.document.mime_type
                if mime_type:
                    mime_extensions = {
                        'image/jpeg': '.jpg',
                        'image/png': '.png',
                        'image/gif': '.gif',
                        'image/webp': '.webp',
                        'video/mp4': '.mp4',
                        'video/webm': '.webm',
                        'application/pdf': '.pdf',
                        'application/msword': '.doc',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
                    }
                    return mime_extensions.get(mime_type)
        except Exception as e:
            logger.error(f"Ошибка определения расширения файла: {e}")
        
        return None

    def get_media_type(self, media) -> str:
        """Определение типа медиа"""
        if isinstance(media, MessageMediaPhoto):
            return 'photo'
        elif isinstance(media, MessageMediaDocument) and media.document:
            for attr in media.document.attributes:
                if isinstance(attr, DocumentAttributeVideo):
                    return 'video'
                elif isinstance(attr, DocumentAttributeAudio):
                    return 'audio'
            return 'document'
        return 'unknown'

    async def cleanup_media_files(self, media_files: List[Dict]):
        """Удаление временных медиа файлов"""
        for media_file in media_files:
            try:
                if os.path.exists(media_file['path']):
                    os.remove(media_file['path'])
                    logger.info(f"Удален временный файл: {media_file['filename']}")
            except Exception as e:
                logger.error(f"Ошибка удаления файла {media_file['filename']}: {e}")

    async def publish_to_targets(self, text: str, target_channels: List[str], media_files: List[Dict]):
        """Публикация сообщения в целевые каналы"""
        for target_channel in target_channels:
            try:
                # Задержка между публикациями
                await asyncio.sleep(self.filters['delay_between_posts'])
                
                # Отправляем медиа файлы, если есть
                if media_files:
                    await self.send_media_to_channel(target_channel, text, media_files)
                else:
                    # Отпр_:авляем только текст
                    if text:
                        await self.client.send_message(target_channel, text)
                
                logger.info(f"Сообщение опубликовано в {target_channel}")
                
            except Exception as e:
                logger.error(f"Ошибка публикации в {target_channel}: {e}")

    async def send_media_to_channel(self, channel: str, text: str, media_files: List[Dict]):
        """Отправка медиа файлов в канал"""
        try:
            if len(media_files) == 1:
                # Один файл - отправляем с подписью
                media_file = media_files[0]
                await self.client.send_file(
                    channel,
                    media_file['path'],
                    caption=text if text else None
                )
                logger.info(f"Отправлен файл {media_file['filename']} в {channel}")
                
            elif len(media_files) > 1:
                # Несколько файлов - отправляем альбомом
                file_paths = [media_file['path'] for media_file in media_files]
                await self.client.send_file(
                    channel,
                    file_paths,
                    caption=text if text else None
                )
                logger.info(f"Отправлен альбом из {len(media_files)} файлов в {channel}")
                
        except Exception as e:
            logger.error(f"Ошибка отправки медиа в {channel}: {e}")
            # Fallback - отправляем текст без медиа
            if text:
                try:
                    await self.client.send_message(channel, text)
                    logger.info(f"Отправлен только текст в {channel} (медиа не удалось)")
                except Exception as text_error:
                    logger.error(f"Ошибка отправки текста в {channel}: {text_error}")
                    
    async def send_media_to_channel_with_topic(self, channel: str, text: str, media_files: List[Dict], topic_id: int = None):
	    """Отправка медиа файлов в канал с поддержкой топиков"""
	    try:
	        send_kwargs = {
	            "caption": text if text else None
	        }
	        
	        # Добавляем параметр топика, если указан
	        if topic_id:
	            send_kwargs["reply_to"] = topic_id
	        
	        if len(media_files) == 1:
	            media_file = media_files[0]
	            await self.client.send_file(
	                channel,
	                media_file['path'],
	                **send_kwargs
	            )
	            logger.info(f"Отправлен файл {media_file['filename']} в {channel}")
	            
	        elif len(media_files) > 1:
	            file_paths = [media_file['path'] for media_file in media_files]
	            await self.client.send_file(
	                channel,
	                file_paths,
	                **send_kwargs
	            )
	            logger.info(f"Отправлен альбом из {len(media_files)} файлов в {channel}")
	            
	    except Exception as e:
	        logger.error(f"Ошибка отправки медиа в {channel}: {e}")
	        # Fallback - отправляем текст без медиа
	        if text:
	            try:
	                if topic_id:
	                    await self.client.send_message(channel, text, reply_to=topic_id)
	                else:
	                    await self.client.send_message(channel, text)
	                logger.info(f"Отправлен только текст в {channel} (медиа не удалось)")
	            except Exception as text_error:
	                logger.error(f"Ошибка отправки текста в {channel}: {text_error}")
	    
    async def stop(self):
        """Остановка мониторинга и очистка ресурсов"""
        try:
            # Очищаем временную директорию
            if os.path.exists(self.media_settings['temp_dir']):
                shutil.rmtree(self.media_settings['temp_dir'])
                logger.info("Временная директория очищена")
            
            # Отключаем клиента
            if self.client.is_connected():
                await self.client.disconnect()
                logger.info("Клиент отключен")
                
        except Exception as e:
            logger.error(f"Ошибка при остановке мониторинга: {e}")
    
    async def get_channel_info(self, channel_identifier: str):
		    """Получение информации о канале"""
		    if channel_identifier in self._channel_info_cache:
		        return self._channel_info_cache[channel_identifier]
		    
		    try:
		    	  await asyncio.sleep(random.uniform(1, 3))
		        entity = await self.client.get_entity(channel_identifier)
		        info = {
		            'id': entity.id,
		            'title': entity.title,
		            'username': entity.username,
		            'participants_count': getattr(entity, 'participants_count', 0)
		        }
		        self._channel_info_cache[channel_identifier] = info
		        return info
		    except Exception as e:
		        logger.error(f"Ошибка получения информации о канале {channel_identifier}: {e}")
		        return None

    async def list_channels(self):
        """Вывод списка доступных каналов"""
        dialogs = await self.client.get_dialogs()
        
        channels = []
        
        for dialog in dialogs:
            if isinstance(dialog.entity, Channel):
                channels.append({
                    'title': dialog.entity.title,
                    'username': dialog.entity.username,
                    'id': dialog.entity.id
                })
        
        return channels

    def print_config_summary(self):
        """Вывод краткой информации о конфигурации"""
        logger.info("=== Конфигурация каналов ===")
        for target_channel, source_channels in self.channel_config.items():
            logger.info(f"Целевой канал: {target_channel}")
            for source in source_channels:
                logger.info(f"  <- {source}")
            logger.info("")

# Функция для запуска мониторинга
async def main():
    # Замените на ваши данные
    API_ID = os.getenv("API_ID") # Получить на https://my.telegram.org
    API_HASH = os.getenv("API_HASH") 
    PHONE = os.getenv("PHONE") 
    AI_API_KEY = os.getenv("AI_API_KEY") 
    monitor = TelegramChannelMonitor(API_ID, API_HASH, PHONE,AI_API_KEY)
    
    try:
        # Выводим информацию о конфигурации
        monitor.print_config_summary()
        
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Мониторинг остановлен пользователем")
        await monitor.stop()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        await monitor.stop()

if __name__ == "__main__":
    # Запуск мониторинга
    asyncio.run(main())