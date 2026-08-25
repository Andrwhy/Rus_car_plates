from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from io import BytesIO

import cv2
import numpy as np
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.config import Settings, load_settings
from app.detector import Plate, PlateDetector, annotate
from app.ocr import read_plate

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO
)
# HTTP request URLs include the Telegram bot token. Keep this noisy dependency logger
# at warning level so credentials never land in local bot logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def _decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Telegram sent an unsupported image format.")
    return image


def _encode_jpeg(image: np.ndarray) -> BytesIO:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")
    file = BytesIO(encoded.tobytes())
    file.name = "plates_detected.jpg"
    return file


def _recognise(plates: list[Plate], image: np.ndarray) -> list[Plate]:
    recognised: list[Plate] = []
    for plate in plates:
        crop = image[plate.y1 : plate.y2, plate.x1 : plate.x2]
        try:
            text = read_plate(crop)
        except Exception:
            LOGGER.exception("OCR failed for a detected plate")
            text = None
        recognised.append(replace(plate, text=text))
    return recognised


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Пришлите фотографию автомобиля — я выделю на ней государственный номер."
        )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        settings: Settings = context.application.bot_data["settings"]
        await update.message.reply_text(
            f"Готов. Модель: {settings.model_path.name}; порог: {settings.confidence:.0%}; "
            f"OCR: {'включён' if settings.enable_ocr else 'выключен'}."
        )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not update.message.photo:
        return
    detector: PlateDetector = context.application.bot_data["detector"]
    settings: Settings = context.application.bot_data["settings"]
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        telegram_photo = update.message.photo[-1]
        remote_file = await telegram_photo.get_file()
        image = _decode_image(bytes(await remote_file.download_as_bytearray()))
        plates = await asyncio.to_thread(detector.detect, image)
        if settings.enable_ocr and plates:
            plates = await asyncio.to_thread(_recognise, plates, image)
        annotated = annotate(image, plates)
        caption = (
            f"Нашёл номеров: {len(plates)}."
            if plates
            else "Номер не найден. Попробуйте более чёткое фото крупнее."
        )
        recognised = [plate.text for plate in plates if plate.text]
        if recognised:
            caption += "\nРаспознано: " + ", ".join(recognised)
        await update.message.reply_photo(photo=_encode_jpeg(annotated), caption=caption)
    except Exception:
        LOGGER.exception("Failed to process incoming photo")
        await update.message.reply_text("Не удалось обработать фото. Попробуйте другое изображение.")


def main() -> None:
    settings = load_settings()
    detector = PlateDetector(settings.model_path, settings.confidence, settings.device)
    application = Application.builder().token(settings.token).build()
    application.bot_data.update(settings=settings, detector=detector)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.PHOTO, photo))
    LOGGER.info("Bot started; model=%s", settings.model_path)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
