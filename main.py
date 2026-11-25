import cv2
import numpy as np
import os
import stat
from datetime import datetime


input_dir = "Lab2_Media/data/input"
output_dir = "Lab2_Media/data/output"
log_file = "Lab2_Media/logs/history.txt"

print("🚀 Программа запущена! Введите 'exit' для выхода.\n")

while True:
    filename = input("Введите имя файла для обработки (или 'exit'): ").strip()

    if filename.lower() == 'exit':
        print("\n👋 Программа завершена.")
        break

    if not filename:
        print("⚠️  Имя файла пустое. Попробуйте ещё раз.\n")
        continue

    filepath = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, "result.jpg")

    status = "Успешно"
    filter_name = "Повышение резкости + Гамма-коррекция"  

    try:
        # 2. Проверка существования файла
        if not os.path.exists(filepath):
            raise FileNotFoundError("Файл не найден")

        # 3. Проверка прав доступа (атрибут "Только чтение")
        file_stat = os.stat(filepath)
        if file_stat.st_mode & stat.S_IREAD and not (file_stat.st_mode & stat.S_IWRITE):
            raise PermissionError(f"Нет прав на чтение файла {filename}")

        img = cv2.imread(filepath)

        if img is None:
            raise ValueError("Файл найден, но не является корректным изображением.")

        # 4. Обработка изображения (ВАРИАНТ 44)
        
        # --- Задание №1: Повышение резкости (Unsharp Masking) ---
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        img_processed = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        # --- Задание №2: Гамма-коррекция ---
        gamma = 1.5  
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img_processed = cv2.LUT(img_processed, table)

        cv2.imwrite(output_path, img_processed)
        print(f"✅ Изображение успешно обработано и сохранено в {output_path}\n")

    # 6. Обработка ошибок
    except PermissionError:
        status = "Ошибка доступа"
        print(f"❌ Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору.\n")

    except FileNotFoundError:
        status = "Ошибка доступа"
        print("❌ Ошибка: Файл не найден\n")

    except Exception as e:
        status = "Ошибка доступа"
        print(f"❌ Произошла ошибка: {e}\n")

    # 7. Логирование (Потоковый вывод)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Файл: {filename} | Фильтр: {filter_name} | Статус: {status}\n"

    with open(log_file, 'a', encoding='UTF-8') as log:
        log.write(log_entry)

    print("📝 Лог обновлён.\n" + "-"*50 + "\n")
