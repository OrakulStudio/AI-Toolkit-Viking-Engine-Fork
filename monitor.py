import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re

# --- НАСТРОЙКИ (просто проверь имя файла) ---
LOG_FILE = "log_mariR1024" 
UPDATE_MS = 5000 # обновлять каждые 5 секунд
# --------------------------------------------

# Создаем окно
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title('Orakul Viking Monitor')

# --- ЭФФЕКТ АКРИЛА И ЦВЕТ ---
# Прозрачность 0.8 (сделай 0.6, если хочешь еще прозрачнее)
try:
    fig.canvas.manager.window.attributes('-alpha', 0.8)
except:
    pass # На случай если бэкенд не поддерживает

# Глубокий темный цвет фона
fig.patch.set_facecolor('#0B0F14') 
ax.set_facecolor('#0B0F14')

def animate(i):
    if not os.path.exists(LOG_FILE):
        return

    losses = []
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Умный поиск: ловит и "0.5125" и "5.125e-01"
            found = re.findall(r'loss[:\s=]+([-+]?\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', content.lower())
            losses = [float(x) for x in found]
    except Exception as e:
        print(f"Ошибка чтения: {e}")

    if losses:
        ax.clear()
        
        # Рисуем ВСЮ историю тончайшей линией (0.5 - волосок)
        ax.plot(range(len(losses)), losses, color='#00FFFF', linewidth=0.5)
        
        # Настройка сетки и шрифтов
        ax.grid(True, alpha=0.05, linestyle=':') 
        ax.set_title(f"Viking Step: {len(losses)} | Loss: {losses[-1]:.4f}", 
                     color='#00FFFF', fontsize=10, loc='left')
        
        # Убираем рамки для чистого вида
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()

# Запуск анимации
ani = animation.FuncAnimation(fig, animate, interval=UPDATE_MS)
plt.show()