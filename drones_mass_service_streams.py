import numpy as np
from matplotlib import pyplot as plt

# Параметри системи
n = 5  # Кількість дронів
max_efficiency = 20  # Максимальна ефективність обробки для всіх дронів, пакетів / с
queue_max_size = 20  # Максимальний розмір черги, пакетів
R = 100  # Початкова швидкість потоку, пакетів / с
# Початкові швидкості та черги
s = np.array([10, 15, 20, 12, 8])  # Швидкості обробки
print(f'Total productivity: {s.sum()}')
queues = np.zeros(n)  # Початкові черги (усі порожні)

def distribute_tasks_with_queues(s, R, queues):
    n = len(s)
    p = np.zeros(n)
    remaining_R = R

    # Обробляємо черги дронів: в кого в черзі щось є - забирають на обробку
    for i in range(n):
        if queues[i] > 0:
            # Обробляємо дані з черги
            to_process = min(queues[i], s[i], max_efficiency)
            # Дані з черги забираються на обробку
            p[i] += to_process
            # Черга зменшується, адже дані забрали
            queues[i] -= to_process

    # Розподіл нового потоку
    for i in np.argsort(-s):  # За спаданням швидкості - перш за все найпотужніші дрони
        if remaining_R <= 0:
            break
        # Залишки вільного місця в черзі дрона
        free_space = queue_max_size - queues[i]
        # Залишки "продуктивності" дрона після забору даних із черги
        leftover_power = min((s[i] - p[i]), max_efficiency)
        # Якщо місця в черзі нема - заздалегідь відомо що й сам дрон зайнятий
        if free_space > 0:
            # Максимально можливий потік для дрона, враховуючи те що він може вже бути зайнятий
            # даними з черги
            max_possible = min(max_efficiency, s[i] - p[i], remaining_R)
            # Скільки може бути оброблено напряму: оскільки розмір черги
            # заздалегідь більший за потужність, то пріорітетніше дані
            # подадуться на саму обробку
            direct_process = min(max_possible, free_space)
            p[i] += direct_process
            remaining_R -= direct_process
            # Додаємо залишок у чергу
            to_queue = min(remaining_R, free_space - direct_process)
            queues[i] += to_queue
            remaining_R -= to_queue

    return p, queues

queues_vals = []
processed_vals = []
r_vals = []

# Симуляція змін характеристик через погіршення зв'язку
iterations = 100
for t in range(iterations):
    # Випадкові зміни характеристик зв'язку
    noise = np.random.uniform(-2, 2, size=n)  # Зміна характеристик у межах [-2, 2]
    s = np.clip(s + noise, 1, max_efficiency)  # Забезпечуємо, що швидкість залишається в межах [1, max_efficiency]

    # Зміна загального потоку R
    R = max(50, R + np.random.randint(-10, 10))  # Потік змінюється випадково, але не менше 50

    # Розподіл потоку з урахуванням черг
    p, queues = distribute_tasks_with_queues(s, R, queues)

    # Виведення результатів
    queues_vals.append(np.sum(queues) / n)
    processed_vals.append(np.sum(p))
    r_vals.append(R)

plt.figure(figsize=(12, 8))
plt.plot(range(iterations), r_vals, label='R values')
plt.plot(range(iterations), queues_vals, label='Queues average values')
plt.plot(range(iterations), processed_vals, label='Processed values')
plt.legend()
plt.grid()
plt.show()