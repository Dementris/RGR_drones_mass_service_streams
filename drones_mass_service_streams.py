from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from itertools import product
import seaborn as sns


@dataclass(init=True)
class Packet:
    process: np.float64 = 0.0
    age: int = 0


# Параметри системи
iterations = 100
n = 5  # Кількість дронів
max_efficiency = 20  # Максимальна ефективність обробки для всіх дронів, пакетів / с
queue_max_size = 20  # Максимальний розмір черги, пакетів
max_age = 40


def distribute_tasks_with_queues(s, R, queues):
    n = len(s)
    p = np.zeros(n)
    d = np.zeros(n)
    remaining_R = R

    # Обробляємо черги дронів: в кого в черзі щось є - забирають на обробку
    for i in range(n):
        if queues[i].age > max_age:
            # Drop process if age > max_age
            d[i] += queues[i].process
            queues[i].process = 0
            queues[i].age = 0
        if queues[i].process > 0:
            # Обробляємо дані з черги
            to_process = min(queues[i].process, s[i], max_efficiency)
            # Дані з черги забираються на обробку
            p[i] += to_process
            # Черга зменшується, адже дані забрали
            queues[i].process -= to_process
            queues[i].age = queues[i].age + 1 if queues[i].process > 0 else 0

    # Розподіл нового потоку
    for i in np.argsort(-s):  # За спаданням швидкості - перш за все найпотужніші дрони
        if remaining_R <= 0:
            break
        # Залишки вільного місця в черзі дрона
        free_space = queue_max_size - queues[i].process
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
            queues[i].process += to_queue
            remaining_R -= to_queue

    return p, queues, d


def experiment():
    R = 100  # Початкова швидкість потоку, пакетів / с
    # Початкові швидкості та черги
    s = np.array([10, 15, 20, 12, 8])  # Швидкості обробки
    queues: list[Packet] = [Packet() for _ in range(n)]  # Початкові черги (усі порожні)

    queues_vals = []
    processed_vals = []
    dropped_vals = []
    r_vals = []

    # Симуляція змін характеристик через погіршення зв'язку

    total_processed = 0
    total_dropped = 0
    total_queues = 0
    for t in range(iterations):
        # Випадкові зміни характеристик зв'язку
        noise = np.random.uniform(-2, 2, size=n)  # Зміна характеристик у межах [-2, 2]
        s = np.clip(s + noise, 1, max_efficiency)  # Забезпечуємо, що швидкість залишається в межах [1, max_efficiency]

        # Зміна загального потоку R
        R = max(50, R + np.random.randint(-10, 10))  # Потік змінюється випадково, але не менше 50

        # Розподіл потоку з урахуванням черг
        p, queues, d = distribute_tasks_with_queues(s, R, queues)

        # Виведення результатів
        queues_vals.append(np.sum([packet.process for packet in queues]) / n)
        processed_vals.append(np.sum(p))
        dropped_vals.append(np.sum(d))
        r_vals.append(R)

        total_processed += np.sum(p)
        total_dropped += np.sum(d)
        total_queues += np.sum([packet.process for packet in queues]) / n

    # plt.figure(figsize=(12, 8))
    # plt.plot(range(iterations), r_vals, label='R values')
    # plt.plot(range(iterations), queues_vals, label='Queues average values')
    # plt.plot(range(iterations), processed_vals, label='Processed values')
    # plt.plot(range(iterations), dropped_vals, label='Dropped values')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return total_processed, total_dropped, total_queues



if __name__ == '__main__':
    queue_max_size_range = range(10, 51, 10)
    max_age_range = range(20, 101, 20)
    experiment_results = []
    for queue_max_size, max_age in product(queue_max_size_range, max_age_range):
        total_processed, total_dropped, total_queues = experiment()

        experiment_results.append({
            "queue_max_size": queue_max_size,
            "max_age": max_age,
            "avg_processed": total_processed / iterations,
            "avg_dropped": total_dropped / iterations,
            "avg_queue_fill": total_queues / iterations,
        })
    results_df = pd.DataFrame(experiment_results)
    print(results_df.head())

    # Візуалізація результатів
    plt.figure(figsize=(18, 6))

    # Графік середньої кількості оброблених пакетів
    plt.subplot(1, 3, 1)
    sns.heatmap(
        results_df.pivot(index="queue_max_size", columns="max_age", values="avg_processed"),
        annot=True, fmt=".1f", cmap="viridis"
    )
    plt.title("Average Processed Packets")
    plt.xlabel("Max Age")
    plt.ylabel("Queue Max Size")

    # Графік середньої кількості втрачених пакетів
    plt.subplot(1, 3, 2)
    sns.heatmap(
        results_df.pivot(index="queue_max_size", columns="max_age", values="avg_dropped"),
        annot=True, fmt=".1f", cmap="magma"
    )
    plt.title("Average Dropped Packets")
    plt.xlabel("Max Age")
    plt.ylabel("Queue Max Size")

    # Графік середнього заповнення черги
    plt.subplot(1, 3, 3)
    sns.heatmap(
        results_df.pivot(index="queue_max_size", columns="max_age", values="avg_queue_fill"),
        annot=True, fmt=".1f", cmap="coolwarm"
    )
    plt.title("Average Queue Fill")
    plt.xlabel("Max Age")
    plt.ylabel("Queue Max Size")

    plt.tight_layout()
    plt.show()