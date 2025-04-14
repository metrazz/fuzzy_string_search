import requests
import time
import os
from uuid import uuid4


class FuzzySearch:
    def __init__(self):
        self.corpuses = {}  # {corpus_id: {"name": str, "words": List[str]}}
        self.current_corpus = None

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return FuzzySearch.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def damerau_levenshtein_distance(s1: str, s2: str) -> int:
        d = {}
        len1 = len(s1)
        len2 = len(s2)

        for i in range(-1, len1 + 1):
            d[(i, -1)] = i + 1
        for j in range(-1, len2 + 1):
            d[(-1, j)] = j + 1

        for i in range(len1):
            for j in range(len2):
                cost = 0 if s1[i] == s2[j] else 1
                d[(i, j)] = min(
                    d[(i - 1, j)] + 1,  # deletion
                    d[(i, j - 1)] + 1,  # insertion
                    d[(i - 1, j - 1)] + cost  # substitution
                )
                if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    d[(i, j)] = min(
                        d[(i, j)],
                        d[(i - 2, j - 2)] + cost  # transposition
                    )

        return d[(len1 - 1, len2 - 1)]

    def load_corpus_from_file(self, file_path: str, name: str) -> str:
        try:
            with open(file_path, 'r', encoding='cp1251') as f:
                content = f.read()

            words = []
            for line in content.split('\n'):
                words.extend(line.split())

            corpus_id = str(uuid4())
            self.corpuses[corpus_id] = {
                "name": name,
                "words": words
            }

            if self.current_corpus is None:
                self.current_corpus = corpus_id

            return corpus_id
        except Exception as e:
            raise Exception(f"Error loading corpus from file: {str(e)}")

    def load_corpus_from_url(self, url: str, name: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text

            words = []
            for line in content.split('\n'):
                words.extend(line.split())

            corpus_id = str(uuid4())
            self.corpuses[corpus_id] = {
                "name": name,
                "words": words
            }

            if self.current_corpus is None:
                self.current_corpus = corpus_id

            return corpus_id
        except Exception as e:
            raise Exception(f"Error loading corpus from URL: {str(e)}")

    def list_corpuses(self) -> list:
        return [
            {
                "id": corpus_id,
                "name": info["name"],
                "word_count": len(info["words"]),
                "is_current": corpus_id == self.current_corpus
            }
            for corpus_id, info in self.corpuses.items()
        ]

    def set_current_corpus(self, corpus_id: str):
        if corpus_id in self.corpuses:
            self.current_corpus = corpus_id
        else:
            raise Exception("Corpus not found")

    def search_all_algorithms(
            self,
            word: str,
            max_distance: int = 2,
            limit: int = 10
    ) -> dict:
        """
        Выполняет нечеткий поиск по обоим алгоритмам

        Параметры:
        - word: искомое слово
        - max_distance: максимальное расстояние для включения в результаты
        - limit: максимальное количество результатов для каждого алгоритма

        Возвращает:
        {
            "levenshtein": {
                "results": список найденных слов с расстояниями,
                "time_taken": время выполнения поиска
            },
            "damerau_levenshtein": {
                "results": список найденных слов с расстояниями,
                "time_taken": время выполнения поиска
            },
            "corpus": имя текущего корпуса
        }
        """
        if not self.corpuses:
            raise Exception("No corpuses available")

        if self.current_corpus is None:
            self.current_corpus = next(iter(self.corpuses.keys()))

        words = self.corpuses[self.current_corpus]["words"]
        corpus_name = self.corpuses[self.current_corpus]["name"]

        results = {
            "levenshtein": {"results": [], "time_taken": 0},
            "damerau_levenshtein": {"results": [], "time_taken": 0},
            "corpus": corpus_name
        }

        # Поиск по Левенштейну
        start_time = time.time()
        lev_results = []
        for candidate in words:
            distance = self.levenshtein_distance(word.lower(), candidate.lower())
            if distance <= max_distance:
                lev_results.append({"word": candidate, "distance": distance})
        lev_results.sort(key=lambda x: x["distance"])
        results["levenshtein"]["results"] = lev_results[:limit]
        results["levenshtein"]["time_taken"] = time.time() - start_time

        # Поиск по Дамерау-Левенштейну
        start_time = time.time()
        dam_lev_results = []
        for candidate in words:
            distance = self.damerau_levenshtein_distance(word.lower(), candidate.lower())
            if distance <= max_distance:
                dam_lev_results.append({"word": candidate, "distance": distance})
        dam_lev_results.sort(key=lambda x: x["distance"])
        results["damerau_levenshtein"]["results"] = dam_lev_results[:limit]
        results["damerau_levenshtein"]["time_taken"] = time.time() - start_time

        return results

    def interactive_search_loop(self):
        print("=== Система нечеткого поиска ===")
        print("Загрузите корпус для начала работы")

        while True:
            if not self.corpuses:
                print("\nНет загруженных корпусов. Пожалуйста, загрузите корпус.")
                self.load_corpus_interactive()
                continue

            print("\nТекущий корпус:", self.corpuses[self.current_corpus]["name"])
            print("1. Выполнить поиск")
            print("2. Загрузить новый корпус")
            print("3. Выбрать другой корпус")
            print("4. Показать список корпусов")
            print("5. Выход")

            choice = input("Выберите действие: ").strip()

            if choice == "1":
                self.run_search()
            elif choice == "2":
                self.load_corpus_interactive()
            elif choice == "3":
                self.select_corpus_interactive()
            elif choice == "4":
                self.show_corpuses()
            elif choice == "5":
                print("Выход из программы...")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")

    def load_corpus_interactive(self):
        print("\nЗагрузка нового корпуса:")
        source = input("Загрузить из (1) файла или (2) URL? ").strip()
        name = input("Название корпуса: ").strip()

        try:
            if source == "1":
                file_path = input("Путь к файлу: ").strip()
                corpus_id = self.load_corpus_from_file(file_path, name)
            elif source == "2":
                url = input("URL файла: ").strip()
                corpus_id = self.load_corpus_from_url(url, name)
            else:
                print("Неверный выбор источника.")
                return

            print(f"\nКорпус '{name}' успешно загружен. ID: {corpus_id}")
            self.current_corpus = corpus_id
        except Exception as e:
            print(f"\nОшибка: {str(e)}")

    def select_corpus_interactive(self):
        print("\nДоступные корпусы:")
        corpuses = self.list_corpuses()
        for i, corpus in enumerate(corpuses, 1):
            current_mark = " (текущий)" if corpus["is_current"] else ""
            print(f"{i}. {corpus['name']} (слов: {corpus['word_count']}){current_mark}")

        choice = input("Выберите номер корпуса: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(corpuses):
                self.set_current_corpus(corpuses[index]["id"])
                print(f"Выбран корпус: {corpuses[index]['name']}")
            else:
                print("Неверный номер корпуса.")
        except ValueError:
            print("Введите число.")

    def show_corpuses(self):
        print("\nДоступные корпусы:")
        for corpus in self.list_corpuses():
            current_mark = " (текущий)" if corpus["is_current"] else ""
            print(f"- {corpus['name']} (ID: {corpus['id']}, слов: {corpus['word_count']}{current_mark}")

    def run_search(self):
        word = input("\nВведите слово для поиска: ").strip()
        if not word:
            print("Слово не может быть пустым.")
            return

        max_dist = input("Максимальное расстояние (по умолчанию 2): ").strip()
        try:
            max_dist = int(max_dist) if max_dist else 2
        except ValueError:
            print("Некорректное значение расстояния. Используется значение по умолчанию (2).")
            max_dist = 2

        limit = input("Лимит результатов (по умолчанию 10): ").strip()
        try:
            limit = int(limit) if limit else 10
        except ValueError:
            print("Некорректное значение лимита. Используется значение по умолчанию (10).")
            limit = 10

        try:
            results = self.search_all_algorithms(word, max_dist, limit)

            print(f"\nРезультаты поиска в корпусе '{results['corpus']}':")
            print(f"Искомое слово: '{word}', Макс. расстояние: {max_dist}")

            print("\nАлгоритм Левенштейна (время: {:.4f} сек):".format(
                results["levenshtein"]["time_taken"]))
            for i, item in enumerate(results["levenshtein"]["results"], 1):
                print(f"{i}. {item['word']} (расстояние: {item['distance']})")

            print("\nАлгоритм Дамерау-Левенштейна (время: {:.4f} сек):".format(
                results["damerau_levenshtein"]["time_taken"]))
            for i, item in enumerate(results["damerau_levenshtein"]["results"], 1):
                print(f"{i}. {item['word']} (расстояние: {item['distance']})")

            lev_words = {item["word"] for item in results["levenshtein"]["results"]}
            dam_lev_words = {item["word"] for item in results["damerau_levenshtein"]["results"]}

            common = lev_words & dam_lev_words
            only_lev = lev_words - dam_lev_words
            only_dam_lev = dam_lev_words - lev_words

            print("\nСравнение результатов:")
            print(f"- Общие результаты: {len(common)}")
            print(f"- Только в Левенштейне: {len(only_lev)}")
            print(f"- Только в Дамерау-Левенштейне: {len(only_dam_lev)}")

        except Exception as e:
            print(f"\nОшибка при выполнении поиска: {str(e)}")


if __name__ == "__main__":
    searcher = FuzzySearch()
    searcher.interactive_search_loop()