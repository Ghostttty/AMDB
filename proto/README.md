# Прототипы к техническому проекту AMDB

Исполняемые фрагменты из [`../docs/Технический проект AMDB.md`](../docs/%D0%A2%D0%B5%D1%85%D0%BD%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%20AMDB.md). Требуется Python 3.11+, `numpy`, `pandas`.

| Скрипт | Что проверяет | Замер на AMD-ноутбуке разработчика (NumPy 2.1.1) |
|---|---|---|
| `core_proto.py` | `build_einsum`; соответствие примеру из статьи Симакова; эквивалентность наивному вычислению; бенчмарк (1,1)-свёртки 100³ | max err 8.9e‑16; **61.7 мс** (в статье — 65.1 мс) |
| `olap_proto.py` | Трансляцию OLAP-запроса `SELECT customer, month, SUM(qty*price) … GROUP BY` в `einsum`; сверку с построчным эталоном; влияние `WHERE`-маски | rel err 2e‑7 (float32); **0.4–0.8 мс** (маска фильтра не добавляет стоимости) |
| `load_proto.py` | Загрузку 500 000 строк реляционного факта в гиперкуб 100×100×100 | **31 мс**, заполненность 39 % |

Запуск:

```bash
python proto/core_proto.py
```

```bash
python proto/olap_proto.py
```

```bash
python proto/load_proto.py
```

Под Windows для корректного вывода кириллицы: `set PYTHONIOENCODING=utf-8`.
