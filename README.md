# swim-bladder-bifurcations

Код и данные к статье «Мягкая и жёсткая потеря пассивной устойчивости глубины у аппарата с плавательным пузырём»
(Н. А. Исмаилов, В. В. Матросов, С. А. Лобов, А. Е. Храмов, В. Б. Казанцев).

Модель — продольное движение аппарата с плавательным пузырём на постоянной скорости хода $U$: глубина $z$ и тангаж $\theta$.
Пузырь даёт силу $F_A = k_b (z + r_A \sin\theta)$, плавники — подъёмную силу $F_L = K U^2 \alpha$,
где $\alpha = \theta - \arctan(\dot z / U)$:

$$m\ddot z = F_A + F_L\cos\gamma,\qquad J\ddot\theta = r_A\cos\theta\,F_A + r_L F_L\cos\alpha - k_\theta\theta.$$

Параметры: $m = J = K = 1$, $k_b = 0{,}5$, $k_\theta = 5$, $r_A = -4{,}6$ м; $d = r_L - r_L(B=0)$.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Для `figures.py movie` нужен ffmpeg.

## Состав

```
bladder/          модель и численные методы
  model.py        правая часть, линеаризация, пороги a4 = 0 и B = 0, сёдла вилки
  normal_form.py  первая ляпуновская величина, нормальная форма в точке Хопф-вилки
  cycles.py       симметричные и несимметричные циклы (стрельба), линии SB и TR
  attractors.py   установление, сечение Пуанкаре, показатель Ляпунова, классификация режимов
scripts/          расчёты к статье (запуск из корня репозитория)
data/             результаты расчётов
figures/          рисунки статьи и Supplementary Movie 1
paper/numbers.tex числа, приводимые в тексте статьи
```

## Воспроизведение

| Скрипт | Результат | Рисунок |
|---|---|---|
| `symmetric_branches.py` | `symmetric_branches.json`, `branch_U2.npy` | 5 |
| `corner.py` | `corner.json`, `sb_corner.npy`, `nf_lines.npy` | 6 |
| `sb_tr_lines.py` | `sb_line.npy`, `tr_line.npy` | 6 |
| `regime_map.py` | `regime_map.npy` | 6 |
| `hopf_sign.py` | `hopf_sign.json`, `l1_map.npz`: знак $l_1$ на $B=0$ | — |
| `no_torus.py` | проверка ухода между $a_4=0$ и TR (вывод в консоль) | 6 |
| `sweeps.py d_up` (и `d_down`, `d_zoom`, `U_down`, `U_up`) | `sweep_*.npz` | 7 |
| `paper_numbers.py` | `paper/numbers.tex` | — |
| `figures.py` | `figures/*.pdf`, `*.png` | 1–7 |

Первые два скрипта и рисунки считаются за минуты; карта режимов и развёртки — часы.
`sb_tr_lines.py` и `regime_map.py` используют все ядра процессора.

```bash
python scripts/corner.py
python scripts/figures.py s06
```

Все результаты уже лежат в `data/`, поэтому рисунки и `numbers.tex` можно получить без долгих расчётов.

Классы режимов в `regime_map.npy` (третий столбец): 0 — равновесие, 1 — симметричный цикл,
2 — несимметричный цикл, 3 — цикл кратного периода, 4 — режим не установился,
5 — хаотический аттрактор, 6 — нет аттрактора.
