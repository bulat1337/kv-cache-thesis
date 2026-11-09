"""Recompute pilot summaries and figures from row-level saved observations."""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = {'full': 'Полное вычисление', 'no_documents': 'Без документов',
          'prefix': 'Доступный префикс', 'prefix_warm': 'Прогретый префикс',
          'repair_all': 'Пересчёт всех строк', 'independent': 'Независимые фрагменты',
          'lego4': 'LegoLink k=4', 'lego16': 'LegoLink k=16',
          'random4-s11': 'Случайный k=4, seed 11', 'random4-s29': 'Случайный k=4, seed 29',
          'random4-s47': 'Случайный k=4, seed 47', 'query4': 'Вопросный k=4 (адаптация)'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interval(rows, field):
    groups = defaultdict(list)
    for r in rows:
        groups[r['group']].append(r[field])
    values = [np.array(v, dtype=float) for v in groups.values()]
    rng = np.random.default_rng(20261107)
    resamples = [np.concatenate([values[i] for i in rng.integers(len(values), size=len(values))]).mean()
                 for _ in range(2000)]
    return [float(v) for v in np.quantile(resamples, [.025, .975])]


def aggregate(rows, baseline):
    loss = sum(baseline[r['id']]['quality']['em'] == 1 and r['quality']['em'] == 0 for r in rows)
    gain = sum(baseline[r['id']]['quality']['em'] == 0 and r['quality']['em'] == 1 for r in rows)
    correct = sum(baseline[r['id']]['quality']['em'] == 1 for r in rows)
    diagnostic = [r for r in rows if baseline[r['id']]['quality']['em'] == 1 and
                  baseline[r['id']]['no_document_em'] == 0]
    f = [dict(group=r['group'], em=r['quality']['em'],
              paired_delta=r['quality']['em'] - baseline[r['id']]['quality']['em']) for r in rows]
    return dict(n=len(rows), groups=len({r['group'] for r in rows}),
                correct=sum(r['quality']['em'] for r in rows), em=st.mean(r['quality']['em'] for r in rows),
                f1=st.mean(r['quality']['f1'] for r in rows), em_cluster_ci95=interval(f, 'em'),
                delta_cluster_ci95=interval(f, 'paired_delta'), lost=loss, gained=gain, full_correct=correct,
                context_dependent_n=len(diagnostic),
                context_dependent_retained=sum(r['quality']['em'] for r in diagnostic),
                median_ttft_s=st.median(r['ttft_s'] for r in rows),
                sum_ttft_s=sum(r['ttft_s'] for r in rows),
                median_complete_s=st.median(r['complete_s'] for r in rows),
                mean_layer_tokens=st.mean(r['computed_layer_tokens'] + r['selector_layer_tokens'] for r in rows),
                max_peak_active_bytes=max(s['peak_active_bytes'] for r in rows for s in r['timing_samples']),
                max_peak_increment_bytes=max(s['peak_increment_bytes'] for r in rows for s in r['timing_samples']),
                truncated=sum(not r['stopped'] for r in rows))


def main():
    root = ROOT / 'results/test'
    rows, checks = read_jsonl(root / 'rows.jsonl'), read_jsonl(root / 'checks.jsonl')
    inputs = {e['id']: e for e in read_jsonl(ROOT / 'data/examples.jsonl') if e['split'] == 'test'}
    assert len(rows) == len(inputs) * len(LABELS) == 240
    assert len({(r['id'], r['method']) for r in rows}) == len(rows)
    assert all(r['split'] == 'test' for r in rows)
    full = {r['id']: r for r in rows if r['method'] == 'full'}
    no_docs = {r['id']: r for r in rows if r['method'] == 'no_documents'}
    for r in full.values():
        r['no_document_em'] = no_docs[r['id']]['quality']['em']
    for r in rows:
        if r['method'] != 'no_documents':
            assert r['token_ids_sha256'] == full[r['id']]['token_ids_sha256']
        assert r['computed_layer_tokens'] == sum(map(len, r['computed_positions_per_layer']))
        if r['method'].startswith('random4') or r['method'] == 'query4':
            lego = next(x for x in rows if x['id'] == r['id'] and x['method'] == 'lego4')
            assert r['computed_layer_tokens'] == lego['computed_layer_tokens']
    summaries = {}
    for dataset in ['synthetic', 'hotpotqa', 'all']:
        summaries[dataset] = {m: aggregate([r for r in rows if r['method'] == m and
                                          (dataset == 'all' or r['dataset'] == dataset)], full) for m in LABELS}
    prep = json.loads((root / 'preparation.json').read_text())
    preparation_s = sum(x['seconds'] for x in prep['independent'])
    prefix_prep_s = sum(x['seconds'] for x in prep['exact_prefix'])
    whole = summaries['all']
    first_prefix = prefix_prep_s + whole['prefix']['sum_ttft_s']
    warm_prefix = whole['prefix_warm']['sum_ttft_s']
    break_even = {}
    for m in ['independent', 'lego4', 'lego16', 'random4-s11', 'random4-s29', 'random4-s47', 'query4']:
        online = whole[m]['sum_ttft_s']
        favorable = [r for r in range(1, 1001) if preparation_s + r * online < first_prefix + (r - 1) * warm_prefix]
        break_even[m] = dict(independent_preparation_s=preparation_s, online_cycle_s=online,
                             first_cycle_total_s=preparation_s + online,
                             prefix_first_cycle_s=first_prefix, prefix_later_cycle_s=warm_prefix,
                             favorable_cycles_1_to_1000=favorable,
                             steady_state_online_faster_than_prefix=online < warm_prefix,
                             first_cycle_vs_full=preparation_s + online < whole['full']['sum_ttft_s'])
    numeric = {label: {stat: max(c[label][stat] for c in checks) for stat in ['max_abs', 'rms']}
               for label in ['prefix', 'all_repair']}
    exact = {}
    for m in ['prefix', 'prefix_warm', 'repair_all']:
        same = sum(r['output_ids'] == full[r['id']]['output_ids'] for r in rows if r['method'] == m)
        exact[m] = {'identical_generation_ids': same, 'n': len(inputs)}
    group_table = {}
    for hops, layout in [(1, 'cross'), (2, 'cross'), (3, 'cross'), (3, 'within')]:
        name = f'{hops}-{layout}'
        group_table[name] = {m: aggregate([r for r in rows if r['method'] == m and r['dataset'] == 'synthetic'
                                          and r['hops'] == hops and r['layout'] == layout], full) for m in LABELS}
    result = dict(summaries=summaries, numerical=numeric, exact=exact, preparation=prep,
                  break_even=break_even, structure=group_table,
                  note='Descriptive pilot; cluster percentile bootstrap 2000 resamples, very few independent groups')
    (ROOT / 'results/summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with open(ROOT / 'results/summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'method', 'n', 'groups', 'EM', 'F1', 'lost', 'full_correct', 'gained',
                         'median_TTFT_ms', 'mean_layer_tokens_including_selection', 'max_active_GiB'])
        for ds, values in summaries.items():
            for m, x in values.items():
                writer.writerow([ds, m, x['n'], x['groups'], x['em'], x['f1'], x['lost'], x['full_correct'],
                                 x['gained'], 1000 * x['median_ttft_s'], x['mean_layer_tokens'],
                                 x['max_peak_active_bytes'] / 2**30])
    chart_methods = ['full', 'prefix', 'prefix_warm', 'independent', 'lego4', 'lego16', 'random4-s11', 'query4']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    y = np.arange(len(chart_methods))
    values = summaries['hotpotqa']
    ems = [values[m]['em'] * 100 for m in chart_methods]
    times = [values[m]['median_ttft_s'] * 1000 for m in chart_methods]
    colors = ['#707A84' if m in ['full', 'prefix', 'prefix_warm'] else '#3477AD' for m in chart_methods]
    axes[0].barh(y, ems, color=colors)
    lower = [100 * values[m]['em_cluster_ci95'][0] for m in chart_methods]
    upper = [100 * values[m]['em_cluster_ci95'][1] for m in chart_methods]
    axes[0].errorbar(ems, y, xerr=np.array([np.array(ems) - lower, np.array(upper) - ems]),
                     fmt='none', color='#24292E', capsize=3, linewidth=1)
    axes[0].set_xlim(0, 110)
    axes[0].set_xlabel('Answer EM, %')
    axes[0].set_yticks(y, [LABELS[m] for m in chart_methods])
    axes[0].invert_yaxis()
    for i, v in enumerate(ems):
        axes[0].text(v + 1.2, i - .23, f'{v:.1f}', va='center', fontsize=10)
    axes[1].barh(y, times, color=colors)
    axes[1].set_xlim(0, max(times) * 1.2)
    axes[1].set_xlabel('Медиана времени до первого токена, мс')
    for i, v in enumerate(times):
        axes[1].text(v + max(times) * .015, i, f'{v:.0f}', va='center')
    fig.suptitle('HotpotQA: качество и задержка локального пилота', fontsize=15, x=.60)
    fig.text(.60, .035, '6 вопросов × 2 порядка; 95% bootstrap по вопросам. Пилотная выборка, Qwen3-1.7B FP16.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .07, 1, .95])
    fig.savefig(ROOT / 'results/hotpot-quality-latency.png', dpi=170)
    plt.close(fig)
    headers = ['Метод', 'EM', 'F1', 'Потеря / верных full', 'Новых верных', 'TTFT, мс', 'Layer-tokens*']
    report = ['# Первый пилот повторного использования KV-кэша', '',
              '07.11.2026. Результаты собственного запуска; фиксированная модель, один backend. '
              'Статус: технический пилот завершён, выводы о научной новизне и устойчивой зависимости остаются открытыми.', '',
              '## Главный результат', '',
              f"На проверочной части полное вычисление дало {whole['full']['correct']:.0f}/20 точных ответов. "
              f"LegoLink k=4 — {whole['lego4']['correct']:.0f}/20, "
              f"k=16 — {whole['lego16']['correct']:.0f}/20. "
              'Численные проверки префикса и пересчёта всех строк пройдены в допуске, закреплённом после калибровки. '
              'Подробности совпадения готовых ответов приведены ниже.', '',
              'На HotpotQA полный эталон решил 3/12 запросов, режим без документов — 4/12. '
              'В этой части нет случаев, где full верен, а режим без документов ошибочен (0/12). '
              'Текущая малая естественная выборка не демонстрирует полезность контекста и требует проверки модели/входного режима. '
              'На синтетических данных full решил 7/8, без документов — 0/8; эта часть подходит для продолжения проверки механизма.', '',
              'На синтетической части LegoLink k=16 сохранил средний EM 7/8, потеряв один из семи правильных ответов full '
              'и исправив другой ответ. Это наблюдение подтверждает необходимость парной оценки потерь и приобретений; '
              'монотонная связь качества с k или числом связей из пилота не установлена.', '',
              '## Данные и протокол', '',
              '20 проверочных запросов: 8 синтетических по двум независимым мирам и 12 HotpotQA по шести вопросам '
              'в двух порядках документов. Отдельные 8 калибровочных запросов в таблицы проверки не входят. '
              'HotpotQA: все десять документов сохранены, ограничение 2048 токенов применяется при отборе примеров; '
              'обрезания текста и отбора по ответам модели нет. Документы повторяются при изменении порядка. '
              'Точный префикс берётся из завершённых предыдущих запросов; отдельно измерен полностью прогретый префикс.', '',
              'Оценка EM/F1 соответствует answer scorer HotpotQA. Строгий EM учитывает также соблюдение формата '
              'короткого ответа: содержательный ответ с пояснением может получить EM=0. Сырые ответы сохранены. '
              'Генерация жадная, thinking=False, предел 48 новых токенов. TTFT — медиана трёх завершённых prefill, '
              'затем в таблице медиана по запросам. Полная генерация выполнена один раз на режим и запрос.', '']
    for ds, title in [('hotpotqa', 'HotpotQA: 12 запросов, 6 независимых вопросов'),
                      ('synthetic', 'Контролируемые задачи: 8 запросов, 2 независимых мира')]:
        report += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
        for m, x in summaries[ds].items():
            report.append(f"| {LABELS[m]} | {x['correct']:.0f}/{x['n']} ({100*x['em']:.1f}%) | "
                          f"{100*x['f1']:.1f}% | {x['lost']}/{x['full_correct']} | {x['gained']} | "
                          f"{1000*x['median_ttft_s']:.1f} | {x['mean_layer_tokens']:.0f} |")
        report += ['', '*Layer-tokens включает новый суффикс и проход вопросного выбора; '
                   'дополнительные операции вычисления attention scores отражены также в измеренном времени.', '']
    report += ['![Качество и задержка HotpotQA](results/hotpot-quality-latency.png)', '',
               'Интервалы неопределённости сохранены в summary.json: percentile bootstrap по общим вопросам/мирам, '
               '2000 повторов. Шесть и два независимых кластера дают нестабильные интервалы; доказательств превосходства '
               'или общей гарантии сохранения качества из этого пилота не следует.', '',
               '## Зависимость от числа связей и границ', '',
               '| Задача | full | independent | LegoLink 4 | LegoLink 16 | random seed11 | query4 |',
               '|---|---|---|---|---|---|---|']
    for condition, values in group_table.items():
        report.append('| ' + condition + ' | ' + ' | '.join(f"{values[m]['correct']:.0f}/{values[m]['n']}"
                      for m in ['full', 'independent', 'lego4', 'lego16', 'random4-s11', 'query4']) + ' |')
    report += ['', 'Каждая строка содержит два запроса. Связь между hops и необходимым бюджетом ещё не установлена. '
               'Вариант within меняет группировку и порядок фактов и названия фрагментов; чистый эффект границ требует '
               'дополнительного контроля этих факторов.', '', '## Численная корректность', '',
               f"Максимальные отличия логитов: prefix max={numeric['prefix']['max_abs']:.6f}, "
               f"RMS={numeric['prefix']['rms']:.6f}; all-repair max={numeric['all_repair']['max_abs']:.6f}, "
               f"RMS={numeric['all_repair']['rms']:.6f}. Допуск и его калибровка — в calibration-decision.md.", '']
    for m, value in exact.items():
        report.append(f"- {LABELS[m]}: полная последовательность сгенерированных token IDs совпала с full "
                      f"в {value['identical_generation_ids']}/{value['n']} запросах.")
    report += ['', 'Шесть тестов на маленькой детерминированной FP32-модели проверяют строгую численную эквивалентность, '
               'RoPE, причинность, бюджет, сохранность пула и scorer. На настоящих весах проверены все 20 запросов. '
               'Точная префиксная семантика допускает малые вычислительные расхождения FP16 при разных формах матриц.', '',
               '## Затраты и окупаемость', '',
               f"Подготовка уникального независимого пула: {preparation_s:.3f} с, "
               f"{sum(r['bytes'] for r in prep['independent'])/2**30:.3f} GiB логических K/V. "
               f"Первый проход точного префикса с подготовкой общей инструкции: {first_prefix:.3f} с; "
               f"повторный прогретый проход: {warm_prefix:.3f} с. Это суммы TTFT по 20 запросам, без декодирования.", '',
               '| Метод | Первый цикл с подготовкой, с | Онлайн-цикл, с | Выгодные циклы среди 1…1000* |',
               '|---|---|---|---|']
    for m, x in break_even.items():
        favorable = x['favorable_cycles_1_to_1000']
        description = f"{min(favorable)}…{max(favorable)}" if favorable else 'нет'
        report.append(f"| {LABELS[m]} | {x['first_cycle_total_s']:.3f} | {x['online_cycle_s']:.3f} | {description} |")
    report += ['', '*Сценарий повторения фиксированной проверочной последовательности: независимый пул готовится один раз; '
               'префикс на первом цикле использует только предыдущие запросы, на последующих сохраняет все префиксы документов. '
               'Формула: T_ind_prepare + r×T_ind_online < T_prefix_first + (r−1)×T_prefix_warm. '
               'Нельзя переносить её на нагрузку с новыми сочетаниями документов без нового измерения. '
               'Скорость рассматривается совместно с таблицей качества: более быстрый ошибочный режим не означает сохранения качества.', '',
               f"Максимум активных аллокаций MLX в измерениях: {max(x['max_peak_active_bytes'] for x in whole.values())/2**30:.3f} GiB. "
               'Он включает веса и удерживаемые для сравнения пулы. Отдельные значения, приросты, allocator cache и RSS '
               'сохранены построчно. Логические байты K/V не включают padding аллокатора и рабочие копии.', '',
               '## Границы переноса публичных методов', '',
               'В стенде реализован принцип LegoLink из EPIC §4.2: фиксированный общий префикс подготовлен точно, '
               'независимые документы начинаются с позиции 0, первые k строк документов и весь вопрос проходят все слои. '
               'Использованы Qwen3 и MLX, один запрос, плотная сборка K/V, без планировщика и CUDA-ядер авторской системы. '
               'Это перенос принципа алгоритма; опубликованные ускорения EPIC не воспроизводились.', '',
               'query4 использует идею ProphetKV §4.2–4.3: проход вопроса по приближённому кэшу и усреднение attention по слоям. '
               'Отличия: фиксированная квота четыре строки на документ, весь шаблонный суффикс вместо выделения только слов вопроса, '
               'отдельное явное вычисление attention scores. Два прохода и все затраты отбора учтены. '
               'Это адаптированный критерий; превосходство над полной опубликованной реализацией не проверялось.', '',
               '## Решение и следующий основной эксперимент', '',
               'Локальный стенд пригоден для продолжения: есть полный эталон, известный набор, независимые фрагменты '
               'и реальный частичный пересчёт. Первый следующий результат — пригодный полный эталон на новой калибровочной '
               'выборке HotpotQA: проверить точное понимание вопросов, шаблон, режим генерации и при необходимости модель. '
               'Текущие проверочные примеры уже просмотрены, для новой итоговой оценки используются другие группы. '
               'Далее — расширенное парное сравнение по числу связей и границам '
               'при контроле длины, порядка и положения ответа. До окончательной проверки выбирается допустимая потеря качества; '
               'бюджеты k калибруются на отдельных мирах. Пилотные 20 запросов не следует использовать как новую закрытую проверку '
               'после подбора параметров. Расширение эксперимента требует новых независимых групп и заранее выбранного допуска потери качества.', '',
               '## Источники и воспроизведение', '',
               '- [Команды и механизм](README.md), [конфигурация](config.json), [версии](requirements.lock.txt).',
               '- [Входы](data/examples.jsonl), [происхождение данных](data/manifest.json), [источники и хэши](sources/manifest.json).',
               '- [Построчные результаты](results/test/rows.jsonl), [численные проверки](results/test/checks.jsonl), '
               '[подготовка](results/test/preparation.json), [агрегаты CSV](results/summary.csv), [все агрегаты JSON](results/summary.json).',
               '- [EPIC](https://proceedings.mlr.press/v267/hu25j.html), '
               '[ProphetKV, §4](https://arxiv.org/html/2602.02579v3#S4), '
               '[HotpotQA](https://hotpotqa.github.io/), [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B).', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(report))
    print(json.dumps({'full': whole['full'], 'lego4': whole['lego4'], 'numerical': numeric, 'exact': exact}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
