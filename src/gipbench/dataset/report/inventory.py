#!/usr/bin/env python3
"""Analyze the current E3 manifests and write a reproducible dataset EDA report."""
import array
import csv
import datetime as dt
import hashlib
import gzip
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

from gipbench.core.fileio import sha256_file
from gipbench.core.paths import REPO as ROOT
from gipbench.core.leak import is_test_only, norm_text
from gipbench.core.manifest import span_meta, write_jsonl
from gipbench.core.textnorm import Catalog, normalize_ref, tagged_well_formed
from gipbench.core.scoring import train_term_freq
from gipbench.dataset.report.lineage import collect_lineage, render_lineage, LINEAGE_CSS

OUT = ROOT / 'reports' / 'dataset' / 'eda_current'
CATALOG = ROOT / 'data/bench_v0_draft/catalog_v0_draft.tsv'
ACTIVE = {
    'train_v1': ('Curate v1 · trước Lhotse', 'data/e3_v1/train_curated.jsonl'),
    'train_v1b': ('Curate v1b · trước Lhotse', 'data/e3_v1b/train_curated.jsonl'),
    'dev_cs': ('Dev chen Anh', 'data/e3_v1/dev_cs.jsonl'),
    'dev_vi': ('Dev tiếng Việt', 'data/e3_v1/dev_vi.jsonl'),
    'bench': ('Bench v0-draft', 'data/bench_v0_draft/bench_v0_draft.jsonl'),
}
INPUTS = [
    'data/e3_v1/train_vimedcss_abs.jsonl', 'data/e3_v1/train_vss_cs.jsonl',
    'data/e3_v1/train_vivos.jsonl', 'data/raw/cv/train_cv.jsonl',
    'data/e3_v1/distill.jsonl', 'data/e3_v1/train_confusable.jsonl',
]
INVENTORY = [
    'data/raw/vimedcss_full/train_vimedcss.jsonl', 'data/raw/vimedcss_full/dev_vimedcss.jsonl',
    'data/raw/vss_full/train_vss_dom.jsonl', 'data/raw/vss_full/dev_vss.jsonl',
    'data/raw/vss_full/vss_cs_candidates.jsonl', 'data/raw/vss_full/vss_cs_pre_rebuild.jsonl',
    'data/raw/vss_full/vss_cs_clean_dom.jsonl', 'data/raw/vivos/train_vivos.jsonl',
    'data/raw/vivos/b5_vivos.jsonl', 'data/raw/cv/train_cv.jsonl', 'data/raw/cv/b5_cv.jsonl',
    'data/raw/ggroup/b1_candidates.jsonl', 'data/raw/ggroup/b5_ggroup.jsonl',
    'data/raw/consensus/b2_consensus_candidates.jsonl', 'data/raw/consensus/b5_consensus_product.jsonl',
    'data/raw/consensus/disputed_candidates.jsonl', 'data/raw/confusable/train_confusable.jsonl',
    'data/raw/confusable/b5_confusable.jsonl',
]
DURATION_EDGES = [0, 1, 2, 4, 6, 8, 10, 15, 20, 30, float('inf')]
ISSUES = []
PROVENANCE = {}


def fraction(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def audio_path(value, manifest):
    value = str(value or '')
    if not value:
        return ''
    mounted = re.match(r'^/mnt/([a-zA-Z])/(.*)$', value)
    if mounted:
        value = f'{mounted[1].upper()}:/{mounted[2]}' if os.name == 'nt' else value
    elif re.match(r'^[a-zA-Z]:[/\\]', value) and os.name != 'nt':
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return str(path.resolve())


def load_manifest(relative, catalog):
    path = ROOT / relative
    rows = []
    invalid = 0
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                ISSUES.append({'view': relative, 'id': str(number), 'issue': 'invalid_json', 'detail': ''})
                continue
            text = row.get('text_tagged') or row.get('text') or ''
            tokens, spans = normalize_ref(text, catalog)
            row['_duration'] = float(row.get('duration') or 0)
            row['_path'] = audio_path(row.get('audio'), path)
            row['_text'] = norm_text(text, catalog)
            row['_spans'] = [' '.join(term) for term in spans]
            row['_tokens'] = len(tokens)
            row['_en_tokens'] = sum(token.span >= 0 for token in tokens)
            row['_unit'] = row.get('speaker_cluster') or row.get('speaker') or ''
            row['_dataset'] = str(row.get('dataset') or 'unknown').split(':')[0]
            row['_tag_error'] = tagged_well_formed(text)
            positions = []
            for index in range(len(spans)):
                indices = [i for i, token in enumerate(tokens) if token.span == index]
                positions.append('đầu câu' if indices[0] == 0 else 'cuối câu' if indices[-1] == len(tokens) - 1 else 'giữa câu')
            row['_positions'] = positions
            rows.append(row)
    PROVENANCE[relative] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
                            'modified_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                            'rows': len(rows), 'invalid_json': invalid}
    return rows


def distribution(rows, field, cs_only=False):
    counts, seconds = Counter(), Counter()
    for row in rows:
        if cs_only and not row['_spans']:
            continue
        key = row.get(field) or 'unknown'
        counts[key] += 1
        seconds[key] += row['_duration']
    total = sum(seconds.values())
    return [{'value': key, 'rows': counts[key], 'hours': seconds[key] / 3600,
             'hours_pct': fraction(seconds[key], total)} for key in sorted(counts, key=lambda x: (-seconds[x], str(x)))]


def inventory_manifest(relative):
    path = ROOT / relative
    digest = hashlib.sha256()
    count = 0
    seconds = 0
    cs_count = 0
    invalid = 0
    with path.open('rb') as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                ISSUES.append({'view': relative, 'id': str(number), 'issue': 'invalid_json', 'detail': ''})
                continue
            count += 1
            seconds += float(row.get('duration') or 0)
            cs_count += bool(re.search(r'<en>\s*\S.*?</en>', row.get('text_tagged') or '', re.S))
    PROVENANCE[relative] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
                            'rows': count, 'invalid_json': invalid,
                            'modified_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}
    return {'path': relative, 'rows': count, 'hours': seconds / 3600, 'cs_rows': cs_count, 'audio_check': 'manifest_only'}


def summarize(rows, key, label, source, catalog):
    durations = [row['_duration'] for row in rows]
    cs = [row for row in rows if row['_spans']]
    terms = Counter(term for row in rows for term in row['_spans'])
    term_speakers, term_contexts = defaultdict(set), defaultdict(set)
    for row in rows:
        for term in row['_spans']:
            if row['_unit']:
                term_speakers[term].add(row['_unit'])
            term_contexts[term].add(row['_text'])
    bins = Counter()
    for value in durations:
        for index, (low, high) in enumerate(zip(DURATION_EDGES, DURATION_EDGES[1:])):
            if low <= value < high:
                bins[index] += 1
                break
    seconds = sum(durations)
    ids = Counter(row.get('id') for row in rows)
    paths = Counter(row['_path'] for row in rows if row['_path'])
    texts = Counter(row['_text'] for row in rows if row['_text'])
    speaker_seconds = Counter()
    for row in rows:
        if row['_unit']:
            speaker_seconds[row['_unit']] += row['_duration']
    token_total = sum(row['_tokens'] for row in rows)
    english_total = sum(row['_en_tokens'] for row in rows)
    issues = Counter()
    required = ['id', 'audio', 'duration', 'speaker', 'source']
    for row in rows:
        failures = []
        if any(row.get(field) in (None, '') for field in required):
            failures.append('missing_required_field')
        if row['_duration'] <= 0 or not math.isfinite(row['_duration']):
            failures.append('invalid_duration')
        if row['_duration'] < 1 or row['_duration'] > 30:
            failures.append('duration_outside_1_30s')
        if row['_tag_error']:
            failures.append('invalid_en_tags')
        if not row['_text']:
            failures.append('empty_text')
        if key.startswith('train') and is_test_only(row):
            failures.append('test_only_in_train')
        if key == 'dev_vi' and row.get('split') == 'train':
            failures.append('dev_file_split_field_train')
        if key == 'bench' and row.get('split') == 'B5' and row['_spans']:
            failures.append('b5_with_en_span')
        for failure in failures:
            issues[failure] += 1
            ISSUES.append({'view': key, 'id': row.get('id'), 'issue': failure, 'detail': row['_tag_error'] or ''})
    levels = Counter(span_meta(row, index).get('viet_level') or 'chưa gán'
                     for row in rows for index in range(len(row['_spans'])))
    categories = Counter()
    for row in rows:
        overrides = row.get('term_category') or {}
        for term in row['_spans']:
            entry = catalog.terms.get(tuple(term.split()))
            category = overrides.get(term) or (entry.category if entry and entry.category else 'chưa có trong catalog')
            categories[category] += 1
    return {
        'key': key, 'label': label, 'source': source, 'rows': len(rows), 'hours': seconds / 3600,
        'cs_rows': len(cs), 'cs_hours': sum(row['_duration'] for row in cs) / 3600,
        'cs_row_pct': fraction(len(cs), len(rows)), 'en_tokens': english_total,
        'tokens': token_total, 'en_token_pct': fraction(english_total, token_total),
        'spans': sum(terms.values()), 'distinct_terms': len(terms),
        'unique_ids': len(ids), 'duplicate_id_excess': sum(count - 1 for count in ids.values()),
        'unique_audio_paths': len(paths), 'duplicate_audio_path_excess': sum(count - 1 for count in paths.values()),
        'unique_normalized_texts': len(texts), 'duplicate_text_excess': sum(count - 1 for count in texts.values()),
        'speaker_ids': len({row.get('speaker') for row in rows if row.get('speaker')}),
        'speaker_units': len(speaker_seconds), 'sources': len({row.get('source') for row in rows if row.get('source')}),
        'top1_speaker_hours_pct': fraction(sum(value for _, value in speaker_seconds.most_common(1)), seconds),
        'top10_speaker_hours_pct': fraction(sum(value for _, value in speaker_seconds.most_common(10)), seconds),
        'top_speakers': [{'unit': unit, 'hours': value / 3600, 'pct': fraction(value, seconds)}
                         for unit, value in speaker_seconds.most_common(10)],
        'duration': {'min': min(durations, default=0), 'p10': quantile(durations, .1),
                     'median': quantile(durations, .5), 'p90': quantile(durations, .9),
                     'p95': quantile(durations, .95), 'max': max(durations, default=0),
                     'mean': statistics.mean(durations) if durations else 0},
        'duration_bins': [{'label': f'{low:g}–{high:g} s' if math.isfinite(high) else f'≥ {low:g} s',
                           'count': bins[index]} for index, (low, high) in enumerate(zip(DURATION_EDGES, DURATION_EDGES[1:]))],
        'datasets': distribution(rows, '_dataset'), 'domains': distribution(rows, 'domain'),
        'cs_domains': distribution(rows, 'domain', cs_only=True),
        'region': distribution(rows, 'region'), 'gender': distribution(rows, 'gender'),
        'channel': distribution(rows, 'channel'), 'label_source': distribution(rows, 'label_source'),
        'synthetic_rows': sum(bool(row.get('synthetic')) for row in rows),
        'split_values': dict(Counter(str(row.get('split') or 'unknown') for row in rows)),
        'review_values': dict(Counter(str(row.get('review') or 'missing') for row in rows)),
        'span_position': dict(Counter(position for row in rows for position in row['_positions'])),
        'viet_level': dict(levels), 'categories': dict(categories),
        'term_buckets': [{'label': name, 'count': sum(low <= value <= high for value in terms.values())}
                         for name, low, high in [('1 lần', 1, 1), ('2 lần', 2, 2), ('3–5 lần', 3, 5), ('6–20 lần', 6, 20), ('>20 lần', 21, float('inf'))]],
        'terms': [{'term': term, 'occurrences': count, 'speaker_units': len(term_speakers[term]),
                   'contexts': len(term_contexts[term]),
                   'reserved_b3': bool(catalog.terms.get(tuple(term.split())) and catalog.terms[tuple(term.split())].reserved_b3)}
                  for term, count in terms.most_common()],
        'issues': dict(issues),
    }


def inspect_audio_file(filename):
    path = Path(filename)
    result = {'path': filename, 'exists': path.is_file(), 'sha256': None}
    if not result['exists']:
        return result
    result['bytes'] = path.stat().st_size
    try:
        result['sha256'] = sha256_file(path)
        if path.suffix.lower() == '.wav':
            result.update(wav_metadata(path))
    except (OSError, wave.Error, EOFError) as error:
        result['header_error'] = type(error).__name__
    return result


def wav_metadata(path):
    with wave.open(str(path), 'rb') as audio:
        frames = audio.getnframes()
        result = {'header_duration': frames / audio.getframerate(), 'sample_rate': audio.getframerate(),
                  'channels': audio.getnchannels(), 'zero_frames': frames == 0}
        if frames:
            silent = True
            silence_byte = b'\x80' if audio.getsampwidth() == 1 else b'\x00'
            for chunk in iter(lambda: audio.readframes(16000), b''):
                if chunk != silence_byte * len(chunk):
                    silent = False
                    break
            result['digital_silence'] = silent
        return result


def audio_sample(row):
    result = {'path': row['_path'], 'id': row.get('id'), 'dataset': row['_dataset'],
              'duration_manifest': row['_duration']}
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        probe = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'a:0',
                                '-show_entries', 'stream=sample_rate,channels,duration:format=duration',
                                '-of', 'json', row['_path']], capture_output=True, timeout=15, creationflags=flags)
        if probe.returncode:
            result['error'] = 'ffprobe_failed'
            return result
        metadata = json.loads(probe.stdout)
        stream = metadata.get('streams', [{}])[0]
        result['sample_rate'] = int(stream['sample_rate'])
        result['channels'] = stream.get('channels')
        result['duration_actual'] = float(stream.get('duration') or metadata.get('format', {}).get('duration') or 0)
        decode = subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-i', row['_path'], '-t', '30',
                                 '-f', 's16le', '-ac', '1', '-ar', '16000', '-'], capture_output=True,
                                timeout=20, creationflags=flags)
        if decode.returncode or not decode.stdout:
            result['error'] = 'decode_failed'
            return result
        samples = array.array('h')
        samples.frombytes(decode.stdout)
        if sys.byteorder != 'little':
            samples.byteswap()
        peak = max(abs(value) for value in samples) / 32768
        mean_square = sum(value * value for value in samples) / len(samples) / (32768 ** 2)
        result.update(decoded_seconds=len(samples) / 16000,
                      rms_dbfs=10 * math.log10(max(mean_square, 1e-12)),
                      peak_dbfs=20 * math.log10(max(peak, 1e-12)),
                      near_full_scale_pct=fraction(sum(abs(value) >= 32700 for value in samples), len(samples)))
        frame_size = 320
        quiet = 0
        frames = len(samples) // frame_size
        for index in range(frames):
            frame = samples[index * frame_size:(index + 1) * frame_size]
            energy = sum(value * value for value in frame) / frame_size / (32768 ** 2)
            quiet += energy < 10 ** (-45 / 10)
        result['quiet_frame_pct_minus45dbfs'] = fraction(quiet, frames)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
        result['error'] = type(error).__name__
    return result


def compare_sets(left, right, left_name, right_name, audio_audit):
    output = {'left': left_name, 'right': right_name}
    fields = {'id': 'id', 'speaker': 'speaker', 'speaker_unit': '_unit', 'source': 'source',
              'normalized_text': '_text', 'audio_path': '_path'}
    for label, field in fields.items():
        left_values = {row.get(field) for row in left if row.get(field)}
        right_values = {row.get(field) for row in right if row.get(field)}
        intersection = left_values & right_values
        hits = [row for row in right if row.get(field) and row.get(field) in intersection]
        output[label] = {'shared_values': len(intersection), 'right_rows': len(hits),
                         'right_hours': sum(row['_duration'] for row in hits) / 3600}
        if hits and label in ['id', 'source', 'normalized_text', 'audio_path']:
            for row in hits:
                ISSUES.append({'view': right_name, 'id': row.get('id'),
                               'issue': f'overlap_{label}_with_{left_name}', 'detail': str(row.get(field))})
    left_hash = {audio_audit[row['_path']].get('sha256') for row in left if row['_path'] in audio_audit}
    left_hash.discard(None)
    hits = [row for row in right if audio_audit.get(row['_path'], {}).get('sha256') in left_hash]
    output['audio_sha256'] = {'shared_values': len({audio_audit[row['_path']]['sha256'] for row in hits}),
                              'right_rows': len(hits), 'right_hours': sum(row['_duration'] for row in hits) / 3600,
                              'silent_right_rows': sum(bool(audio_audit[row['_path']].get('digital_silence')) for row in hits)}
    for row in hits:
        ISSUES.append({'view': right_name, 'id': row.get('id'), 'issue': f'overlap_audio_sha256_with_{left_name}', 'detail': row['_path']})
    return output


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attach_audio_summary(rows, summary, audit, key):
    missing = [row for row in rows if not audit.get(row['_path'], {}).get('exists')]
    mismatched = [row for row in rows if 'header_duration' in audit.get(row['_path'], {})
                  and abs(audit[row['_path']]['header_duration'] - row['_duration']) > .1]
    digests = Counter(audit[row['_path']]['sha256'] for row in rows if audit.get(row['_path'], {}).get('sha256'))
    groups = defaultdict(list)
    for row in rows:
        digest = audit.get(row['_path'], {}).get('sha256')
        if digest:
            groups[digest].append(row)
    conflicts = [group for group in groups.values() if len({row['_text'] for row in group}) > 1]
    summary['audio'] = {
        'missing_rows': len(missing), 'missing_hours': sum(row['_duration'] for row in missing) / 3600,
        'unique_content_hashes': len(digests), 'duplicate_content_excess': sum(count - 1 for count in digests.values()),
        'unique_content_hours': sum(max(row['_duration'] for row in group) for group in groups.values()) / 3600,
        'hash_label_conflict_groups': len(conflicts),
        'wav_headers_checked_rows': sum('header_duration' in audit.get(row['_path'], {}) for row in rows),
        'wav_duration_mismatch_rows': len(mismatched),
        'digital_silence_rows': sum(bool(audit.get(row['_path'], {}).get('digital_silence')) for row in rows),
        'zero_frame_rows': sum(bool(audit.get(row['_path'], {}).get('zero_frames')) for row in rows),
        'sample_rates_wav': dict(Counter(audit[row['_path']]['sample_rate'] for row in rows if 'sample_rate' in audit.get(row['_path'], {}))),
    }
    for row in missing:
        ISSUES.append({'view': key, 'id': row.get('id'), 'issue': 'missing_audio', 'detail': row['_path']})
    for row in mismatched:
        ISSUES.append({'view': key, 'id': row.get('id'), 'issue': 'wav_duration_mismatch_gt_0.1s',
                       'detail': f"manifest={row['_duration']} header={audit[row['_path']]['header_duration']}"})
    for row in rows:
        if audit.get(row['_path'], {}).get('digital_silence'):
            ISSUES.append({'view': key, 'id': row.get('id'), 'issue': 'wav_all_digital_silence', 'detail': row['_path']})
    for group in conflicts:
        for row in group:
            ISSUES.append({'view': key, 'id': row.get('id'), 'issue': 'same_audio_different_normalized_text', 'detail': row['_text']})


def analyze_lhotse(datasets, summaries, catalog, audit, overlaps, coverage):
    manifests = []
    group_stats = []
    row_index = {row['id']: row for row in datasets['train_v1']}
    kept_ids = set()
    pool_rows = []
    for path in sorted((ROOT / 'data/e3_v1/lhotse').glob('*_cuts_raw.jsonl.gz')):
        group = path.name.removesuffix('_cuts_raw.jsonl.gz')
        originals = []
        cut_hours = 0
        cut_count = 0
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            for line in source:
                cut = json.loads(line)
                cut_count += 1
                cut_hours += cut['duration'] / 3600
                recording_id = cut['recording']['id']
                if recording_id.endswith(('_sp0.9', '_sp1.1')):
                    continue
                rid = recording_id.removeprefix(group + '-')
                if group not in ['dev_cs', 'dev_vi']:
                    expected = row_index[rid]
                    if norm_text(cut['supervisions'][0]['text'], catalog) != expected['_text']:
                        raise ValueError(f'Cut text differs from curated text: {rid}')
                    if audio_path(cut['recording']['sources'][0]['source'], path) != expected['_path']:
                        raise ValueError(f'Cut audio differs from curated audio: {rid}')
                originals.append((rid, cut['duration']))
        relative = path.relative_to(ROOT).as_posix()
        PROVENANCE[relative] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                                'rows': cut_count, 'invalid_json': 0,
                                'modified_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}
        manifests.append(relative)
        group_stats.append({'group': group, 'base_rows': len(originals), 'base_hours': sum(duration for _, duration in originals) / 3600,
                            'augmented_cuts': cut_count, 'augmented_hours': cut_hours})
        if group in ['dev_cs', 'dev_vi']:
            source_rows = {row['id']: row for row in datasets[group]}
            rows = [source_rows[rid] for rid, _ in originals]
            key = group + '_pool'
            datasets[key] = rows
            summaries[key] = summarize(rows, key, ('Dev CS' if group == 'dev_cs' else 'Dev Việt') + ' · cuts cho loss', relative, catalog)
            attach_audio_summary(rows, summaries[key], audit, key)
        else:
            for rid, duration in originals:
                if rid not in row_index:
                    raise ValueError(f'Cut ID is missing from curated train: {rid}')
                row = dict(row_index[rid])
                row['_duration'] = duration
                row['_group'] = group
                kept_ids.add(rid)
                pool_rows.append(row)
    datasets['train_pool'] = pool_rows
    summaries['train_pool'] = summarize(pool_rows, 'train_pool', 'Pool train v1 · sau Lhotse', manifests, catalog)
    attach_audio_summary(pool_rows, summaries['train_pool'], audit, 'train_pool')
    for other in ['dev_cs', 'dev_vi', 'bench', 'dev_cs_pool', 'dev_vi_pool']:
        overlaps.append(compare_sets(pool_rows, datasets[other], 'train_pool', other, audit))
    temporary = OUT / '.train_pool_frequency.jsonl'
    try:
        write_jsonl(str(temporary), pool_rows)
        terms = {tuple(term.split()) for row in datasets['bench'] for term in row['_spans']}
        frequencies = train_term_freq([str(temporary)], catalog, terms)
    finally:
        temporary.unlink(missing_ok=True)
    for split in ['B1', 'B2', 'B3', 'main_B1+B2']:
        rows = [row for row in datasets['bench'] if row['split'] in (['B1', 'B2'] if split == 'main_B1+B2' else [split])]
        terms = Counter(term for row in rows for term in row['_spans'])
        buckets = Counter()
        reserved_seen = []
        for term, count in terms.items():
            real, synthetic = frequencies.get(term, (0, 0))
            bucket = 'unseen' if real == 0 and synthetic == 0 else 'synth_only' if real == 0 else 'real_1-4' if real < 5 else 'real_5+'
            buckets[bucket] += count
            entry = catalog.terms.get(tuple(term.split()))
            if entry and entry.reserved_b3 and real + synthetic:
                reserved_seen.append(term)
        coverage['train_pool/' + split] = {'spans': sum(terms.values()), 'terms': len(terms), 'buckets': dict(buckets),
                                          'unseen_terms': [{'term': term, 'bench_occurrences': count} for term, count in terms.most_common()
                                                           if frequencies.get(term, (0, 0)) == (0, 0)],
                                          'seen_distinct_terms': sum(frequencies.get(term, (0, 0)) != (0, 0) for term in terms),
                                          'reserved_b3_seen': reserved_seen,
                                          'reserved_b3_terms': sum(bool(catalog.terms.get(tuple(term.split())) and catalog.terms[tuple(term.split())].reserved_b3) for term in terms)}
    dropped = [row for row in datasets['train_v1'] if row['id'] not in kept_ids]
    write_csv('lhotse_groups.csv', group_stats)
    write_csv('curated_rows_not_in_train_pool.csv', [{'id': row['id'], 'dataset': row['_dataset'], 'source': row.get('source'),
                                                    'duration': row['_duration'], 'audio': row['_path']} for row in dropped])
    write_csv('training_pool_rows.csv', [{'id': row['id'], 'group': row['_group'], 'source': row.get('source'),
                                         'duration': row['_duration'], 'audio': row['_path']} for row in pool_rows])
    mix = json.loads((ROOT / 'data/e3_v1/lhotse/train_mix_cuts.jsonl.gz.json').read_text(encoding='utf-8'))
    mix_exposure = analyze_mix([{'id': row['id'], 'group': row['_group']} for row in pool_rows])
    return {'groups': group_stats, 'removed_from_curated': len(dropped),
            'removed_hours': sum(row['_duration'] for row in dropped) / 3600,
            'mix_plan': mix, 'mix_exposure': mix_exposure,
            'dev_loss_vs_decode': {'dev_cs': [len(datasets['dev_cs_pool']), len(datasets['dev_cs'])],
                                                  'dev_vi': [len(datasets['dev_vi_pool']), len(datasets['dev_vi'])]}}


def analyze_mix(pool_rows):
    index = {row['group'] + '-' + row['id']: row for row in pool_rows}
    exposures = Counter()
    hours = Counter()
    counts = Counter()
    path = ROOT / 'data/e3_v1/lhotse/train_mix_cuts.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        for line in source:
            cut = json.loads(line)
            base = re.sub(r'_sp(?:0\.9|1\.1)$', '', cut['recording']['id'])
            row = index[base]
            exposures[base] += 1
            hours[row['group']] += cut['duration'] / 3600
            counts[row['group']] += 1
    by_group = defaultdict(list)
    for base, row in index.items():
        by_group[row['group']].append(exposures[base])
    rows = [{'group': group, 'base_rows': len(values), 'cuts_in_mix': counts[group], 'hours_in_mix': hours[group],
             'mean_uses_per_original': statistics.mean(values), 'min_uses': min(values),
             'median_uses': statistics.median(values), 'max_uses': max(values)} for group, values in sorted(by_group.items())]
    write_csv('mix_exposure.csv', [{'id': row['id'], 'group': row['group'], 'uses_in_mix': exposures[base]} for base, row in index.items()])
    relative = path.relative_to(ROOT).as_posix()
    PROVENANCE[relative] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                            'rows': sum(counts.values()), 'invalid_json': 0,
                            'modified_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}
    return {'groups': rows, 'cuts': sum(counts.values()), 'hours': sum(hours.values()),
            'base_rows_used': sum(value > 0 for value in exposures.values())}


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(str(CATALOG))
    datasets = {}
    summaries = {}
    for key, (label, relative) in ACTIVE.items():
        rows = load_manifest(relative, catalog)
        datasets[key] = rows
        summaries[key] = summarize(rows, key, label, relative, catalog)
        print(key, len(rows), round(summaries[key]['hours'], 3), 'hours', flush=True)
    datasets['train_input'] = [row for relative in INPUTS for row in load_manifest(relative, catalog)]
    summaries['train_input'] = summarize(datasets['train_input'], 'train_input', 'Đầu vào curate v1', INPUTS, catalog)
    for split in ['B1', 'B2', 'B3', 'B5']:
        key = 'bench_' + split
        datasets[key] = [row for row in datasets['bench'] if row.get('split') == split]
        summaries[key] = summarize(datasets[key], key, 'Bench ' + split, ACTIVE['bench'][1], catalog)

    inventory = [inventory_manifest(relative) for relative in INVENTORY]
    print('Source inventory complete', flush=True)

    active_rows = [row for key in ACTIVE for row in datasets[key]]
    paths = sorted({row['_path'] for row in active_rows if row['_path']})
    print('Checking and hashing', len(paths), 'unique active audio files', flush=True)
    audit = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, result in enumerate(pool.map(inspect_audio_file, paths), 1):
            audit[result['path']] = result
            if index % 5000 == 0:
                print('Audio files checked', index, '/', len(paths), flush=True)
    for key in ACTIVE:
        attach_audio_summary(datasets[key], summaries[key], audit, key)

    overlaps = []
    for train in ['train_v1', 'train_v1b']:
        for other in ['dev_cs', 'dev_vi', 'bench']:
            overlaps.append(compare_sets(datasets[train], datasets[other], train, other, audit))
    overlaps.append(compare_sets(datasets['dev_cs'] + datasets['dev_vi'], datasets['bench'], 'dev_combined', 'bench', audit))

    all_bench_terms = {tuple(term.split()) for row in datasets['bench'] for term in row['_spans']}
    coverage = {}
    for train in ['train_v1', 'train_v1b']:
        frequencies = train_term_freq([str(ROOT / ACTIVE[train][1])], catalog, all_bench_terms)
        for split in ['B1', 'B2', 'B3', 'main_B1+B2']:
            rows = [row for row in datasets['bench'] if row['split'] in (['B1', 'B2'] if split == 'main_B1+B2' else [split])]
            buckets = Counter()
            terms = Counter(term for row in rows for term in row['_spans'])
            for term, count in terms.items():
                real, synthetic = frequencies.get(term, (0, 0))
                bucket = 'unseen' if real == 0 and synthetic == 0 else 'synth_only' if real == 0 else 'real_1-4' if real < 5 else 'real_5+'
                buckets[bucket] += count
            coverage[train + '/' + split] = {'spans': sum(terms.values()), 'terms': len(terms), 'buckets': dict(buckets),
                                           'unseen_terms': [{'term': term, 'bench_occurrences': count}
                                                            for term, count in terms.most_common() if frequencies.get(term, (0, 0)) == (0, 0)],
                                           'seen_distinct_terms': sum(frequencies.get(term, (0, 0)) != (0, 0) for term in terms)}

    rng = random.Random(0)
    selected = {}
    for key in ['train_v1', 'dev_cs', 'dev_vi', 'bench']:
        groups = defaultdict(list)
        for row in datasets[key]:
            if row['_duration'] > 0 and audit.get(row['_path'], {}).get('exists'):
                groups[(row['_dataset'], row.get('split') if key == 'bench' else '')].append(row)
        for group, rows in sorted(groups.items()):
            ordered = sorted(rows, key=lambda row: str(row.get('id')))
            for row in rng.sample(ordered, min(8, len(ordered))):
                selected.setdefault(row['_path'], row)
    samples = []
    if shutil.which('ffprobe') and shutil.which('ffmpeg'):
        print('Decoding stratified audio sample', len(selected), 'files', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            samples = list(pool.map(audio_sample, selected.values()))

    lhotse = analyze_lhotse(datasets, summaries, catalog, audit, overlaps, coverage)

    flat = [dict(view=key, **{field: value for field, value in summary.items()
                            if isinstance(value, (int, float, str)) and field != 'key'}) for key, summary in summaries.items()]
    write_csv('dataset_overview.csv', flat)
    write_csv('term_frequency.csv', [dict(view=key, **term) for key, summary in summaries.items() for term in summary['terms']])
    write_csv('overlap.csv', [{'left': overlap['left'], 'right': overlap['right'], 'criterion': criterion, **value}
                              for overlap in overlaps for criterion, value in overlap.items() if isinstance(value, dict)])
    write_csv('audio_sample.csv', samples)
    write_csv('audit_issues.csv', ISSUES)
    write_csv('source_inventory.csv', inventory)
    write_csv('audio_files.csv', audit.values())
    v1_ids = {row.get('id') for row in datasets['train_v1']}
    v1b_ids = {row.get('id') for row in datasets['train_v1b']}
    result = {'generated_at': dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat(),
              'catalog': str(CATALOG.relative_to(ROOT)), 'catalog_sha256': sha256_file(CATALOG),
              'views': summaries, 'inventory': inventory, 'overlaps': overlaps, 'coverage': coverage,
              'audio_sample': samples, 'audio_files_checked': len(audit),
              'audio_files_missing': sum(not value['exists'] for value in audit.values()),
              'audio_content_bytes_hashed': sum(value.get('bytes', 0) for value in audit.values() if value.get('sha256')),
              'variant': {'v1_only_ids': len(v1_ids - v1b_ids), 'v1b_only_ids': len(v1b_ids - v1_ids), 'shared_ids': len(v1_ids & v1b_ids)},
              'provenance': PROVENANCE, 'lhotse': lhotse,
              'method': {'audio_sample_seed': 0, 'sample_per_dataset_and_role': 8,
                         'decode_limit_seconds': 30, 'quiet_frame_dbfs': -45, 'frame_seconds': .02,
                         'hash': 'sha256_of_file_bytes', 'duration_warning_seconds': .1}}
    result['dataset_lineage'] = collect_lineage(result)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Collection complete', json.dumps({'audio_files': len(audit), 'missing': result['audio_files_missing'],
                                           'sample': len(samples), 'issues': len(ISSUES)}), flush=True)
    return result


DOMAIN_LABELS = {'medical': 'Y khoa', 'tech': 'Công nghệ', 'edu': 'Giáo dục', 'pubadmin': 'Hành chính',
                 'product': 'Sản phẩm', 'finance': 'Tài chính', 'business': 'Kinh doanh', 'read': 'Giọng đọc',
                 'other': 'Chưa xác định', 'daily': 'Đời sống', 'entertainment': 'Giải trí', 'unknown': 'Thiếu nhãn'}


def number(value, digits=0):
    return f'{value:,.{digits}f}'.replace(',', ' ') if value is not None else '—'


def html_table(headers, rows):
    header = ''.join(f'<th scope="col">{escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def bar_chart(title, values, unit='', description='', color='--s1'):
    if not values:
        return f'<figure><h3>{escape(title)}</h3><p class="note">Tập này chưa có mục để vẽ.</p></figure>'
    maximum = max([value for _, value in values] + [1e-9])
    rows = []
    for label, value in values:
        display = number(value, 2 if isinstance(value, float) else 0) + unit
        rows.append('<div class="bar-row"><div class="bar-label">'
                    f'<span>{escape(str(label))}</span><b>{escape(display)}</b></div>'
                    f'<div class="bar-track" aria-hidden="true"><span style="width:{100 * value / maximum:.5f}%;'
                    f'background:var({color})"></span></div></div>')
    chart = f'<div class="chart-bars" role="group" aria-label="{escape(title, quote=True)}">' + ''.join(rows) + '</div>'
    return f'<figure><h3>{escape(title)}</h3>{chart}<figcaption>{description}</figcaption></figure>'


def view_panel(view):
    key = view['key']
    cards = [('Câu / dòng', number(view['rows'])), ('Giờ ghi trong manifest', number(view['hours'], 2)),
             ('Câu có đoạn Anh', number(view['cs_rows'])), ('Thuật ngữ khác nhau', number(view['distinct_terms'])),
             ('Độ dài trung vị', number(view['duration']['median'], 2) + ' giây')]
    kpis = '<div class="kpis">' + ''.join(f'<div class="kpi"><strong>{value}</strong><span>{label}</span></div>' for label, value in cards) + '</div>'
    plots = [
        bar_chart('Thời lượng theo nguồn dữ liệu', [(row['value'], row['hours']) for row in view['datasets']], ' h', 'Mỗi thanh tính tổng thời lượng của tất cả câu thuộc nguồn đó.'),
        bar_chart('Độ dài các đoạn ghi âm', [(row['label'], row['count']) for row in view['duration_bins']], ' câu', 'Các khoảng là trái đóng, phải mở. Ví dụ 2–4 giây gồm 2 giây và các giá trị dưới 4 giây.'),
        bar_chart('Lĩnh vực trong toàn bộ tập', [(DOMAIN_LABELS.get(row['value'], row['value']), row['hours']) for row in view['domains']], ' h', 'Tính cả câu tiếng Việt thuần. “Chưa xác định” là nhãn other, có thể chứa nhiều chủ đề.'),
        bar_chart('Lĩnh vực của các câu có đoạn Anh', [(DOMAIN_LABELS.get(row['value'], row['value']), row['hours']) for row in view['cs_domains']], ' h', 'Đơn vị là giờ của toàn câu có nhãn &lt;en&gt;, gồm cả phần Việt của câu.'),
        bar_chart('Thuật ngữ xuất hiện nhiều nhất', [(row['term'], row['occurrences']) for row in view['terms'][:15]], ' lần', 'Tính số lần xuất hiện trong nhãn, sau chuẩn hoá strict theo catalog.'),
        bar_chart('Độ hiếm của thuật ngữ', [(row['label'], row['count']) for row in view['term_buckets']], ' thuật ngữ', 'Mỗi thuật ngữ được đếm một lần ở khoảng tần suất tương ứng. Nhiều từ chỉ có một lần làm điểm nhóm từ đó kém ổn định.'),
        bar_chart('Mười mã người nói/cụm chiếm nhiều giờ nhất', [(row['unit'][:25], row['hours']) for row in view['top_speakers']], ' h', 'Nhãn dài được rút gọn trong hình. Tên đầy đủ có trong summary.json. Một số mã đại diện video hoặc cụm ước lượng.'),
        bar_chart('Vị trí đoạn Anh trong câu', list(view['span_position'].items()), ' đoạn', 'Đầu hoặc cuối dựa trên vị trí sau chuẩn hoá. Câu hoàn toàn tiếng Anh được xếp vào đầu câu.'),
    ]
    metadata_rows = []
    for field, label in [('region', 'Vùng giọng'), ('gender', 'Giới tính'), ('channel', 'Kênh âm thanh')]:
        for row in view[field]:
            value = {'unk': 'Chưa rõ', 'u': 'Chưa rõ', 'unknown': 'Thiếu nhãn', 'north': 'Bắc', 'central': 'Trung',
                     'south': 'Nam', 'f': 'Nữ', 'm': 'Nam', 'wideband': 'Wideband', 'phone8k': 'Điện thoại 8 kHz'}.get(row['value'], row['value'])
            metadata_rows.append([label, value, number(row['rows']), number(row['hours'], 2), number(row['hours_pct'], 1) + '%'])
    duration = view['duration']
    stats = html_table(['Nhỏ nhất', 'P10', 'Trung vị', 'P90', 'P95', 'Lớn nhất'],
                       [[number(duration[name], 2) + ' s' for name in ['min', 'p10', 'median', 'p90', 'p95', 'max']]])
    notes = (f'<p class="explain">{number(view["en_token_pct"], 2)}% đơn vị chữ sau chuẩn hoá được gắn là tiếng Anh. '
             f'{number(view["cs_row_pct"], 1)}% số câu có ít nhất một đoạn Anh. '
             f'Mã/cụm lớn nhất chiếm {number(view["top1_speaker_hours_pct"], 2)}% số giờ, '
             f'mười mã/cụm lớn nhất chiếm {number(view["top10_speaker_hours_pct"], 2)}%.</p>')
    return (f'<div class="view-panel" data-view="{key}"' + ('' if key == 'train_pool' else ' hidden') + '>'
            + kpis + notes + '<div class="chart-grid">' + ''.join(plots) + '</div>'
            + '<h3>Tóm tắt thời lượng</h3>' + stats
            + '<p class="note">P90 nghĩa là 90% số đoạn có thời lượng nhỏ hơn hoặc bằng giá trị này.</p>'
            + '<h3>Metadata hiện có</h3>' + html_table(['Thông tin', 'Giá trị', 'Câu', 'Giờ', '% giờ'], metadata_rows)
            + '<p class="note">Nhãn vùng giọng, giới tính và kênh thu được đọc từ manifest. Nhãn thiếu/chưa rõ không được suy đoán từ âm thanh.</p>'
            + '</div>')


def render_report(result):
    lineage_html = render_lineage(result['dataset_lineage'])
    views = result['views']
    train = views['train_pool']
    dev_overlap = next(row for row in result['overlaps'] if row['left'] == 'train_pool' and row['right'] == 'dev_cs')
    main_coverage = result['coverage']['train_pool/main_B1+B2']
    unseen_pct = fraction(main_coverage['buckets'].get('unseen', 0), main_coverage['spans'])
    nonmissing_sample = [row for row in result['audio_sample'] if not row.get('error')]
    retention = []
    input_sources = {row['value']: row for row in views['train_input']['datasets']}
    train_sources = {row['value']: row for row in views['train_v1']['datasets']}
    pool_sources = {row['value']: row for row in train['datasets']}
    for source, row in input_sources.items():
        kept = train_sources.get(source, {'rows': 0, 'hours': 0})
        pooled = pool_sources.get(source, {'rows': 0, 'hours': 0})
        retention.append([source, number(row['rows']), number(row['hours'], 2), number(kept['rows']),
                          number(kept['hours'], 2), number(pooled['rows']), number(pooled['hours'], 2)])

    findings = [
        ('Pool train v1 sau Lhotse', f'{number(train["rows"])} câu / {number(train["hours"], 2)} giờ',
         f'Trong đó {number(train["cs_rows"])} câu có đoạn Anh, tương ứng {number(train["cs_hours"], 2)} giờ toàn câu.'),
        ('Dev chen Anh trùng nguồn train', f'{number(dev_overlap["source"]["right_rows"])} / {number(views["dev_cs"]["rows"])} câu',
         'Trùng theo trường source. Cần tách nguồn khi đánh giá khả năng áp dụng sang giọng/ngữ cảnh mới.'),
        ('Thuật ngữ B1+B2 chưa có trong train', f'{number(unseen_pct, 1)}% số lần xuất hiện',
         f'{number(main_coverage["buckets"].get("unseen", 0))}/{number(main_coverage["spans"])} đoạn thuật ngữ thuộc từ chưa xuất hiện trong pool train v1.'),
        ('Audio trên ổ đĩa', f'{number(result["audio_files_checked"] - result["audio_files_missing"])} / {number(result["audio_files_checked"])} file',
         'Đếm đường dẫn riêng biệt của các tập chính, đã đổi đường dẫn WSL sang Windows và kiểm file thực tế.'),
    ]
    kpis = '<div class="finding-grid">' + ''.join(f'<article class="finding"><span>{escape(label)}</span><strong>{escape(value)}</strong><p>{escape(note)}</p></article>' for label, value, note in findings) + '</div>'
    overview = html_table(['Tập', 'Câu', 'Giờ theo dòng', 'Giờ sau gộp hash', 'Câu có Anh', 'Giờ câu có Anh', 'Mã/cụm nói', 'Audio riêng theo SHA256'],
                          [[views[key]['label'], number(views[key]['rows']), number(views[key]['hours'], 2),
                            number(views[key]['audio']['unique_content_hours'], 2),
                            number(views[key]['cs_rows']), number(views[key]['cs_hours'], 2), number(views[key]['speaker_units']),
                            number(views[key]['audio']['unique_content_hashes'])] for key in ['train_pool', *ACTIVE, 'dev_cs_pool', 'dev_vi_pool']])
    source_stages = (f'<div class="flow"><div><b>Đầu vào curate v1</b><strong>{number(views["train_input"]["hours"], 2)} h</strong>'
                     f'<span>{number(views["train_input"]["rows"])} dòng từ 6 manifest</span></div><span class="flow-link">lọc theo quy tắc</span>'
                     f'<div><b>Curate v1</b><strong>{number(views["train_v1"]["hours"], 2)} h</strong><span>{number(views["train_v1"]["rows"])} dòng</span></div>'
                     f'<span class="flow-link">lọc thêm khi chuyển Lhotse</span><div><b>Pool train v1</b><strong>{number(train["hours"], 2)} h</strong>'
                     f'<span>{number(train["rows"])} câu trước tăng cường tốc độ</span></div></div>')

    domain_values = []
    for key, label in [('train_pool', 'Pool train v1'), ('train_v1', 'Curate v1'), ('dev_cs', 'Dev CS'), ('bench_B1', 'B1'), ('bench_B2', 'B2'), ('bench_B3', 'B3')]:
        for row in views[key]['cs_domains']:
            domain_values.append([label, DOMAIN_LABELS.get(row['value'], row['value']), number(row['hours'], 2), number(row['hours_pct'], 1) + '%'])
    domain_plots = ''.join(bar_chart(views[key]['label'], [(DOMAIN_LABELS.get(row['value'], row['value']), row['hours_pct']) for row in views[key]['cs_domains']], '%',
                                   'Tỉ trọng tính trong số giờ của các câu có nhãn Anh ở chính tập này.') for key in ['train_pool', 'dev_cs', 'bench_B1', 'bench_B2'])

    overlap_rows = []
    for row in result['overlaps']:
        overlap_rows.append([row['left'] + ' / ' + row['right']] + [number(row[field]['right_rows']) for field in ['id', 'source', 'speaker', 'speaker_unit', 'normalized_text', 'audio_path', 'audio_sha256']])
    overlap_table = html_table(['Cặp so sánh', 'Trùng ID', 'Trùng nguồn', 'Trùng mã nói', 'Trùng mã/cụm', 'Trùng câu chữ', 'Trùng đường dẫn', 'Trùng byte audio'], overlap_rows)

    coverage_rows = []
    for train_key in ['train_pool', 'train_v1', 'train_v1b']:
        for split in ['B1', 'B2', 'B3', 'main_B1+B2']:
            data = result['coverage'][train_key + '/' + split]
            coverage_rows.append([train_key, split, number(data['spans']), number(data['terms']),
                                  number(data['buckets'].get('unseen', 0)), number(fraction(data['buckets'].get('unseen', 0), data['spans']), 1) + '%',
                                  number(data['buckets'].get('real_1-4', 0)), number(data['buckets'].get('real_5+', 0)), number(data['buckets'].get('synth_only', 0))])
    unseen_top = main_coverage['unseen_terms'][:20]
    term_plot = bar_chart('Thuật ngữ B1+B2 thường gặp nhưng chưa có trong pool train v1', [(row['term'], row['bench_occurrences']) for row in unseen_top], ' lần',
                          'Danh sách theo nhãn bench hiện tại. Cần duyệt nhãn/cách đọc trước khi quyết định tạo dữ liệu cho từng từ.', '--s2')

    quality_rows = []
    for key in ['train_pool', *ACTIVE, 'dev_cs_pool', 'dev_vi_pool']:
        view = views[key]
        audio = view['audio']
        quality_rows.append([view['label'], number(view['duplicate_id_excess']), number(view['duplicate_audio_path_excess']),
                             number(audio['duplicate_content_excess']), number(audio['hash_label_conflict_groups']),
                             number(audio.get('digital_silence_rows', 0)), number(audio.get('zero_frame_rows', 0)),
                             number(audio['missing_rows']), number(audio['wav_duration_mismatch_rows']),
                             number(view['issues'].get('invalid_en_tags', 0))])
    audio_groups = defaultdict(list)
    for row in result['audio_sample']:
        audio_groups[row['dataset']].append(row)
    audio_rows = []
    for key, rows in sorted(audio_groups.items()):
        valid = [row for row in rows if not row.get('error')]
        audio_rows.append([key, len(rows), len(valid),
                           number(statistics.median(row['rms_dbfs'] for row in valid), 1) if valid else '—',
                           number(statistics.median(row['quiet_frame_pct_minus45dbfs'] for row in valid), 1) + '%' if valid else '—',
                           sum(row['near_full_scale_pct'] > .1 for row in valid),
                           ', '.join(f'{rate} Hz × {count}' for rate, count in sorted(Counter(row['sample_rate'] for row in valid).items()))])
    audio_headlines = html_table(['Phạm vi kiểm', 'Số lượng / cách làm'], [
        ['Đường dẫn audio riêng biệt của 5 tập chính', number(result['audio_files_checked'])],
        ['File vắng mặt', number(result['audio_files_missing'])],
        ['Dữ liệu file đã tính SHA256', number(result['audio_content_bytes_hashed'] / (1024 ** 3), 2) + ' GiB'],
        ['Mẫu phân tầng đã thử giải mã', f'{len(nonmissing_sample)} / {len(result["audio_sample"])} file'],
        ['Chọn mẫu', '8 file mỗi nhóm nguồn × vai trò, seed 0, gộp trùng đường dẫn'],
        ['Phần audio thử giải mã', 'Tối đa 30 giây đầu của mỗi file mẫu'],
    ])

    unknown_rows = []
    for key in ACTIVE:
        view = views[key]
        values = []
        for field in ['region', 'gender']:
            unknown = sum(row['rows'] for row in view[field] if row['value'] in ['unk', 'u', 'unknown', ''])
            values.append(number(fraction(unknown, view['rows']), 1) + '%')
        unknown_levels = view['viet_level'].get('chưa gán', 0)
        values.append(number(fraction(unknown_levels, view['spans']), 1) + '%' if view['spans'] else 'Không có span')
        unknown_rows.append([view['label']] + values)

    inventory_table = html_table(['Manifest nguồn / giai đoạn', 'Câu', 'Giờ theo nhãn', 'Câu có Anh'],
                                 [[row['path'], number(row['rows']), number(row['hours'], 2), number(row['cs_rows'])] for row in result['inventory']])
    provenance_table = html_table(['File', 'Dòng đọc được', 'JSON hỏng', 'SHA256'],
                                  [[key, number(value['rows']), value['invalid_json'], value['sha256']] for key, value in result['provenance'].items()])
    panel_keys = ['train_pool', 'train_v1', 'train_v1b', 'dev_cs', 'dev_cs_pool', 'dev_vi', 'dev_vi_pool', 'bench', 'bench_B1', 'bench_B2', 'bench_B3', 'bench_B5', 'train_input']
    options = ''.join(f'<option value="{key}">{escape(views[key]["label"])}</option>' for key in panel_keys)
    panels = ''.join(view_panel(views[key]) for key in panel_keys)
    mix_rows = result['lhotse']['mix_exposure']['groups']
    mix_table = html_table(['Nhóm', 'Câu gốc trong pool', 'Cut trong file trộn', 'Giờ sau lấy lặp', 'Lần dùng TB / câu gốc', 'Ít nhất', 'Nhiều nhất'],
                           [[row['group'], number(row['base_rows']), number(row['cuts_in_mix']), number(row['hours_in_mix'], 2),
                             number(row['mean_uses_per_original'], 1), row['min_uses'], row['max_uses']] for row in mix_rows])
    reserved = result['coverage']['train_pool/B3']
    pool_bench = next(row for row in result['overlaps'] if row['left'] == 'train_pool' and row['right'] == 'bench')
    with (OUT / 'audit_issues.csv').open(encoding='utf-8-sig') as source:
        silent_ids = {row['id'] for row in csv.DictReader(source)
                      if row['view'] == 'train_pool' and row['issue'] == 'wav_all_digital_silence'}
    with (OUT / 'mix_exposure.csv').open(encoding='utf-8-sig') as source:
        exposures = {row['id']: int(row['uses_in_mix']) for row in csv.DictReader(source)}
    with (OUT / 'training_pool_rows.csv').open(encoding='utf-8-sig') as source:
        silent_pool = [row for row in csv.DictReader(source) if row['id'] in silent_ids]
    silent_table = html_table(['ID trong pool train', 'Nhóm', 'Giây', 'Lượt trong file trộn đã chuẩn bị'],
                              [[row['id'], row['group'], number(float(row['duration']), 2), exposures[row['id']]] for row in silent_pool])

    body = f'''<header>
<span class="eyebrow">GIPBENCH · EDA TỪ MANIFEST VÀ AUDIO HIỆN CÓ</span>
<h1>Dữ liệu đang có gì, thiếu gì và trùng ở đâu?</h1>
<p class="lead">Phân tích pool train E3 v1 thực tế sau Lhotse, manifest trước đó, biến thể v1b, dev và bench. Có biểu đồ, kiểm audio và đối chiếu từng giai đoạn.</p>
<p class="note">Tạo lúc {escape(result['generated_at'])}. Thời lượng, nhãn và thuật ngữ được tính lại từ các file đã liệt kê ở cuối báo cáo.</p>
<nav><a href="#lineage">Nguồn dataset và xử lý</a><a href="#overview">Tổng quan</a><a href="#explore">Khám phá từng tập</a><a href="#curation">Trước/sau lọc</a><a href="#domain">Lệch lĩnh vực</a><a href="#coverage">Độ phủ từ Anh</a><a href="#overlap">Trùng dữ liệu</a><a href="#quality">Chất lượng</a><a href="#audio">Audio</a><a href="#actions">Hướng xử lý</a><a href="#sources">Nguồn</a></nav>
</header>
{lineage_html}
<section id="overview"><h2>Những điểm cần thấy trước</h2>{kpis}
<div class="alert"><b>Các phát hiện cần xử lý khi chọn mô hình.</b> Pool train v1 còn {number(train['audio'].get('digital_silence_rows', 0))} dòng WAV im lặng toàn bộ. Có {dev_overlap['audio_sha256']['right_rows']} dòng dev chen Anh trùng byte với pool train, trong đó {dev_overlap['audio_sha256'].get('silent_right_rows', 0)} dòng có WAV toàn im lặng. Dev chen Anh còn {dev_overlap['source']['right_rows']} câu chung nguồn với pool.</div>
<p class="explain">Một “câu” trong báo cáo là một dòng manifest. Các bản train, dev và các giai đoạn lọc có thể chứa lại cùng audio. Các số giữa các tập dùng để so sánh, việc cộng chúng thành tổng dữ liệu riêng biệt sẽ đếm lặp.</p>
{overview}
<p class="note">“Mã/cụm nói” lấy speaker_cluster nếu có, hoặc speaker. VSS/ViMedCSS thường dùng mã video, G-Group/consensus có cụm ước lượng. Các con số này cần được hiểu là đơn vị đại diện theo metadata.</p>
<p class="note">Giờ sau gộp hash lấy một thời lượng manifest lớn nhất cho mỗi nhóm audio có cùng SHA256. Nó giúp thấy phần giờ lặp do cùng nội dung byte xuất hiện nhiều dòng. Các bản nén/cắt khác byte có thể còn trùng nội dung nói.</p>
</section>
<section id="explore"><h2>Khám phá từng tập bằng biểu đồ</h2><label for="dataset-view">Chọn tập dữ liệu cần xem</label><select id="dataset-view">{options}</select>
<p class="note mobile-hint">Các bảng nhiều cột có thể kéo ngang khi màn hình hẹp.</p>{panels}</section>
<section id="curation"><h2>Trước và sau lọc dữ liệu train</h2>{source_stages}
<p>Curate v1b có {number(views['train_v1b']['rows'])} câu / {number(views['train_v1b']['hours'], 2)} giờ, là biến thể lọc từ cùng đầu vào. Có {number(result['variant']['shared_ids'])} ID chung với curate v1 và {number(result['variant']['v1b_only_ids'])} ID thêm ở v1b. Các cut của pool v1 được đối chiếu bằng recording ID, đường dẫn audio và câu chữ sau chuẩn hoá.</p>
{html_table(['Nguồn', 'Câu vào', 'Giờ vào', 'Câu curate v1', 'Giờ curate v1', 'Câu pool Lhotse', 'Giờ pool Lhotse'], retention)}
<p class="explain">Bước Lhotse lọc thêm {number(result['lhotse']['removed_from_curated'])} câu / {number(result['lhotse']['removed_hours'], 2)} giờ từ manifest curate v1. Các dòng 0 giây ở manifest curate đã nằm ngoài pool train. Điều này giải thích sự khác nhau giữa thống kê manifest và dữ liệu sẵn sàng để trộn học.</p>
<h3>Dữ liệu sau tăng cường và lấy lặp</h3>{mix_table}
<p>File train_mix đã tạo có {number(result['lhotse']['mix_exposure']['cuts'])} cut / {number(result['lhotse']['mix_exposure']['hours'], 2)} giờ, lấy từ {number(result['lhotse']['mix_exposure']['base_rows_used'])} câu gốc. “Cut” ở đây có thể là một bản tốc độ hoặc lần dùng lặp. Số giờ này mô tả toàn file trộn đã chuẩn bị, lượng mô hình thực sự đã học còn phụ thuộc tiến độ chạy.</p>
<p class="note">Dev để tính loss trong Lhotse có {views['dev_cs_pool']['rows']} câu CS và {views['dev_vi_pool']['rows']} câu Việt. Dev được giải mã để chọn checkpoint dùng các manifest {views['dev_cs']['rows']} câu CS và {views['dev_vi']['rows']} câu Việt. Các đồ thị loss và các điểm giải mã vì vậy còn khác phạm vi câu được đo.</p></section>
<section id="domain"><h2>Train và bài kiểm tra có cùng chủ đề?</h2><p>Các biểu đồ sau cùng tính trong những câu có nhãn Anh của từng tập, giúp thấy lệch miền giữa dữ liệu học và dữ liệu cần đo.</p><div class="chart-grid">{domain_plots}</div>
<details><summary>Bảng lĩnh vực và thời lượng</summary>{html_table(['Tập', 'Lĩnh vực', 'Giờ câu có Anh', '% giờ trong tập'], domain_values)}</details>
<p class="explain">Lĩnh vực của vss_cs được suy từ từ khoá trong văn bản và nhiều câu còn là other. Điều này giới hạn độ chính xác của nhận xét theo lĩnh vực. Thời lượng ở đây tính toàn câu có từ Anh, gồm cả phần tiếng Việt.</p></section>
<section id="coverage"><h2>Từ Anh trong bench đã được gặp trong train bao nhiêu lần?</h2>
<p>Tra thuật ngữ của bench trong toàn văn bản train sau chuẩn hoá strict, dùng cùng hàm đếm với score.py. Đếm cả các lần xuất hiện trong câu train chưa gắn tag Anh. Số lần này xét dữ liệu học tiếp của dự án.</p>
{html_table(['Train', 'Bench', 'Đoạn Anh', 'Từ khác nhau', 'Chưa thấy', '% chưa thấy', 'Thật 1–4 lần', 'Thật ≥5 lần', 'Chỉ TTS'], coverage_rows)}
<p>B3 có {reserved['terms']} thuật ngữ khác nhau, trong đó {reserved['reserved_b3_terms']} thuật ngữ được catalog đánh dấu reserved_b3 để giữ riêng. Có {len(reserved['reserved_b3_seen'])} thuật ngữ reserved_b3 xuất hiện trong pool train. Các đoạn khác trong câu B3 có thể chứa từ thông dụng đã có ở train, nên toàn bộ B3 có thể có một phần từ đã thấy.</p>
{term_plot}<p class="note">“Thật” nghĩa là dòng không có synthetic=true, gồm audio người thật với nhãn người hoặc nhãn máy. Nhóm synth_only là từ chỉ có ở dòng tổng hợp. Nhãn bench còn là bản nháp nên danh sách ưu tiên cũng cần được duyệt.</p></section>
<section id="overlap"><h2>Kiểm tra mức trùng giữa train, dev và bench</h2>
{overlap_table}
<p><b>Pool train v1 sau Lhotse so với bench.</b> Có {pool_bench['source']['right_rows']} câu bench trùng nguồn, {pool_bench['normalized_text']['right_rows']} câu trùng chữ sau chuẩn hoá và {pool_bench['audio_sha256']['right_rows']} câu trùng byte audio. Manifest trước Lhotse còn các dòng mà bước chuyển này đã loại.</p>
<p class="explain">Mỗi ô là số dòng ở tập bên phải có giá trị trùng với tập bên trái. Các tiêu chí có thể cùng bắt một câu nên không cộng các cột. Trùng nguồn/giọng làm kết quả kém độc lập. Trùng câu chữ có thể là câu thông dụng được nhiều người nói, cần kiểm audio và nguồn để diễn giải.</p>
<p class="note">“Trùng byte audio” dùng SHA256 toàn bộ file. Nó phát hiện file giống byte dù khác đường dẫn. Các bản cắt, nén hoặc tăng giảm âm lượng khác nhau có thể khác hash. Các file vắng mặt nằm ngoài phép so hash.</p></section>
<section id="quality"><h2>Dữ liệu lặp, nhãn và trường thông tin còn thiếu</h2>
{html_table(['Tập', 'Dòng thừa ID', 'Dòng thừa path', 'Dòng thừa hash', 'Nhóm audio khác nhãn', 'WAV toàn im lặng', 'WAV 0 mẫu', 'Thiếu audio', 'WAV lệch >0.1s', 'Tag Anh hỏng'], quality_rows)}
<p class="explain">“Dòng thừa” là số dòng vượt quá một dòng trên mỗi nhóm trùng. Cùng audio có thể được dùng với nhiều vai trò, chẳng hạn replay và confusable. Cần xem nhãn và mục đích lấy mẫu trước khi quyết định loại dòng. Nhóm cùng hash có câu chuẩn hoá khác nhau được liệt kê để nghe kiểm.</p>
<h3>Các WAV im lặng toàn bộ còn nằm trong pool train</h3>{silent_table}
<p class="note">Số lượt lấy lặp đọc từ toàn file train_mix đã chuẩn bị. Đây là các file có khung âm thanh nhưng dữ liệu PCM đều ở mức im lặng. Các file WAV 0 mẫu là trường hợp riêng và đã được lọc khỏi pool Lhotse.</p>
<h3>Metadata ảnh hưởng khả năng phân tích</h3>{html_table(['Tập', '% câu chưa rõ vùng giọng', '% câu chưa rõ giới tính', '% đoạn Anh chưa rõ mức Việt hoá'], unknown_rows)}
<p>File dev tiếng Việt có {number(views['dev_vi']['issues'].get('dev_file_split_field_train', 0))} dòng mang trường split=train. Báo cáo xác định vai trò dev theo file thực sự được script E3 dùng. Trường split cần được thống nhất để công cụ khác đọc đúng vai trò.</p>
<p>Bench dùng nhãn v0-draft. Kiểm JSON/tag xác nhận cấu trúc, còn chất lượng câu đúng và cách phát âm cần nghe duyệt. Các chỉ số về từ Anh trong EDA phụ thuộc vào nhãn này.</p>
<p><a href="audit_issues.csv">Mở danh sách dòng cần kiểm</a></p></section>
<section id="audio"><h2>Kiểm audio thực tế</h2>{audio_headlines}
<p>Toàn bộ đường dẫn riêng của năm manifest chính được kiểm file và tính hash nếu đọc được; pool Lhotse là tập con trong phạm vi audio đó. Các WAV đọc được bằng thư viện wave được kiểm thời lượng từ header và kiểm mẫu PCM im lặng toàn bộ. File nguồn trong bảng inventory được thống kê theo manifest.</p>
<h3>Mẫu giải mã phân tầng</h3>{html_table(['Nguồn', 'File mẫu', 'Giải mã được', 'RMS trung vị dBFS', 'Khung nhỏ hơn −45 dBFS', 'File gần toàn biên >0.1%', 'Sample rate thực'], audio_rows)}
<p class="explain">RMS càng âm lớn thì âm lượng trung bình càng nhỏ. Khung nhỏ hơn −45 dBFS là phép đo năng lượng cố định trên khung 20 ms, có thể gồm khoảng nghỉ và giọng nhỏ. “Gần toàn biên” đếm mẫu PCM có trị tuyệt đối ≥32700/32768 sau giải mã về mono 16 kHz. Đây là dấu hiệu nên nghe kiểm.</p>
<p class="note">Các chỉ số âm lượng chỉ áp dụng mẫu phân tầng, tối đa 30 giây đầu mỗi file. Lấy đều theo nguồn/vai trò làm mẫu khác phân bố toàn bộ corpus. EDA chưa chấm chất lượng phát âm, độ đúng nhãn hay SNR bằng người nghe.</p>
<p><a href="audio_sample.csv">Mở bảng audio mẫu</a> · <a href="audio_files.csv">Mở bảng kiểm file/hash/header</a></p></section>
<section id="actions"><h2>Hướng xử lý từ các kết quả EDA</h2><ol>
<li><b>Tách lại dev theo nguồn/người nói.</b> Dùng bảng trùng để kiểm trước khi chọn checkpoint. Bổ sung dev hội thoại Việt giúp theo dõi suy giảm sát mục tiêu hơn.</li>
<li><b>Bổ sung từ và lĩnh vực còn thiếu.</b> Bắt đầu từ nhóm thuật ngữ chưa có trong train, sau khi duyệt nhãn/cách đọc. Thử câu khó người thật và TTS có nhóm đối chứng.</li>
<li><b>Kiểm audio lặp và các nhãn khác nhau.</b> Dùng ID/file trong audit_issues.csv để xác nhận các lần lặp là có chủ đích và mức lấy mẫu không quá tập trung.</li>
<li><b>Hoàn thiện metadata và quy ước split.</b> Vùng giọng, giới tính, mức Việt hoá còn thiếu làm việc phân tích các nhóm này bị hạn chế. Mã video/cụm cần được ghi rõ ý nghĩa.</li>
<li><b>Kiểm các WAV im lặng và nhãn đi kèm trước khi train tiếp.</b> Các file có nhãn lời nói nhưng PCM toàn số 0 cần được nghe kiểm, loại hoặc lấy lại audio đúng. Ghi lại manifest và hash dùng cho từng lần chạy.</li>
</ol></section>
<section id="sources"><h2>Phạm vi, nguồn và cách chạy lại</h2>
<p>Năm manifest chính lấy theo đường dẫn trong run_e3_v1.sh và biến thể v1b. Pool train được dựng lại bằng các recording ID gốc trong năm file nhóm *_cuts_raw.jsonl.gz, bỏ các bản tốc độ 0.9/1.1 để đếm câu gốc. Thống kê file trộn được đọc trực tiếp theo từng cut và ghi rõ là sau tăng cường/lấy lặp. Các tập con B1/B2/B3/B5 tách từ bench hiện tại.</p>
<p>Chuẩn hoá từ dùng lib.textnorm và catalog hiện tại. Tần suất thuật ngữ dùng score.train_term_freq. Người nói/cụm dùng speaker_cluster nếu có, hoặc speaker. SHA256 kiểm nội dung byte file. Các khuyến nghị là diễn giải từ số đo.</p>
<div class="download-links"><a href="summary.json">Số liệu đầy đủ JSON</a><a href="dataset_overview.csv">Tổng quan CSV</a><a href="term_frequency.csv">Thuật ngữ CSV</a><a href="overlap.csv">Trùng dữ liệu CSV</a><a href="source_inventory.csv">Nguồn CSV</a><a href="training_pool_rows.csv">Câu trong pool CSV</a><a href="curated_rows_not_in_train_pool.csv">Dòng bị lọc thêm CSV</a><a href="mix_exposure.csv">Lượt lấy lặp CSV</a><a href="../../../src/gipbench/dataset/report/inventory.py">Script tái lập</a></div>
<p>Chạy tại thư mục dự án bằng <code>python -m gipbench.dataset.report.inventory</code>. Phần kiểm audio mẫu cần ffmpeg và ffprobe trong PATH. Script chạy phân tích bằng CPU và đọc dữ liệu tại chỗ.</p>
<details><summary>Danh mục nguồn và các giai đoạn đang có</summary><p>Các giai đoạn và tập con có quan hệ lồng nhau. Bảng này cho biết phạm vi có trong manifest, các giờ giữa các dòng có thể trùng.</p>{inventory_table}</details>
<details><summary>File manifest và SHA256 lúc phân tích</summary>{provenance_table}</details>
</section>'''

    css = '''
:root{color-scheme:light dark;--bg:#f7f7f4;--surface:#fff;--text:#191a1a;--muted:#5d6265;--line:#dce0e0;--blue:#246bb1;--orange:#b95a20;--soft:#edf4fa;--warning:#fff3db;--s1:#246bb1;--s2:#b95a20;--q3:#246bb1;--grid:#dce0e0}
@media(prefers-color-scheme:dark){:root{--bg:#121514;--surface:#1b1f1e;--text:#edf1ee;--muted:#b9c3bf;--line:#39413e;--blue:#77b5f1;--orange:#e9a16a;--soft:#1c2c39;--warning:#352c1c;--s1:#77b5f1;--s2:#e9a16a;--q3:#77b5f1;--grid:#39413e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 'Segoe UI',system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:38px 24px 80px}header{display:flex;flex-direction:column;gap:16px}h1,h2,h3,p,figure{margin:0}h1{font-size:34px;line-height:1.25;max-width:900px}h2{font-size:25px;line-height:1.3}h3{font-size:17px;line-height:1.45}.lead{font-size:18px;max-width:90ch}.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--muted)}nav,.download-links{display:flex;flex-wrap:wrap;gap:8px 18px}a{color:var(--blue);overflow-wrap:anywhere}section{display:flex;flex-direction:column;gap:18px;margin-top:56px;min-width:0}p{max-width:100ch}.note,figcaption{font-size:13px;color:var(--muted)}.explain{border-left:3px solid var(--blue);background:var(--soft);padding:14px 18px}.finding-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.finding{padding:20px;border:1px solid var(--line);border-radius:8px;background:var(--surface);display:flex;flex-direction:column;gap:8px}.finding>span{font-size:14px;color:var(--muted)}.finding strong{font-size:25px;line-height:1.35}.finding p{font-size:14px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}.kpi{background:var(--surface);padding:15px;border:1px solid var(--line);border-radius:6px;display:flex;flex-direction:column;gap:5px}.kpi strong{font-size:24px}.kpi span{font-size:13px;color:var(--muted)}.view-panel{display:flex;flex-direction:column;gap:20px}[hidden]{display:none!important}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px;display:flex;flex-direction:column;gap:14px;min-width:0}.table-scroll{overflow:auto;max-width:100%;min-width:0}table{border-collapse:collapse;min-width:100%;font-size:13px;line-height:1.55}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;color:var(--muted)}td:not(:first-child){font-variant-numeric:tabular-nums}td:first-child{overflow-wrap:anywhere;max-width:430px}select{align-self:flex-start;background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:5px;padding:10px 14px;font:inherit;max-width:100%;min-width:240px}label{font-weight:600}select:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:3px}details{padding:16px;border:1px solid var(--line);border-radius:6px;background:var(--surface);min-width:0}summary{cursor:pointer;color:var(--blue);font-weight:600;font-size:14px}details>p,details>.table-scroll{margin-top:16px}.flow{display:flex;gap:12px;align-items:stretch}.flow>div{flex:1;padding:18px;border:1px solid var(--line);border-radius:7px;background:var(--surface);display:flex;flex-direction:column;gap:7px}.flow strong{font-size:28px}.flow span{font-size:13px;color:var(--muted)}.flow .flow-link{align-self:center;max-width:95px;text-align:center}code{font:13px Consolas,monospace;background:var(--soft);padding:3px 6px;overflow-wrap:anywhere}ol{margin:0;padding-left:24px;display:flex;flex-direction:column;gap:12px;max-width:100ch}.mobile-hint{display:none}
@media(max-width:800px){.chart-grid{grid-template-columns:1fr}.flow{flex-direction:column}.flow .flow-link{max-width:none}.mobile-hint{display:block}}@media(max-width:520px){main{padding:24px 16px 60px}h1{font-size:27px}h2{font-size:22px}.finding-grid{grid-template-columns:1fr}.finding strong{font-size:23px}figure{padding:14px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpi strong{font-size:21px}select{width:100%}}
'''
    css += '''
.alert{background:var(--warning);border-left:3px solid var(--orange);padding:16px 18px;border-radius:0 6px 6px 0}
.chart-bars{display:flex;flex-direction:column;gap:13px;min-width:0}.bar-row{display:flex;flex-direction:column;gap:5px}.bar-label{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:start;font-size:13px;line-height:1.5}.bar-label span{overflow-wrap:anywhere}.bar-label b{font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}.bar-track{height:10px;border-radius:2px;overflow:hidden;background:var(--line)}.bar-track>span{height:100%;display:block;border-radius:2px}
'''
    css += LINEAGE_CSS
    script = '''
const selector = document.getElementById('dataset-view');
selector.addEventListener('change', () => {
  document.querySelectorAll('[data-view]').forEach(panel => {
    panel.hidden = panel.dataset.view !== selector.value;
  });
});
'''
    report = ('<!doctype html><html lang="vi"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>EDA dataset hiện tại — Gipbench</title><style>' + css + '</style></head><body><main>'
              + body + '</main><script>' + script + '</script></body></html>')
    (OUT / 'index.html').write_text(report, encoding='utf-8')
    print('HTML report written', OUT / 'index.html', flush=True)


if __name__ == '__main__':
    render_report(collect())
