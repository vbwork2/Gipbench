"""Document dataset origins and processing using local manifests and run evidence."""
import csv
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'eda_current'


def collect_lineage(result):
    """Read metadata only; preserve the existing audio audit snapshot."""
    sources = [
        dict(key='vimedcss', name='ViMedCSS',
             links=[['Hugging Face: tensorxt/ViMedCSS', 'https://huggingface.co/datasets/tensorxt/ViMedCSS'],
                    ['Bài báo mô tả cách tạo nhãn, mục 3.1–3.2', 'https://arxiv.org/html/2602.12911v1']],
             local='data/raw/vimedcss_full/train_vimedcss.jsonl',
             label='Transcript gốc do Gemini 2.5 Pro tạo, sau đó lọc/ngữ nghĩa và căn chỉnh. Bài báo mô tả kiểm thủ công trên 500 câu lấy mẫu; không phải toàn bộ transcript do người chép.',
             processing='Converter dùng cs_terms_list để dựng text_tagged_orig; text_tagged áp thêm quy ước từ mượn, catalog và luật nhận diện của dự án. Audio được đưa về WAV 16 kHz mono; thời lượng đo từ audio.',
             role='train → cs_real; validation → một phần dev CS; hard → B3.',
             evidence=['converters/conv_vimedcss.py', 'data/raw/vimedcss_full/conv.log', 'ft/run_e3_v1.sh']),
        dict(key='vss', name='VietSuperSpeech',
             links=[['Hugging Face: thanhnew2001/VietSuperSpeech', 'https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech']],
             local='data/raw/vss_full/vss_cs_candidates.jsonl',
             label='Transcript nguồn là nhãn máy của Zipformer-30M-RNNT-6000h. Nhánh vss_cs kiểm lại bằng ASR; nhánh distill dùng transcript mới của Gipformer gốc.',
             processing='Tách nhánh có từ Anh thành vss_cs và nhánh Việt thuần thành nguồn distill. Lọc vss_cs theo hai tầng: Qwen trước, faster-whisper cho câu còn span chưa xác nhận; giữ nhãn hiện có, không tự thay bằng câu thầy.',
             role='train → cs_pseudo và distill; 91 câu vss_cs giữ riêng cho dev CS.',
             evidence=['converters/conv_local.py', 'tools/vss_cs_prefilter.py', 'tools/vss_cs_run.py', 'tools/agreement_filter.py', 'ft/prep_e3_v1.py', 'ft/make_distill.py']),
        dict(key='vivos', name='VIVOS',
             links=[['Hugging Face: AILAB-VNUHCM/vivos', 'https://huggingface.co/datasets/AILAB-VNUHCM/vivos']],
             local='data/dl/vivos',
             label='Câu đọc tiếng Việt và transcript trong prompts.txt. Dùng transcript đi kèm bộ dữ liệu, không tạo lại bằng ASR ở nhánh replay.',
             processing='Đọc prompts.txt và WAV; giữ câu Việt thuần. Tách train/dev theo speaker. Confusable chọn lại câu trong VIVOS train có âm Việt dễ nhầm từ Anh, rồi bỏ speaker của dev VI.',
             role='train → replay_human và confusable; một phần train → dev VI; test → B5.',
             evidence=['converters/conv_local.py', 'tools/make_confusables.py', 'ft/prep_e3_v1.py', 'reports/replay_report.md']),
        dict(key='cv', name='Common Voice 27.0 · tiếng Việt', links=[],
             local='data/dl/cv-corpus-27.0-2026-09-11/vi',
             label='Transcript đọc từ cột sentence trong train.tsv/test.tsv. Dự án dùng lại nhãn đi kèm bộ dữ liệu, không tạo lại bằng ASR ở nhánh replay.',
             processing='Đọc clip và TSV, dùng client_id làm mã đại diện người nói; lọc câu có từ Anh khỏi replay Việt thuần. Bản cục bộ ghi phiên bản 27.0, ngày 2026-09-11.',
             role='train → replay_human. Có manifest test trên máy nhưng chưa nằm trong bench đang phân tích.',
             evidence=['converters/conv_local.py', 'reports/replay_report.md']),
        dict(key='ggroup', name='G-Group · bốn bộ vi-asr test',
             links=[[f'g-group-ai-lab/vi-asr-{domain}-test', f'https://huggingface.co/datasets/g-group-ai-lab/vi-asr-{domain}-test']
                    for domain in ['tech', 'edu', 'finance', 'pubadmin']],
             local='data/raw/ggroup/b1_candidates.jsonl',
             label='Converter coi transcript nguồn là trusted_text. Tag <en> do luật của dự án gắn; trusted_text không có nghĩa mọi tag Anh đã được nghe duyệt.',
             processing='Đọc split test; chuyển MP3 sang WAV 16 kHz mono. Câu có tag Anh → ứng viên B1; câu Việt thuần của pubadmin → B5. speaker ban đầu là mã câu, có thể bổ sung cụm ước lượng.',
             role='Chỉ bench: B1 và một phần B5; không đưa vào train.',
             evidence=['converters/conv_ggroup.py', 'data/bench_v0_draft/build.log']),
        dict(key='consensus', name='ZAC / consensus · bộ cục bộ', links=[],
             local='data/dl/zac/consensus_clips_anon.csv',
             label='Dùng consensus_text và trạng thái agreed/disputed/pending trong CSV. Không có URL xuất bản hay thông tin phiên bản công khai được ghi nhận trong repo.',
             processing='Ghép (id, turn) với audio; chỉ agreed làm ứng viên trực tiếp. Đổi số ra chữ, chuẩn hoá và gắn tag Anh. Giới hạn tối đa 10 câu/thuật ngữ B2; disputed/pending cần quyết định duyệt riêng.',
             role='Chỉ test theo cờ test_only của dự án: B2 và B5 product; không dùng để train.',
             evidence=['converters/conv_consensus.py', 'data/raw/consensus/consensus_stats.json', 'data/bench_v0_draft/build.log']),
    ]
    evidence_paths = {path for source in sources for path in source['evidence']}
    evidence_paths.update([
        'data/dl/zac/consensus_clips_anon.csv',
        'data/dl/cv-corpus-27.0-2026-09-11/vi/train.tsv',
        'data/dl/cv-corpus-27.0-2026-09-11/vi/test.tsv',
        'data/raw/vss_full/vss_cs_pre_rebuild.jsonl.report.md',
        'data/raw/vss_full/vss_cs_clean.jsonl.report.md',
        'data/e3_v1/distill.jsonl.report.md',
        'data/e3_v1/train_curated.jsonl.report.md',
        'data/e3_v1/dev_cs.jsonl', 'data/e3_v1b/train_curated.jsonl',
        'ft/curate_train.py', 'ft/to_lhotse.py', 'ft/mix_cuts.py',
        'converters/build_bench_v0.py', 'configs/train_mix.tsv',
        'tools/eda_current.py', 'tools/eda_lineage.py',
        'configs/confusables.tsv', 'configs/term_decisions.tsv',
        'data/bench_v0_draft/catalog_v0_draft.tsv',
        'reports/eda_current/training_pool_rows.csv',
    ])
    stages = []
    cache = {}

    def manifest(path):
        if path not in cache:
            content = (ROOT / path).read_bytes()
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            cache[path] = dict(rows=len(rows), hours=sum(float(row.get('duration') or 0) for row in rows) / 3600,
                               sha256=hashlib.sha256(content).hexdigest())
        return cache[path]

    def add(origin, branch, stage, path, method, parent='', stats=None):
        measured = stats if stats is not None else manifest(path)
        stages.append(dict(origin=origin, branch=branch, stage=stage, parent_stage=parent, path=path,
                           rows=measured['rows'], hours=measured['hours'],
                           sha256=measured.get('sha256', ''), method=method))
        evidence_paths.add(path)

    branches = [
        ('vimedcss', 'ViMedCSS → cs_real', 'ViMedCSS', 'data/raw/vimedcss_full/train_vimedcss.jsonl', 'data/e3_v1/train_vimedcss_abs.jsonl', 'Đổi đường dẫn audio thành tuyệt đối; giữ nguyên các dòng train nguồn.'),
        ('vss', 'VSS → vss_cs / cs_pseudo', 'vss_cs', 'data/raw/vss_full/vss_cs_clean_dom.jsonl', 'data/e3_v1/train_vss_cs.jsonl', 'Tách theo video/speaker, seed 0; 91 câu sang dev, phần còn lại vào train.'),
        ('vivos', 'VIVOS → replay_human', 'vivos', 'data/raw/vivos/train_vivos.jsonl', 'data/e3_v1/train_vivos.jsonl', 'Tách theo speaker, seed 0; 4 speaker, 1 099 câu sang dev VI.'),
        ('cv', 'Common Voice → replay_human', 'commonvoice', 'data/raw/cv/train_cv.jsonl', 'data/raw/cv/train_cv.jsonl', 'Dùng manifest train Việt thuần; không tách dev mới.'),
        ('vss', 'VSS → distill', 'distill', 'data/e3_v1/distill_pool.jsonl', 'data/e3_v1/distill.jsonl', 'Gipformer gốc tạo đáp án: greedy phải trùng modified beam; giữ tốc độ 1–6 từ/giây.'),
        ('vivos', 'VIVOS → confusable', 'confusable', 'data/raw/confusable/train_confusable.jsonl', 'data/e3_v1/train_confusable.jsonl', 'Chọn câu theo âm dễ nhầm; bỏ speaker thuộc dev VI. Không tạo audio tổng hợp.'),
    ]
    for origin, branch, dataset, raw, prepared, method in branches:
        add(origin, branch, 'Nguồn vào nhánh', raw, 'Manifest cục bộ, trước bước chuẩn bị riêng cho E3 v1.')
        add(origin, branch, 'Đầu vào curate v1', prepared, method, 'Nguồn vào nhánh')
        for key, stage, parent in [('train_v1', 'Sau curate v1', 'Đầu vào curate v1'), ('train_pool', 'Pool sau Lhotse', 'Sau curate v1')]:
            stats = next(row for row in result['views'][key]['datasets'] if row['value'] == dataset)
            path = 'data/e3_v1/train_curated.jsonl' if key == 'train_v1' else 'reports/eda_current/training_pool_rows.csv'
            add(origin, branch, stage, path,
                'Đếm riêng dataset trong manifest chung.' if key == 'train_v1' else 'Đếm recording ID gốc của các cut, bỏ bản tốc độ 0.9/1.1; đối chiếu lại manifest.',
                parent, stats)

    for stage, path in [
        ('Ứng viên có tag Anh', 'data/raw/vss_full/vss_cs_candidates.jsonl'),
        ('Sau lọc rẻ', 'data/raw/vss_full/vss_cs_pre_rebuild.jsonl'),
        ('Sau lọc ASR và suy domain', 'data/raw/vss_full/vss_cs_clean_dom.jsonl'),
        ('Nguồn Việt thuần cho distill', 'data/raw/vss_full/train_vss_dom.jsonl'),
    ]:
        add('vss', 'VSS trước khi tách E3', stage, path, 'Số dòng và giờ của manifest hiện có; không phải toàn bộ quy mô dataset trên Hub.')
    for origin, branch, stage, path in [
        ('vimedcss', 'ViMedCSS validation → dev CS', 'Validation nguồn', 'data/raw/vimedcss_full/dev_vimedcss.jsonl'),
        ('vss', 'VSS → dev CS', 'Dev theo video', 'data/e3_v1/dev_cs_vss.jsonl'),
        ('vivos', 'VIVOS → dev VI', 'Dev theo speaker', 'data/e3_v1/dev_vi.jsonl'),
        ('vimedcss', 'ViMedCSS hard → B3', 'Ứng viên hard', 'data/raw/vimedcss_hard/b3_candidates.jsonl'),
        ('ggroup', 'G-Group → B1', 'Ứng viên B1', 'data/raw/ggroup/b1_candidates.jsonl'),
        ('ggroup', 'G-Group → B5', 'Ứng viên B5', 'data/raw/ggroup/b5_ggroup.jsonl'),
        ('consensus', 'Consensus → B2', 'Ứng viên B2 sau trần', 'data/raw/consensus/b2_consensus_candidates.jsonl'),
        ('consensus', 'Consensus → B5', 'Ứng viên B5', 'data/raw/consensus/b5_consensus_product.jsonl'),
        ('vivos', 'VIVOS test → B5', 'Ứng viên B5', 'data/raw/vivos/b5_vivos.jsonl'),
    ]:
        add(origin, branch, stage, path, 'Đếm trực tiếp manifest cục bộ.')

    bench_path = 'data/bench_v0_draft/bench_v0_draft.jsonl'
    bench = [json.loads(line) for line in (ROOT / bench_path).read_bytes().splitlines() if line.strip()]
    bench_groups = defaultdict(list)
    for row in bench:
        dataset = str(row['dataset']).split(':')[0]
        origin = 'ggroup' if dataset.startswith('vi-asr-') else 'vimedcss' if dataset == 'ViMedCSS' else dataset
        bench_groups[(origin, row['split'])].append(row)
    bench_composition = []
    for (origin, split), rows in sorted(bench_groups.items()):
        stats = dict(rows=len(rows), hours=sum(float(row['duration']) for row in rows) / 3600)
        bench_composition.append(dict(origin=origin, split=split, **stats))
        add(origin, f'{origin} → {split}', 'Trong bench hiện tại', bench_path, 'Đếm dataset và split trong file bench đầu ra.', stats=stats)

    def log_text(path):
        evidence_paths.add(path)
        return (ROOT / path).read_text(encoding='utf-8')

    agreement_text = log_text('data/raw/vss_full/vss_cs_clean.jsonl.report.md')
    match = re.search(r'Giữ \*\*(\d+) / (\d+)\*\*', agreement_text)
    agreement = dict(kept=int(match[1]), examined=int(match[2])) if match else {}
    untested = re.search(r'chưa có hyp tầng 1: (\d+)', agreement_text)
    if untested:
        agreement['without_first_teacher_hyp'] = int(untested[1])

    def markdown_steps(path):
        steps = []
        for line in log_text(path).splitlines():
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if line.startswith('|') and len(cells) >= 3 and cells[1].isdigit():
                steps.append(dict(step=cells[0], rows=int(cells[1]), hours=float(cells[2])))
            elif steps and not line.startswith('|'):
                break
        return steps

    evidence = []
    for relative in sorted(evidence_paths):
        path = ROOT / relative
        if path.is_file():
            payload = path.read_bytes()
            evidence.append(dict(path=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
        else:
            evidence.append(dict(path=relative, bytes=None, sha256='', missing=True))
    hashes = {row['path']: row['sha256'] for row in evidence}
    for relative, previous in result['provenance'].items():
        if relative in hashes and previous['sha256'] != hashes[relative]:
            raise ValueError(f'Manifest changed since the audio audit; rerun tools/eda_current.py: {relative}')
    for stage in stages:
        stage['sha256'] = hashes[stage['path']]
    lineage = dict(documented_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat(),
                   audio_audit_generated_at=result['generated_at'], sources=sources, stages=stages,
                   bench_composition=bench_composition, agreement=agreement,
                   vss_prefilter_steps=markdown_steps('data/raw/vss_full/vss_cs_pre_rebuild.jsonl.report.md'),
                   curate_steps=markdown_steps('data/e3_v1/train_curated.jsonl.report.md'),
                   evidence=evidence,
                   version_note='Public URLs identify upstream datasets. No immutable Hub revision or archive checksum was recorded for every download. Local manifest and evidence SHA256 values identify this analyzed snapshot.')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [('dataset_lineage.csv', stages), ('lineage_evidence.csv', evidence)]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (OUT / 'source_provenance.json').write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding='utf-8')
    return lineage


def render_lineage(lineage):
    """Render a standalone, responsive explanation with source and evidence links."""
    def local(path, label=None):
        return f'<a href="{escape("../../" + path, quote=True)}">{escape(label or path)}</a>'

    def table(headings, rows):
        return ('<div class="table-scroll"><table><thead><tr>'
                + ''.join(f'<th>{escape(cell)}</th>' for cell in headings) + '</tr></thead><tbody>'
                + ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
                + '</tbody></table></div>')

    def count(value):
        return f'{value:,}'.replace(',', ' ')

    names = {source['key']: source['name'] for source in lineage['sources']}
    cards = []
    for source in lineage['sources']:
        links = '<br>'.join(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>' for label, url in source['links'])
        if not links:
            links = 'Nguồn cục bộ; xem đường dẫn và giới hạn truy xuất bên dưới.'
        evidence = ' · '.join(local(path) for path in source['evidence'])
        cards.append(f'''<article class="origin-card"><h3>{escape(source['name'])}</h3><div>{links}</div>
<p><b>Nhãn từ đâu?</b> {escape(source['label'])}</p><p><b>Đã xử lý gì?</b> {escape(source['processing'])}</p>
<p><b>Dùng ở đâu?</b> {escape(source['role'])}</p><p class="note">Bản cục bộ: {local(source['local'])}</p>
<details><summary>Mã và bằng chứng</summary><p>{evidence}</p></details></article>''')

    routes = []
    grouped = defaultdict(list)
    for stage in lineage['stages']:
        if stage['stage'] in ['Nguồn vào nhánh', 'Đầu vào curate v1', 'Sau curate v1', 'Pool sau Lhotse']:
            grouped[stage['branch']].append(stage)
    for branch, stages in grouped.items():
        nodes = []
        for stage in stages:
            nodes.append(f'<div class="lineage-node"><small>{escape(stage["stage"])}</small><b>{count(stage["rows"])} câu</b>'
                         f'<span>{stage["hours"]:.2f} giờ</span><span class="note">{local(stage["path"])}</span></div>')
        routes.append(f'<figure class="lineage-route"><h3>{escape(branch)}</h3><div class="lineage-flow">'
                      + '<span class="lineage-arrow" aria-hidden="true">→</span>'.join(nodes) + '</div>'
                      + f'<figcaption>{escape(stages[1]["method"])}</figcaption></figure>')

    composition = table(['Nguồn gốc', 'Nhánh bench', 'Số câu', 'Giờ'], [
        [escape(names[row['origin']]), escape(row['split']), count(row['rows']), f'{row["hours"]:.2f}']
        for row in lineage['bench_composition']])
    vss_steps = table(['Bước lọc rẻ đã chạy', 'Câu còn', 'Giờ còn'], [
        [escape(row['step']), count(row['rows']), f'{row["hours"]:.2f}'] for row in lineage['vss_prefilter_steps']])
    curate_steps = table(['Bước curate v1 đã chạy', 'Câu còn', 'Giờ còn'], [
        [escape(row['step']), count(row['rows']), f'{row["hours"]:.2f}'] for row in lineage['curate_steps']])
    agreement = lineage['agreement']
    methods = [
        ('1. Đưa về một cấu trúc chung',
         'Converter tạo manifest JSONL: mỗi dòng có id, audio, duration, text_tagged, dataset, source, speaker và split. Tag <en> đánh dấu cụm tiếng Anh để phân tích; đây là nhãn xử lý của dự án.',
         ['converters/conv_vimedcss.py', 'converters/conv_local.py', 'converters/conv_ggroup.py', 'converters/conv_consensus.py']),
        ('2. Lọc vss_cs bằng ASR',
         f'Qwen3-ASR-0.6B kiểm trước; faster-whisper large-v3 int8 chỉ chạy cho câu còn span chưa được xác nhận. Mỗi span cần ít nhất 1 ASR xác nhận, 100% span trong câu phải đạt; WER phần Việt thấp nhất trong các ASR có hyp phải ≤ 0.15. Log ghi {count(agreement.get("kept", 0))}/{count(agreement.get("examined", 0))} câu đã có hyp được giữ. {count(agreement.get("without_first_teacher_hyp", 0))} ứng viên chưa có hyp tầng 1 không tính là đã bị ASR loại. Đây không phải yêu cầu cả hai ASR cùng đồng ý mọi câu.',
         ['data/raw/vss_full/vss_cs_clean.jsonl.report.md', 'tools/vss_cs_run.py', 'tools/agreement_filter.py']),
        ('3. Tách dev trước khi curate',
         'VSS tách theo video: 1 566 train và 91 dev. VIVOS tách theo speaker: 10 559 train và 1 099 dev VI của 4 người. ViMedCSS validation được xáo với seed 0, lấy 600 câu đầu rồi bỏ 1 câu dưới 0.5 giây, còn 599. Ghép 599 + 91 thành dev CS 690 câu. Cách làm hiện tại vẫn còn nguồn ViMedCSS trùng train; không thể coi dev đã tách nguồn hoàn toàn.',
         ['ft/prep_e3_v1.py', 'ft/run_e3_v1.sh']),
        ('4. Tạo distill và confusable',
         'Distill: lấy VSS Việt thuần, bỏ ID ứng viên CS và video thuộc nhánh CS, giữ độ dài 1–20 giây; pool 2 630 câu/9.44 giờ. Gipformer gốc giải mã greedy và modified beam; hai chuỗi giống nhau và 1–6 từ/giây mới giữ, còn 1 889 câu/6.78 giờ. Confusable: chọn audio VIVOS đã có chứa cách đọc dễ nhầm như “tim/team”; nhãn vẫn là tiếng Việt. Các nhánh này không tạo thêm người nói hay bản ghi âm mới.',
         ['ft/prep_e3_v1.py', 'ft/make_distill.py', 'data/e3_v1/distill.jsonl.report.md', 'tools/make_confusables.py']),
        ('5. Curate để bớt lệch dữ liệu',
         'Gộp 6 manifest; chặn test_only, giữ finance ngoài train, giới hạn 50 câu/thuật ngữ, 2% giờ/người nói và 35% giờ CS/domain (miễn other). Log lần v1 ghi trần synthetic 20%, nhưng không có audio TTS nên bước này không loại câu nào. Phải dùng log của lần chạy để biết tham số lịch sử; giá trị mặc định trong code/config hiện tại có thể đã thay đổi.',
         ['data/e3_v1/train_curated.jsonl.report.md', 'ft/curate_train.py']),
        ('6. Chuyển Lhotse và tăng cường',
         'Bỏ speaker/source/text trùng bench, thuật ngữ reserved_b3, câu ngoài 1–20 giây, text rỗng hoặc audio không đọc được. Chuẩn hoá text, bỏ tag, đổi IN HOA cho tokenizer; audio về 16 kHz, mô phỏng điện thoại 8 kHz cho khoảng 30% cut, thêm tốc độ 0.9/1.1. Sau lọc còn 19 228 câu gốc. File trộn lấy lặp theo nhóm đạt khoảng 960 giờ cut; đây là giờ kế hoạch sau tăng cường, không phải 960 giờ audio gốc hay xác nhận mô hình đã học hết.',
         ['ft/to_lhotse.py', 'ft/run_e3_v1.sh', 'ft/mix_cuts.py', 'reports/eda_current/training_pool_rows.csv']),
        ('7. Dựng bench tách khỏi train',
         'G-Group → B1; consensus agreed → B2; ViMedCSS hard → B3; Việt thuần của VIVOS test/G-Group/consensus → B5. Log build dùng --accept all cho các ứng viên đủ điều kiện, nên chưa có nghĩa mọi câu/tag đã được người nghe duyệt. Riêng 134/658 ứng viên hard bị bỏ vì không còn tag Anh theo nhãn Spec, còn B3 524 câu. 315 thuật ngữ reserved_b3 được chặn khỏi pool train; các thuật ngữ thông thường khác trong B3 có thể đã xuất hiện ở train.',
         ['converters/build_bench_v0.py', 'data/bench_v0_draft/build.log', 'data/raw/vimedcss_hard/b3_candidates.jsonl']),
    ]
    method_html = ''.join(f'<details class="method-detail" open><summary>{escape(title)}</summary><p>{escape(text)}</p><p class="note">Đối chiếu: '
                          + ' · '.join(local(path) for path in paths) + '</p></details>' for title, text, paths in methods)
    evidence_table = table(['Bằng chứng cục bộ', 'SHA256 của file'], [
        [local(row['path']), f'<code>{escape(row["sha256"])}</code>'] for row in lineage['evidence']])
    return f'''<section id="lineage"><h2>Lấy dữ liệu từ đâu, rồi xử lý như thế nào?</h2>
<p class="lead">Có sáu nguồn gốc. Các tên vss_cs, distill, confusable và replay là nhánh xử lý hoặc vai trò học, không phải bốn dataset độc lập mới tải về.</p>
<p class="explain"><b>Đọc nhanh các từ dùng bên dưới:</b> nhãn/transcript là câu chữ làm đáp án cho audio; ASR là hệ nhận dạng giọng nói. Manifest là danh sách file audio và nhãn tương ứng. Curate là chọn/lọc dữ liệu theo quy tắc. Pool sau Lhotse là tập câu đã qua bước chuyển định dạng và lọc kỹ thuật, sẵn sàng lấy mẫu để học. Train dùng để học; dev dùng để chọn mô hình; bench dùng để đánh giá.</p>
<div class="alert"><b>Làm rõ nguồn nhãn ViMedCSS.</b> cs_real là tên nhóm trong code. Audio là lời nói thật; nhãn gốc vẫn do Gemini tạo và được kiểm trên mẫu. Không dùng tên nhóm này để kết luận toàn bộ nhãn do người chép.</div>
<div class="origin-grid">{''.join(cards)}</div>
<h3>Bản đồ từ nguồn đến vai trò</h3>
<div class="origin-map"><div><b>ViMedCSS</b><span>train → cs_real</span><span>validation → dev CS</span><span>hard → B3</span></div>
<div><b>VietSuperSpeech</b><span>có Anh → lọc ASR → cs_pseudo</span><span>nhánh Việt → Gipformer → distill</span><span>một phần CS → dev CS</span></div>
<div><b>VIVOS</b><span>train → replay + confusable</span><span>speaker giữ riêng → dev VI</span><span>test → B5</span></div>
<div><b>Common Voice 27.0 vi</b><span>train → replay_human</span></div><div><b>G-Group test</b><span>có Anh → B1</span><span>pubadmin Việt thuần → B5</span></div>
<div><b>Consensus cục bộ</b><span>agreed có Anh → B2</span><span>agreed Việt thuần → B5</span><span>test_only, không train</span></div></div>
<h3>Số liệu từng nhánh đi vào train v1</h3><p>Mỗi hàng sơ đồ đi từ trái sang phải. “Nguồn vào nhánh” đã qua converter hoặc bước chọn ứng viên; đó không phải toàn bộ dataset gốc trên mạng. Giờ tính theo dòng manifest; confusable có thể dùng lại cùng WAV với replay.</p>
{''.join(routes)}
<details><summary>VSS trước bước tách train/dev: lọc những gì?</summary>{vss_steps}
<p>Sau lọc rẻ còn 12 351 ứng viên. Log ASR ghi 6 412 câu đã có hyp được xét, giữ 1 657 câu/6.09 giờ; 5 939 câu chưa có hyp tầng 1. Vì vậy 1 657/12 351 không phải tỉ lệ đạt của một lượt ASR đã chạy hết toàn bộ ứng viên.</p></details>
<h3>Phương pháp đã dùng và bằng chứng</h3>{method_html}
<details><summary>Từng bước curate v1 loại bao nhiêu?</summary>{curate_steps}<p>Đọc từ báo cáo lần chạy v1; số giờ trong log được làm tròn hai chữ số. Thống kê EDA phía trên tính lại từ manifest hiện có.</p></details>
<h3>Bench hiện tại thực sự lấy từ những nguồn nào?</h3>{composition}
<p class="explain">Có file trên ổ đĩa không đồng nghĩa đã dùng trong lần chạy. Common Voice test có trong kho nhưng chưa có trong bench hiện tại. VSS Việt thuần ngoài pool distill cũng chưa tự động trở thành train E3. Các bộ test-only của G-Group/consensus nằm ở nhánh đánh giá.</p>
<h3>Ví dụ đọc một mẫu từ nguồn đến câu học</h3>
<figure><div class="lineage-flow"><div class="lineage-node"><small>Transcript nguồn</small><b>Audio người thật + câu chữ</b><span>“em gửi file qua email”</span></div>
<span class="lineage-arrow" aria-hidden="true">→</span><div class="lineage-node"><small>Manifest của dự án</small><b>Gắn phạm vi từ Anh</b><code>em gửi &lt;en&gt;file&lt;/en&gt; qua &lt;en&gt;email&lt;/en&gt;</code></div>
<span class="lineage-arrow" aria-hidden="true">→</span><div class="lineage-node"><small>Supervision để train</small><b>Chuẩn hoá và bỏ tag</b><code>EM GỬI FILE QUA EMAIL</code></div></div>
<figcaption>Ví dụ minh hoạ cách biến đổi text, không phải bản chép đã nghe xác nhận của một clip cụ thể. Audio không được tạo lại chỉ vì gắn tag. Nếu nhãn từ ASR vẫn sai thì lọc tự động chưa bảo đảm đáp án đúng.</figcaption></figure>
<h3>Truy xuất lại bản dữ liệu đã phân tích</h3>
<p>Số liệu lấy từ bản cục bộ, không lấy số tổng hiện trên Hub để thay cho dữ liệu đã dùng. Repo chưa ghi đầy đủ commit/revision bất biến trên Hub hoặc checksum gói tải gốc cho mọi nguồn. Do đó xác định được dataset gốc, nhánh xử lý và hash file cục bộ; chưa khẳng định có thể tải lại đúng từng byte của mọi bản gốc chỉ bằng URL.</p>
<p class="note">Bổ sung truy xuất lúc {escape(lineage['documented_at'])}; kết quả kiểm audio giữ theo snapshot {escape(lineage['audio_audit_generated_at'])}. Hash code/config ghi trạng thái khi bổ sung nguồn, còn tham số lần chạy lịch sử dựa vào log liên kết.</p>
<div class="download-links"><a href="dataset_lineage.csv">Các giai đoạn và số liệu CSV</a><a href="source_provenance.json">Nguồn và xử lý JSON</a><a href="lineage_evidence.csv">Hash bằng chứng CSV</a><a href="../../tools/eda_lineage.py">Mã dựng phần truy xuất</a></div>
<details><summary>File bằng chứng và hash để đối chiếu</summary>{evidence_table}</details></section>'''


LINEAGE_CSS = '''
.origin-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.origin-card{display:flex;flex-direction:column;gap:12px;min-width:0;padding:20px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}.origin-card p{font-size:14px}.origin-card>div{font-size:13px}.origin-card details{padding:10px 12px}.origin-map{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.origin-map>div{display:flex;flex-direction:column;gap:8px;padding:16px;border:1px solid var(--line);border-top:3px solid var(--blue);border-radius:6px;background:var(--surface)}.origin-map span{font-size:13px}.lineage-flow{display:flex;gap:10px;align-items:stretch;min-width:0}.lineage-node{display:flex;flex:1;flex-direction:column;gap:8px;padding:14px;min-width:0;border:1px solid var(--line);border-radius:6px;background:var(--soft)}.lineage-node small{color:var(--muted)}.lineage-node b{font-size:18px;line-height:1.45}.lineage-node span,.lineage-node a{overflow-wrap:anywhere}.lineage-node code{padding:0}.lineage-arrow{align-self:center;color:var(--blue);font-size:23px}.method-detail p{margin-top:14px}#lineage td code{word-break:break-all;display:block;max-width:500px}
@media(max-width:850px){.lineage-flow{flex-direction:column}.lineage-arrow{transform:rotate(90deg)}.origin-map{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.origin-grid,.origin-map{grid-template-columns:1fr}.origin-card{padding:16px}}
'''
