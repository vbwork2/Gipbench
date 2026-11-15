# Phân bố domain theo nhóm dữ liệu

Thuật ngữ = span `<en>` sau chuẩn hoá (gộp biến thể catalog). `nguồn domain`: converter = domain mặc định của converter / dataset, inferred = suy từ text (`tools/infer_domain.py`, `configs/domain_keywords.tsv`), dataset = nhãn của dataset.

| Nhóm | Câu | Giờ | Thuật ngữ khác nhau |
| --- | --- | --- | --- |
| B1 | 1330 | 1.38 | 1518 |
| B2 | 218 | 0.40 | 134 |
| B3 | 658 | 1.38 | 582 |
| cs_real | 11832 | 24.30 | 2117 |
| cs_pseudo | 1133 | 4.15 | 665 |

## B1

Nguồn: `data/raw/ggroup/b1_candidates.jsonl`. 1330 câu, 1.38 h, 1518 thuật ngữ khác nhau; nguồn domain: converter 1330.

Top 15 thuật ngữ (số lần): website 30 · ok 25 · gov 24 · code 18 · online 15 · ielts 13 · visa 11 · edu 11 · token 10 · app 10 · email 10 · click 8 · api 8 · link 8 · photo 8

| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |
| --- | --- | --- | --- | --- | --- |
| tech | 469 | 0.46 | 33.6% | 676 | code 14, website 9, ok 9, api 8, app 8 |
| edu | 355 | 0.40 | 28.9% | 431 | ielts 13, ok 8, edu 8, website 6, bachelor 6 |
| finance | 316 | 0.32 | 23.1% | 392 | vietcombank 7, coin 7, bitcoin 6, ethereum 5, check 4 |
| pubadmin | 190 | 0.20 | 14.4% | 140 | gov 23, website 15, online 10, photo 7, scan 7 |

## B2

Nguồn: `data/raw/consensus/b2_consensus_candidates.jsonl`. 218 câu, 0.40 h, 134 thuật ngữ khác nhau; nguồn domain: converter 218.

Top 15 thuật ngữ (số lần): zalo 10 · sms 10 · ok 10 · check 9 · email 9 · pizza 6 · st 5 · metro 5 · sarah 5 · rv 4 · shop 4 · online 4 · pancake 4 · zkn 4 · vitech 4

| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |
| --- | --- | --- | --- | --- | --- |
| product | 218 | 0.40 | 100.0% | 134 | zalo 10, sms 10, ok 10, check 9, email 9 |

## B3

Nguồn: `data/dl/vimedcss_meta/hard_set.csv`. 658 câu, 1.38 h, 582 thuật ngữ khác nhau; nguồn domain: dataset 658.

Top 15 thuật ngữ (số lần): bilirubin 92 · aldosterone 39 · listeria 28 · angiotensin 9 · kali 9 · enzyme 9 · renin 8 · amin 7 · gilbert 6 · proline 5 · koch 5 · monocytogenes 5 · mg 5 · theophyllin 4 · dna 4

| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |
| --- | --- | --- | --- | --- | --- |
| medical | 658 | 1.38 | 100.0% | 582 | bilirubin 92, aldosterone 39, listeria 28, angiotensin 9, kali 9 |

## cs_real

Nguồn: `data/dl/vimedcss_meta/train_set.csv`. 11832 câu, 24.30 h, 2117 thuật ngữ khác nhau; nguồn domain: dataset 11832.

Top 15 thuật ngữ (số lần): colagen 289 · glucose 251 · hormone 221 · inulin 204 · candida 169 · vaccine 166 · corticoid 163 · polyp 131 · lupus 130 · lympho 114 · rh 111 · estrogen 108 · stent 105 · testosterone 100 · amin 100

| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |
| --- | --- | --- | --- | --- | --- |
| medical | 11832 | 24.30 | 100.0% | 2117 | colagen 289, glucose 251, hormone 221, inulin 204, candida 169 |

## cs_pseudo

Nguồn: `data/raw/vss_full/vss_cs_clean_dom.jsonl`. 1133 câu, 4.15 h, 665 thuật ngữ khác nhau; nguồn domain: inferred 1133.

Top 15 thuật ngữ (số lần): amazon 14 · happy 13 · mindset 13 · victor 13 · clip 12 · yoga 12 · tinder 12 · hit 12 · camera 12 · host 11 · series 11 · mc 10 · level 10 · plan 10 · trend 10

| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |
| --- | --- | --- | --- | --- | --- |
| other | 924 | 3.38 | 81.4% | 581 | yoga 12, clip 11, mindset 11, series 11, mc 10 |
| business | 50 | 0.18 | 4.4% | 49 | sale 5, amazon 4, feedback 4, start 4, deadline 4 |
| entertainment | 42 | 0.16 | 3.7% | 47 | tiktok 5, mv 5, shop 4, rap 3, livestream 3 |
| tech | 40 | 0.15 | 3.6% | 52 | online 7, internet 6, blockchain 5, camera 4, cloud 3 |
| daily | 27 | 0.10 | 2.4% | 28 | tinder 7, logic 2, arabica 2, noel 2, valentine 1 |
| edu | 18 | 0.06 | 1.6% | 22 | happy 2, house 2, grab 2, ship 2, job 1 |
| medical | 15 | 0.06 | 1.3% | 16 | ivf 2, protein 2, virus 2, clinic 1, dr 1 |
| finance | 15 | 0.05 | 1.3% | 16 | crypto 4, stock 2, invest 2, not 2, search 1 |
| pubadmin | 2 | 0.01 | 0.2% | 2 | philippines 2, skill 1 |

