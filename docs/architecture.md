# Моделі і структура коду

## Класи модуля виявлення атак

Код відповідає підрозділам 3.1 і 3.3 пояснювальної записки: діаграмі класів (рисунок 3.10),
діаграмі станів (рисунок 3.11) і формулам (3.1)–(3.4).

| Клас з рисунка 3.10 | Файл | Що робить |
|---|---|---|
| `ChallengeGenerator`, `Challenge`, `OcclusionChallenge`, `HeadTurnChallenge`, `HandSwipeChallenge` | `pad/challenges.py` | випадкові виклики через `secrets.SystemRandom`, 5 типів, без двох однакових поспіль, 3 за замовчуванням |
| `ChallengeVerifier` | `pad/verifier.py` | MediaPipe Hands і Face Mesh; частка перекриття через опуклі оболонки (≥ 0,4 протягом 5 кадрів поспіль), yaw за `cv2.solvePnP` по 6 точках (\|yaw\| ≥ 30°), перетин середньої лінії обличчя центром кисті |
| `DriftFeatureExtractor`, `FeatureVector` | `pad/features.py` | s_ref, s_enr, вікна «до» 1 с, «під час» до 5 с, «після» 1,5 с; мінімум, середнє, дисперсія, найбільший стрибок і час відновлення – 26 ознак |
| `AttackClassifier`, `ThresholdClassifier`, `LogisticClassifier` | `pad/classifiers.py` | поріг на мінімумі s_ref і логістична регресія (StandardScaler + L2, `class_weight='balanced'`); оцінка сеансу – максимум оцінок викликів |
| `SessionManager`, `VerificationSession`, `SessionDecision` | `pad/session.py`, `pad/structures.py` | стани з рисунка 3.11, не більше 2 повторів, сеанс ≤ 20 с, при збої – «не пройдено» |
| `PadSettings` | `pad/settings.py`, `configs/pad.yaml` | параметри модуля і їх межі (FR-62) |

Допоміжні модулі: `pad/embedder.py` (адаптер `embed(face_crop) -> ndarray[128]`), `pad/detector.py`
(тимчасовий детектор замість MTCNN), `pad/pipeline.py` (кадр -> `FrameData`), `pad/metrics.py`
(APCER, BPCER, BPCER20, ACER, EER, AUC), `pad/dataset.py`, `pad/synthetic.py`, `pad/ui.py`, `pad/app.py`.
Веб-інтерфейс (екранні форми 4.7 і 4.8, вкладка «Моніторинг»): `pad/server.py` (FastAPI, REST і
WebSocket), `pad/web/` (односторінковий інтерфейс HTML/JS без фреймворків), `pad/identify.py`
(ідентифікація 1:N для моніторингу), `scripts/serve.py`.
Віддалений запис і офлайн-атаки: `remote/remote_recorder.html`, `pad/offline.py` (обробка відео за
анотацією), `pad/swapper.py` (inswapper_128), `scripts/import_remote.py`, `scripts/generate_attacks.py`.

Назви методів записано за PEP 8: `startSession` -> `start_session`, `onFrame` -> `on_frame`,
`isPerformed` -> `is_performed`, `headPose` -> `head_pose`, `windowStats` -> `window_stats`.

## Моделі розпізнавання

Моделі з ТЗ – MTCNN (детекція і 5 ключових точок) і MobileFaceNet з INT8-квантуванням (ембединг 128) –
готуються з відкритих попередньо навчених ваг, без донавчання:

| Модель | Джерело ваг | Ліцензія | Файл | Розмір |
|---|---|---|---|---|
| MTCNN (P-Net, R-Net, O-Net) | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch), `data/*.pt` | MIT | `models/mtcnn_{pnet,rnet,onet}.onnx` | 2,0 МБ разом |
| MobileFaceNet | [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) v1.0, `mobilefacenet.pt` | Apache 2.0 | `models/mobilefacenet_int8.onnx` | 1,26 МБ (FP32 – 4,0 МБ) |

Експорт в ONNX і квантування потребують PyTorch, тому виконуються в окремому середовищі один раз:

```bash
py -3.12 -m venv .venv-export
.venv-export\Scripts\activate
pip install -r requirements.txt -r requirements-export.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/eval_lfw.py --download      # LFW потрібен для калібрування INT8 і перевірки точності
python scripts/export_models.py
```

`export_models.py` завантажує ваги, перевіряє SHA-256, експортує мережі в ONNX (opset 17) і
квантує MobileFaceNet статично (формат QDQ, ваги INT8 по каналах, калібрування на 300 вирівняних
обличчях LFW). Тимчасові моделі OpenCV Zoo (YuNet, SFace) з `download_models.py` лишаються
запасним варіантом: їх можна вказати в `configs/pad.yaml`.

## Демонстрація без браузера

```bash
python scripts/demo.py
```

Клавіші: `E` – зареєструвати обличчя заявленої особи (30 кадрів), `S` – почати сеанс, `Q` – вихід.
У вікні: відео з точками обличчя і кисті, інструкція виклику з таймером, індикатор виконання,
графік s_ref(t) і s_enr(t), оцінки викликів, рішення, FPS і час етапів. Після виходу у `runs/`
зберігається `demo_stats_*.json` з FPS, часом етапів, витратами модуля на кадр і затримкою рішення.

Інші варіанти: `--camera 1` (наприклад, OBS Virtual Camera), `--enrolled data/enrolled/P01.npy.enc`
(потрібен ключ, розділ «Шифрування записів»), `--video file.mp4 --headless --auto-start` (без вікна,
для вимірювань).

Ключ `PAD_DATASET_KEY` для демонстрації не обов'язковий. Якщо його задано, ембединг з клавіші `E`
зберігається зашифрованим у `data/enrolled/demo.npy.enc`. Без ключа демонстрація працює як раніше,
але ембединг лишається лише в пам'яті до виходу і на диск не пишеться.

До навчання на пілотному наборі використовується базовий поріг (атака, якщо мінімум s_ref під
час виклику менший за 0,3). Це лише заглушка, поріг треба підібрати на даних.
