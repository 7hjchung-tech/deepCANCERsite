# NRP Coder에서 deepCANCERsite 돌리기

대상: NRP Nautilus의 **Coder** 워크스페이스 (`https://coder.nrp-nautilus.io`).
전제: 약리학교실 NRP 가이드 개론의 "2. 기본 설정"(CILogon 이메일 → nrp.ai 로그인 →
namespace 승인)과 "3. Nautilus Support Channel 가입"이 끝나 있어야 합니다.

---

## 0. 이 저장소가 서버에서 필요로 하는 것

| 항목 | 값 | 비고 |
|---|---|---|
| GPU | 1장, **24GB 이상 권장** | M3/M4는 ESM-2 650M end-to-end + LoRA (fp16, batch 8, grad ckpt) |
| RAM | 16GB | 가이드 기본값으로 충분 |
| CPU | 8 cores | 가이드 기본값 |
| 디스크 | **최소 15GB, 50GB 권장** | Coder 기본 5GB로는 부족 (§2 참조) |
| 다운로드 | ESM-2 650M 체크포인트 ~2.5GB | `dl.fbaipublicfiles.com`에서 자동 |

디스크 내역: ESM 체크포인트 2.5GB + (이미지에 torch가 없을 경우) torch CUDA 휠 ~3GB
+ 저장소/데이터 ~10MB + `data/diff_emb_raw.pt` ~30MB.

---

## 1. 워크스페이스 생성

1. `https://coder.nrp-nautilus.io` → **Sign in with Authentik**
2. 상단 **Templates** → **Cuda/Pytorch/TensorFlow** → **Create Workspace**
3. 파라미터:
   - Workspace Name: `deepcancersite` (자유)
   - CPU cores: `8`
   - GPUs: `1`
   - GPU Type: `Any`로 시작하되, **M3/M4를 돌릴 거면 24GB급을 지정**
     (A10 / A40 / L40 / RTX 3090 등). `Any`는 11GB짜리 2080Ti가 걸릴 수 있고
     그러면 e2e LoRA 학습에서 OOM이 납니다. A100은 Coder에서는 배정 불가 —
     K8s로 가야 합니다.
   - Memory: `16GB`
4. 오래 기다렸는데 Failed가 뜨면 `https://nrp.ai/viz/resources/` 표에서
   (1) 빨간색이 아니고 (2) Taints 열이 빈 서버 사양에 맞춰 조정.

> 사양은 같은 워크스페이스 안에서 나중에 바꿀 수 있습니다.
> 단 **Workspace를 삭제하면 파일이 전부 날아갑니다** (Stop은 안전).
> 그리고 마지막 활동 후 **24시간이 지나면 자동 종료**됩니다.

---

## 2. 디스크 늘리기 (먼저 해두세요, 승인에 시간이 걸립니다)

초기 볼륨은 5GB인데 ESM 체크포인트만으로 절반이 찹니다.
Element(Matrix)의 **Nautilus Support** 채널에 아래 메시지를 보내세요:

```
Hi, I am from the Yonsei University Genome Editing Laboratory, and my email is
seungwonjo810@yonsei.ac.kr. Would it be possible to increase my Coder Volume of
Workspace name deepcancersite to 50 GB? Thank you very much.
```

(워크스페이스 이름은 실제로 만든 이름으로 바꾸세요.)

---

## 3. 접속

워크스페이스가 뜨면 버튼이 여러 개 생깁니다:

- **VS Code Desktop** — 로컬에 VS Code가 깔려 있으면 원클릭 원격 연결 (추천)
- **code-server** — 브라우저 VSCode, 설치 불필요
- **Terminal** — 리눅스 터미널

---

## 4. 저장소 클론 + 환경 세팅

워크스페이스 터미널에서:

```bash
git clone https://github.com/7hjchung-tech/deepCANCERsite.git
cd deepCANCERsite
bash scripts/nrp_setup.sh
```

`nrp_setup.sh`가 하는 일:
1. GPU / 디스크 / 이미지에 미리 깔린 torch 확인
2. `.venv`를 `--system-site-packages`로 생성 → 이미지의 CUDA torch를 재사용
   (5GB 볼륨에 torch 휠 2.5GB를 또 받지 않기 위함)
3. 빠진 패키지만 설치 (numpy / pandas / pyyaml)
4. ESM-2 650M 체크포인트 다운로드 (`~/.cache/torch/hub/checkpoints/`, 홈은 영구 볼륨)
5. `torch.cuda` + ESM forward 1회로 검증

재실행해도 안전합니다. 끝나면:

```bash
source .venv/bin/activate
```

구조 피처 파이프라인(`data/structure/code/`)이나 baseline까지 돌릴 거면 추가로:

```bash
pip install -r requirements.txt
```

---

## 5. 실제 실행

### 5-1. diff embedding 캐시 생성 (GPU가 실제로 필요한 단계)

M1/M2는 frozen ESM이라 임베딩을 미리 한 번만 계산해두면 됩니다.
5,887개 variant × (WT 1회 + MUT) forward:

```bash
# 먼저 8개로 smoke 확인 (1분)
python dump_diff_emb.py --device cuda --limit 8 --out data/diff_emb_raw_smoketest.pt

# 전체 dump (GPU에서 대략 5~15분)
python dump_diff_emb.py --device cuda --batch-size 32
```

출력: `data/diff_emb_raw.pt` (~30MB, **CPU 텐서로 저장**되므로 로컬로 가져와도 그대로 로드됨)

### 5-2. 배선 확인

```bash
python smoke_test.py
```

M1~M4를 실배치 1개씩 forward(+M3/M4는 backward)해서 shape / NaN / concat_dim /
LoRA gradient 흐름을 검증합니다. ⚠️ 아래 §6의 블로커 때문에 지금은 실패합니다.

### 5-3. 단위 테스트

```bash
python -m pytest tests/ -q
```

---

## 6. 지금 상태에서 막히는 지점 (서버 문제 아님, 코드 쪽 미완성)

1. **`data/structure/results/rad51c_meta_X.npy`가 없습니다.**
   `smoke_test.py:33`과 `model.py`의 `compute_dims()`가 이 파일을 읽는데,
   저장소 어디에서도 생성하지 않습니다. `rad51c_meta.csv`(var_type, anchor_pos,
   del_len, ins_len …)는 있으니, 여기서 meta 6차원(변이 타입 one-hot + 연속값)을
   만드는 스크립트가 하나 필요합니다. 어떤 컬럼을 어떻게 인코딩할지는 연구 판단이라
   임의로 정하지 않았습니다.

2. **training loop가 아직 없습니다.** `model.py`는 모델 정의까지고,
   optimizer / loss / epoch 루프 / 체크포인트 저장 / 평가(Spearman, AUC)는
   미구현입니다. configs의 `lr.head` / `lr.lora` / `gradient_accumulation` /
   `precision` 같은 키를 실제로 읽어 쓰는 코드가 없습니다.

즉 지금 서버에서 의미 있게 돌릴 수 있는 건 **5-1 (diff embedding dump)** 와
**5-3 (pytest)** 입니다. 위 두 개가 채워지면 5-2와 학습이 돌아갑니다.

---

## 7. 결과물 가져오기

- 작은 파일: VS Code Explorer에서 드래그 앤 드롭, 또는 git commit & push
- 큰 파일: Nextcloud (`https://nextcloud.nrp-nautilus.io`) 또는 S3-compatible storage.
  Rclone 사용법은 `약리학교실_NRP_가이드_rclone.pdf` 참조.

---

## 8. 자주 걸리는 것

| 증상 | 원인 / 해결 |
|---|---|
| Coder 로그인 후 Permission Denied | namespace 미승인. 관리자(민준구/조우성/김승민/여주혜)에게 포탈 이메일 전달 |
| `nvidia-smi` 없음 | 워크스페이스 파라미터에서 GPU가 0. Stop → 파라미터 수정 → Start |
| 체크포인트 다운로드 중 `No space left on device` | §2 볼륨 증설 |
| M3/M4에서 CUDA OOM | GPU Type을 24GB급으로 지정, 또는 `configs/m3.yaml`의 `batch_size` 축소 |
| 다음날 접속하니 워크스페이스가 꺼져 있음 | 24시간 무활동 자동 종료. 다시 Start하면 파일은 그대로 |
| `ModuleNotFoundError: yaml` | `source .venv/bin/activate` 안 함 |
