# deepRAD51C — 구조 메모

## 1. ESM2.forward 시그니처 및 repr_layers 인덱스 규약

파일: `esm/model/esm2.py:77`

```python
def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False)
    -> {"logits": Tensor, "representations": dict[int, Tensor], ...}
```

**인덱스 규약 (repr_layers에 넘길 값)**

| 인덱스 | 내용 | shape | 비고 |
|--------|------|-------|------|
| 0 | 토큰 임베딩 (embed_tokens + dropout scaling 후) | (B, T, E) | transformer 진입 전 |
| 1–33 | TransformerLayer i 출력 (layers[i-1]) | (B, T, E) | layer_idx+1 로 키 저장 |
| 33 (마지막) | emb_layer_norm_after 적용 후 덮어씀 | (B, T, E) | LayerNorm이 적용된 표현 |

핵심 코드 (L97–128):
- `repr_layers = set(repr_layers)` — 집합으로 변환
- 레이어 0: `if 0 in repr_layers: hidden_representations[0] = x` (transpose 전, B×T×E)
- 레이어 1-33: `if (layer_idx + 1) in repr_layers: hidden_representations[layer_idx + 1] = x.transpose(0, 1)`
- 마지막 레이어: `emb_layer_norm_after` 적용 뒤 dict 항목을 **덮어쓴다** (L127–128)
  → `repr_layers=[33]`으로 추출한 표현은 LayerNorm이 포함됨

ESM-2 650M 기본값: `num_layers=33`, `embed_dim=1280`, `attention_heads=20`

---

## 2. MultiheadAttention — Q/V projection 위치 및 fast-path 분기

파일: `esm/multihead_attention.py`

### Projection 모듈 위치

```python
# multihead_attention.py:109-113
self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)   # (1280→1280)
self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)   # (1280→1280)
self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)   # (1280→1280)
self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) # (1280→1280)
```

접근 경로: `model.layers[i].self_attn.{q_proj, k_proj, v_proj, out_proj}`

### F.multi_head_attention_forward fast-path 조건 (L196-206)

fast-path를 타는 조건 (모두 True여야 함):
1. `not self.rot_emb` — rotary embedding 없음
2. `self.enable_torch_version` — F.multi_head_attention_forward 존재
3. `not self.onnx_trace`
4. `incremental_state is None`
5. `not static_kv`
6. `not torch.jit.is_scripting()`
7. `not need_head_weights`

### ESM-2에서 fast-path는 이미 우회됨

`modules.py:57`에서 `TransformerLayer`를 `use_rotary_embeddings=True`로 생성하므로
`self.rot_emb = RotaryEmbedding(...)` (truthy) → 조건 1이 False → **항상 slow-path**.

따라서 LoRA 주입 시 `F.multi_head_attention_forward`를 따로 우회할 필요 없이,
**`q_proj`/`v_proj`를 직접 교체**하면 된다.

slow-path에서의 projection 호출 순서 (L242-260):
```python
q = self.q_proj(query)
k = self.k_proj(key)
v = self.v_proj(value)
```

---

## 오늘 구현 범위

**WT-difference embedding ~ BottleneckMLP까지**

포함:
- ESM-2 frozen backbone에서 `repr_layers=[33]`으로 WT/mutant 각각 hidden state 추출
- difference embedding 계산: `z = h_mut - h_wt` (레지듀 토큰 위치)
- BottleneckMLP (projection → ReLU → un-projection 또는 유사 구조)

제외 (다음 단계):
- concat 방식 (차이 외 다른 결합)
- training loop
- classification head
- LoRA 파라미터 주입 (구조 파악만 완료)
