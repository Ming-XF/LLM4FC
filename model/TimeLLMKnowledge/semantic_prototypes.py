"""Two-level semantic prototype banks for the three EEG knowledge families.

The frozen language vocabulary is used only as a deterministic semantic
candidate source. The first-level pool is either cosine-retrieved from the
frozen vocabulary against the aspect phrase (default) or harvested from
external DeepSeek API keyword lists (--anchor_select api, cached to
api_anchor_keywords.json; the cache file is authoritative — if it exists the
API is not called again). A first candidate pool is linearly fused into
several aspect-specific anchors; each anchor then owns a second description
pool which is compressed into the prototypes consumed by reprogramming.
"""

import fcntl
import json
import os
import re
import time

import torch
import torch.nn as nn


_EPS = 1e-8


def _group_average_init(weight):
    """Initialize each output as an average over a sorted candidate group."""
    outputs, inputs = weight.shape
    with torch.no_grad():
        weight.zero_()
        for i in range(outputs):
            start = (i * inputs) // outputs
            end = ((i + 1) * inputs) // outputs
            end = max(end, start + 1)
            weight[i, start:end] = 1.0 / float(end - start)


def _row_normalize(x):
    return x / x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(_EPS)


def _load_api_key():
    """读取 DeepSeek API key：环境变量 DEEPSEEK_API_KEY 优先，其次 ~/.deepseek_key。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    key_path = os.path.expanduser("~/.deepseek_key")
    if os.path.exists(key_path):
        with open(key_path) as f:
            key = f.read().strip()
        if key:
            return key
    raise ValueError(
        "未找到 DeepSeek API key：设置环境变量 DEEPSEEK_API_KEY 或写入 ~/.deepseek_key")


_API_PROMPT_TEMPLATE = (
    "Output exactly {num_keywords} key domain terms (nouns or technical "
    "phrases) about {phrase}. One term per line. No full sentences, no "
    "stopwords, no conjunctions or articles (no 'and', 'if', 'the'), "
    "no numbering, no explanation. Only the terms."
)
_API_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "api_anchor_keywords.json")


def _parse_keywords(text):
    """把 API 响应解析为关键词列表：按行/逗号切分，去首尾编号标点，大小写无关去重。"""
    seen, out = set(), []
    for part in re.split(r"[\n,;]+", text or ""):
        term = re.sub(r"^[\d\.\s\-•*#]+", "", part).strip()
        term = re.sub(r"[\s\-•*#\.]+$", "", term).strip()
        if len(term) < 2 or term.lower() in seen:
            continue
        seen.add(term.lower())
        out.append(term)
    return out


def _call_deepseek_api(phrase, api_key, num_keywords):
    """调 DeepSeek API 生成单方面关键词（temperature 0，瞬时故障重试 3 次）。"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    last_exc = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content":
                           _API_PROMPT_TEMPLATE.format(
                               phrase=phrase, num_keywords=num_keywords)}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # 网络/限流瞬时故障重试
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek API 调用失败: {last_exc}")


def _load_or_fetch_keywords(phrases, num_keywords):
    """缓存文件优先；缺失（或部分缺失、或缓存的 num_keywords 小于当前
    要求）的方面调 API 补全并写回缓存。

    用文件锁串行化多 rank 并发：只有一个进程调 API，其余等锁后直接读
    缓存，保证两 rank 锚点内容一致。要强制重新生成，删除缓存文件即可。
    """
    os.makedirs(os.path.dirname(_API_CACHE_PATH), exist_ok=True)
    with open(_API_CACHE_PATH + ".lock", 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        data = {"keywords": {}}
        if os.path.exists(_API_CACHE_PATH):
            with open(_API_CACHE_PATH) as f:
                data = json.load(f)
        missing = [p for p in phrases
                   if p not in data.get("keywords", {})
                   or data.get("num_keywords", 0) < num_keywords]
        if missing:
            api_key = _load_api_key()
            for phrase in missing:
                keywords = _parse_keywords(
                    _call_deepseek_api(phrase, api_key, num_keywords))
                data["keywords"][phrase] = keywords
                print(f"[gen-anchor] API 生成 '{phrase[:36]}' -> "
                      f"{len(keywords)} keywords")
            data.setdefault("generator", "deepseek-chat")
            data.setdefault("prompt_template", _API_PROMPT_TEMPLATE)
            data["num_keywords"] = num_keywords
            with open(_API_CACHE_PATH, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            source = "deepseek-chat API"
        else:
            source = "cache"
            print(f"[gen-anchor] 缓存命中，未调用 API: {_API_CACHE_PATH}")
        return {p: data["keywords"][p] for p in phrases}, source


def _dedup_token_ids(tokenizer, keywords):
    """关键词按序 tokenize 后对 token 去重（保持首次出现顺序）。"""
    ids, seen = [], set()
    for kw in keywords:
        for tid in tokenizer.encode(" " + kw, add_special_tokens=False):
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
    return ids


def _read_keywords_from_cache(phrases):
    """只读缓存（评估端占位用），缺失或不全时报错而不是调 API。"""
    if not os.path.exists(_API_CACHE_PATH):
        raise ValueError(
            f"关键词缓存不存在，无法确定锚点池 K: {_API_CACHE_PATH}")
    with open(_API_CACHE_PATH) as f:
        data = json.load(f)
    missing = [p for p in phrases if p not in data.get("keywords", {})]
    if missing:
        raise ValueError(
            f"关键词缓存缺少方面 {missing}: {_API_CACHE_PATH}")
    return data["keywords"]


def api_keyword_pool_size(tokenizer, phrases, num_keywords):
    """评估端占位用：只读缓存计算 K（三方面去重 token 数的最大值），不调 API。

    训练与评估在同一缓存下得到相同 K，保证 checkpoint 结构一致。
    """
    keywords = _read_keywords_from_cache(phrases)
    K = 0
    for phrase in phrases:
        K = max(K, len(_dedup_token_ids(
            tokenizer, keywords[phrase][:num_keywords])))
    return K


def api_keyword_candidate_embeddings(word_embeddings, tokenizer, phrases,
                                     num_keywords, gen_log_path=None):
    """第一级候选：外部 DeepSeek API 关键词（缓存文件优先），本地 tokenize
    后按首次出现顺序对 token 去重，K = 三方面去重 token 数的最大值（短
    的方面用零向量填充对齐）。K 是缓存文件的确定性函数：训练端生成与
    评估端占位（api_keyword_pool_size）在同一缓存下算出相同 K，保证
    checkpoint 结构一致。返回 (candidate_embeddings (A, K, d_llm) 位于
    CPU 的 fp32 张量, K)。
    """
    keywords, source = _load_or_fetch_keywords(phrases, num_keywords)
    used = {p: keywords[p][:num_keywords] for p in phrases}
    emb = word_embeddings.weight.detach().float()
    aspect_ids = []
    for phrase in phrases:
        ids = _dedup_token_ids(tokenizer, used[phrase])
        aspect_ids.append(ids)
        print(f"[gen-anchor] '{phrase[:36]}' -> {len(used[phrase])} keywords, "
              f"{len(ids)} unique tokens ({source})")
    K = max(len(ids) for ids in aspect_ids)
    candidates = []
    for ids in aspect_ids:
        rows = emb[torch.tensor(ids, dtype=torch.long)]       # (k_i, d)
        if len(ids) < K:
            # 零填充：0 向量对融合层贡献恒为 0，天然被忽略，
            # 不再引入 EOS 方向的退化槽位
            pad = torch.zeros(K - len(ids), rows.shape[1],
                              dtype=rows.dtype)
            rows = torch.cat([rows, pad], dim=0)
        candidates.append(rows)
    if gen_log_path:
        _append_gen_report(gen_log_path, phrases, used, source, K)
    return torch.stack(candidates), K


def _append_gen_report(gen_log_path, phrases, keywords_by_phrase, source, K):
    """把方面关键词完整记录追加进本次运行的日志文件（供人工检查）。

    生成发生在 init_logger 之前，logger 尚不可用，只能直接写文件；
    两 rank 各写一份，失败仅告警不阻断训练。
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(gen_log_path)),
                    exist_ok=True)
        with open(gen_log_path, 'a') as f:
            for phrase in phrases:
                kw = keywords_by_phrase[phrase]
                f.write(f"[gen-anchor] '{phrase}' -> {len(kw)} keywords "
                        f"({source}):\n")
                f.write(f"  keywords: {kw}\n")
                f.write(f"  text: {', '.join(kw)}\n")
            f.write(f"[gen-anchor] anchor pool K = {K}\n")
    except OSError as exc:
        print(f"[gen-anchor] 写入日志失败: {exc}")


class SemanticPrototypeBank(nn.Module):
    """Build shared two-level semantic banks for three aspect queries.

    Shapes:
      first candidates:       (A, K, d_llm)
      refined anchors:        (A, N, d_llm)
      descriptions:           (A, N, M, d_llm)
      output prototypes:      (A, N, P, d_llm)

    Candidate ids are fixed after initialization. The two Linear layers use
    the existing signed candidate-space mapping style and remain trainable.
    anchor_candidate_embeddings 非 None 时跳过第一级余弦检索，直接使用传入
    的生成候选嵌入。
    """

    def __init__(self, phrase_bank, word_embeddings, tokenizer,
                 anchor_candidates=1000, num_anchors=10,
                 description_candidates=1000, num_prototypes=100,
                 similarity="centered_cosine", aspect_mix=0.6,
                 anchor_fusion_init=0.1,
                 anchor_candidate_embeddings=None):
        super().__init__()
        num_aspects, d_llm = phrase_bank.shape
        vocab_size, emb_dim = word_embeddings.shape
        if emb_dim != d_llm:
            raise ValueError("词表 embedding 维度与短语锚不一致")
        if description_candidates <= 0 or num_prototypes <= 0:
            raise ValueError(
                "description_candidates 和 num_prototypes 必须为正")
        if num_prototypes > description_candidates:
            raise ValueError(
                "num_prototypes 不能大于 description_candidates")
        if similarity not in {"cosine", "centered_cosine"}:
            raise ValueError(
                "similarity 必须是 cosine 或 centered_cosine")
        if not 0.0 <= aspect_mix <= 1.0:
            raise ValueError("aspect_mix 必须位于 [0, 1]")
        if anchor_fusion_init < 0:
            raise ValueError("anchor_fusion_init 不能为负")

        self.num_aspects = num_aspects
        self.num_anchors = num_anchors
        self.description_candidates = description_candidates
        self.num_prototypes = num_prototypes
        self.similarity = similarity
        self.aspect_mix = aspect_mix
        self.anchor_fusion = nn.Parameter(
            torch.tensor(float(anchor_fusion_init)))

        # Candidate retrieval is done once in fp32 so the semantic bank stays
        # deterministic and domain-independent during training.
        # Keep a low-precision copy for the persistent banks. Similarity
        # retrieval is still performed in fp32 below for stable top-k ids.
        embedding_storage_dtype = word_embeddings.dtype
        emb = word_embeddings.detach().float()
        if similarity == "centered_cosine":
            center = emb.mean(dim=0, keepdim=True)
            emb_for_sim = emb - center
            phrase_for_sim = phrase_bank.detach().float() - center
        else:
            center = None
            emb_for_sim = emb
            phrase_for_sim = phrase_bank.detach().float()
        emb_n = _row_normalize(emb_for_sim)
        phrase_n = _row_normalize(phrase_for_sim)

        special_ids = sorted({
            int(i) for i in tokenizer.all_special_ids
            if i is not None and 0 <= int(i) < vocab_size
        })
        if len(special_ids) >= vocab_size:
            raise ValueError("词表中没有可用于语义候选的普通 Token")
        special = torch.tensor(
            special_ids, dtype=torch.long, device=emb.device)
        valid_count = vocab_size - special.numel()

        if anchor_candidate_embeddings is None:
            # ── 第一级（默认对照）：方面查询余弦 top-K 预选 ──
            if anchor_candidates <= 0 or num_anchors <= 0:
                raise ValueError("anchor_candidates 和 num_anchors 必须为正")
            if num_anchors > anchor_candidates:
                raise ValueError("num_anchors 不能大于 anchor_candidates")
            if anchor_candidates > valid_count:
                raise ValueError(
                    f"anchor_candidates={anchor_candidates} 超过可用词表 "
                    f"Token 数 {valid_count}")
            first_scores = phrase_n @ emb_n.T
            if special.numel():
                first_scores[:, special] = -float("inf")
            first_ids = first_scores.topk(anchor_candidates, dim=-1).indices
            first_embeddings = emb[first_ids]
            self.register_buffer("anchor_candidate_ids", first_ids.long())
        else:
            # ── 第一级（generate）：LLM 续写生成 token 的词嵌入，跳过余弦检索 ──
            first_embeddings = anchor_candidate_embeddings.detach().float()
            if (first_embeddings.shape[0] != num_aspects
                    or first_embeddings.shape[2] != d_llm):
                raise ValueError(
                    "anchor_candidate_embeddings 形状须为 "
                    f"({num_aspects}, K, {d_llm})，got "
                    f"{tuple(first_embeddings.shape)}")
            anchor_candidates = first_embeddings.shape[1]
            if num_anchors <= 0 or num_anchors > anchor_candidates:
                raise ValueError(
                    f"num_anchors={num_anchors} 须满足 0 < num_anchors ≤ "
                    f"生成候选数 {anchor_candidates}")
        self.anchor_candidates = anchor_candidates
        self.register_buffer(
            "anchor_candidate_embeddings",
            first_embeddings.to(dtype=embedding_storage_dtype))

        # First mapping: K candidates -> N refined knowledge anchors.
        self.anchor_mapper = nn.Linear(
            anchor_candidates, num_anchors, bias=False)
        _group_average_init(self.anchor_mapper.weight)
        with torch.no_grad():
            refined_init = torch.einsum(
                "nk,akd->and", self.anchor_mapper.weight.float(),
                first_embeddings)

        # The second-level ids are fixed after initialization. The mapping
        # itself is trainable, but retrieval does not jump during training.
        refined_for_sim = refined_init
        if similarity == "centered_cosine":
            refined_for_sim = refined_for_sim - center
        refined_n = _row_normalize(refined_for_sim)
        joint_query = (
            aspect_mix * phrase_n.unsqueeze(1)
            + (1.0 - aspect_mix) * refined_n
        )
        joint_query = _row_normalize(joint_query)
        second_scores = torch.einsum(
            "and,vd->anv", joint_query, emb_n)
        if special.numel():
            second_scores[:, :, special] = -float("inf")
        if description_candidates > valid_count:
            raise ValueError(
                f"description_candidates={description_candidates} 超过可用词表 "
                f"Token 数 {valid_count}")
        description_ids = second_scores.topk(
            description_candidates, dim=-1).indices
        description_embeddings = emb[description_ids]
        self.register_buffer("description_ids", description_ids.long())
        self.register_buffer(
            "description_embeddings",
            description_embeddings.to(dtype=embedding_storage_dtype))

        # Second mapping: M descriptions -> P description prototypes.
        self.description_mapper = nn.Linear(
            description_candidates, num_prototypes, bias=False)
        _group_average_init(self.description_mapper.weight)

    def refined_anchors(self):
        """Return trainable refined anchors with shape ``(A, N, d_llm)``."""
        return torch.einsum(
            "nk,akd->and", self.anchor_mapper.weight,
            self.anchor_candidate_embeddings.to(
                dtype=self.anchor_mapper.weight.dtype))

    def forward(self, aspect_ids=None):
        """Return description prototypes as ``(A, N, P, d_llm)``."""
        prototypes = torch.einsum(
            "pm,anmd->anpd", self.description_mapper.weight,
            self.description_embeddings.to(
                dtype=self.description_mapper.weight.dtype))
        # Keep the first-level fused knowledge trainable in the forward path;
        # the frozen second-level ids only determine its semantic neighborhood.
        prototypes = prototypes + self.anchor_fusion * self.refined_anchors().unsqueeze(2)
        if aspect_ids is not None:
            prototypes = prototypes[aspect_ids]
        return prototypes
