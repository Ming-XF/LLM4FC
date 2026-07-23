"""COT target templates and knowledge anchor contrastive loss for LDDE2th.

Proto-conditional COT target: evidence summary (per-sample, from proto/M)
+ pathophysiological interpretation (per-sample, randomly sampled knowledge
fragments from activated prototypes).  No class label appears in the target
text — zero label leakage.

Knowledge anchor contrastive loss: pulls the COT text hidden representation
toward a pre-encoded class-specific knowledge anchor, using the label ONLY
to select the positive anchor (never in the generation target).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Proto Knowledge Fragments (16 prototypes × 3-5 fragments each)
#
# Each fragment (≤20 tokens) explains the clinical/neuroscientific meaning
# of that prototype's dysfunction from a different angle.  At COT target
# construction time:
#   1. Top-3 activated prototypes by |act| are selected.
#   2. All fragments from these 3 prototypes are pooled (9-15 fragments).
#   3. 3 fragments are randomly sampled from the pool.
#
# This gives per-sample diversity both from which prototypes activate AND
# from random sampling within the top-3 fragment pool.
# ═══════════════════════════════════════════════════════════════════════════════

PROTO_KNOWLEDGE_FRAGMENTS = {
    # ── Proto 0: DMN后部连接完整性 (high=normal → low=abnormal) ──
    0: [
        "后扣带回楔前叶失连接提示默认网络核心节点信息传递效率下降，",
        "DMN后部连接降低是AD最早可检测的脑网络电生理改变之一，",
        "后扣带回tau蛋白沉积导致默认网络核心节点间功能失耦联，",
        "DMN后部完整性下降损害自传体记忆提取和内部心理模拟能力，",
    ],

    # ── Proto 1: 内侧颞叶-海马耦合 (high=normal → low=abnormal) ──
    1: [
        "海马旁回内嗅皮层同步性减弱与内侧颞叶tau沉积空间分布一致，",
        "内侧颞叶功能耦合下降直接损害新近情景记忆编码和提取能力，",
        "海马环路同步性减弱反映兴奋抑制失衡提示突触可塑性障碍，",
        "内侧颞叶是神经退行性变最早受累结构其功能下降具早期诊断敏感性，",
        "内嗅皮层是tau病理向新皮层扩散关键枢纽其失连接预示进展风险，",
    ],

    # ── Proto 2: 额顶控制网络代偿激活 (high=compensatory → 代偿增强) ──
    2: [
        "背外侧前额叶募集增强是认知储备机制启动的代偿性电生理标志，",
        "额顶网络代偿激活反映前额叶正在对抗后部网络的进行性病理负荷，",
        "前额叶过度募集维持了认知表现但代偿效率随病理进展可能衰减，",
    ],

    # ── Proto 3: 突显网络内部同步性 (high=normal → low=abnormal) ──
    3: [
        "前岛叶前扣带回同步性减弱导致默认与任务网络间切换能力下降，",
        "突显网络功能下降使大脑无法有效检测和定向显著内外环境刺激，",
        "前岛叶功能连接减弱与内感受觉察障碍和情绪调节困难相关，",
        "突显网络是DMN与CEN切换枢纽其障碍导致认知灵活性受损，",
    ],

    # ── Proto 4: 视觉网络完整性 (high=normal → low=abnormal) ──
    4: [
        "枕叶初级与联络视觉皮层整合降低可见于后部皮质萎缩相关障碍，",
        "视觉网络完整性下降可能提示后部皮质萎缩谱系改变需关注扩散趋势，",
        "枕叶功能连接减弱影响视觉空间加工和物体识别等高级视觉功能，",
    ],

    # ── Proto 5: 全脑连接弥散度 (high=abnormal → high=bad) ──
    5: [
        "全脑连接弥散度升高提示功能网络特异度下降和去分化趋势，",
        "网络去分化是脑老化和神经退行性变的共性特征反映边界模糊化，",
        "高弥散度表明各网络间信息串扰增加功能分离效率降低，",
    ],

    # ── Proto 6: 半球间同源连接 (high=compensatory → 代偿增强) ──
    6: [
        "半球间胼胝体连接增强反映健侧大脑分担认知负荷，",
        "半球间同源连接增强是单侧病理负荷增加时脑网络重组的积极信号，",
        "双侧信息传递效率代偿性提升有助于维持整体认知功能水平，",
    ],

    # ── Proto 7: 前-后长程连接效率 (high=normal → low=abnormal) ──
    7: [
        "额叶顶枕叶长程连接效率下降提示大规模脑网络整合能力受损，",
        "额枕长程连接是自上而下注意调控基础其减弱导致执行控制力下降，",
        "长程连接效率降低使感知信息前馈与执行调控反馈的平衡失调，",
    ],

    # ── Proto 8: 左侧语言网络激活 (high=compensatory → 代偿增强) ──
    8: [
        "左侧额颞语言区连接增强与语义加工和言语记忆代偿维持有关，",
        "左侧额颞语言网络代偿性激活是认知功能保持的积极标志，",
        "语言相关通路对神经退行性变存在一定韧性提示代偿储备充足，",
    ],

    # ── Proto 9: 感觉运动网络完整性 (high=normal → low=abnormal) ──
    9: [
        "感觉运动网络相对保留是区分AD与其他类型痴呆的关键鉴别特征，",
        "中央沟周围皮层功能连接在多种神经退行性疾病中均为最晚受累系统，",
        "SMN稳定性提示脑网络退行遵循特定时空扩散梯度而非全局均匀下降，",
        "感觉运动皮层完整性是维持日常生活活动自理能力的网络基础，",
    ],

    # ── Proto 10: 眶额-纹状体回路 (high=normal → low=abnormal) ──
    10: [
        "眶额纹状体回路减弱与奖励加工障碍和动机缺乏行为表现相关，",
        "眶额皮层纹状体投射是奖赏预期和决策偏好的核心功能基质，",
        "眶额回路功能下降可能导致淡漠和社交退缩等非认知行为症状，",
    ],

    # ── Proto 11: 前额叶执行功能网络 (high=normal → low=abnormal) ──
    11: [
        "前额叶执行网络保持良好整合是维持日常独立性的结构支撑，",
        "背外侧前额叶内部功能连接是工作记忆和认知灵活性的核心基质，",
        "前额叶功能下降直接损害计划组织和多任务处理等高级认知能力，",
        "前额叶网络完整性是认知储备的结构基础其保留决定代偿潜力上限，",
    ],

    # ── Proto 12: 边缘系统情绪回路 (high=normal → low=abnormal) ──
    12: [
        "杏仁核前扣带回同步性下降可导致情绪识别和社会认知能力受损，",
        "边缘系统功能减弱与痴呆前期常见的情绪调节障碍和社交退缩相关，",
        "杏仁核眶额协调活动减弱影响情绪刺激评估和适当行为反应生成，",
    ],

    # ── Proto 13: 皮质下-皮层连接 (high=normal → low=abnormal) ──
    13: [
        "丘脑皮层中继效率减弱影响持续性注意维持和信息处理速度，",
        "丘脑基底节皮层环路是意识水平和觉醒状态的核心结构基础，",
        "皮质下皮层连接下降可导致精神运动迟缓和认知加工速度减慢，",
    ],

    # ── Proto 14: 注意网络内部耦合 (high=normal → low=abnormal) ──
    14: [
        "背侧注意网络内部耦合减弱导致定向注意和空间加工能力下降，",
        "额叶眼区顶内沟协同减弱使视觉搜索和目标检测效率降低，",
        "注意网络同步性是选择性注意的核心机制其下降影响日常任务表现，",
    ],

    # ── Proto 15: 代偿性跨网络重组 (high=compensatory → 代偿增强) ──
    15: [
        "代偿性跨网络重组增强是脑网络对抗病理损伤的适应性可塑性信号，",
        "跨网络连接增强提示神经网络仍保有损伤条件下资源重新配置能力，",
        "大尺度功能重组是认知储备转化为实际功能代偿的重要神经途径，",
        "跨网络重组受神经营养因子等分子通路调控反映脑可塑性储备水平，",
        "跨网络重组程度与病理负荷的匹配关系决定代偿能否持续有效维持，",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Knowledge Anchor Texts (4 classes)
#
# These short (~60-90 token) class-specific descriptions encode core
# neuroscientific diagnostic knowledge.  They are used ONLY by the
# contrastive loss to pull COT hidden representations toward the correct
# class's semantic anchor.  They NEVER appear in the generation target.
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_ANCHORS = {
    3: ("阿尔茨海默病以DMN后部功能连接显著降低为核心电生理特征，"
        "后扣带回楔前叶tau蛋白沉积导致默认网络核心节点失连接，"
        "内侧颞叶海马旁回通路同步受累反映神经退行性沿功能网络扩散，"
        "感觉运动网络相对保留是区分其他类型痴呆的重要鉴别依据，"
        "额叶代偿性增强反映早期认知储备机制的有限动员"),

    2: ("轻度认知障碍呈现DMN后部连接轻中度降低与额顶网络代偿性增强并存的双重模式，"
        "后扣带回功能连接部分保留使网络拓扑尚未完全崩溃，"
        "背外侧前额叶过度募集反映认知储备正在积极对抗病理进展，"
        "颞叶顶叶跨网络连接下降幅度小于典型AD表明神经退行性处于过渡阶段"),

    1: ("主观记忆障碍整体网络拓扑接近正常但已出现亚临床改变，"
        "DMN后部连接仅轻微下降尚未达统计显著性但方向性提示早期风险，"
        "额叶执行网络早期代偿性增强表明主观认知抱怨可能伴随客观网络层面改变，"
        "全脑功能连接整合度轻微下降且各网络间耦合基本正常"),

    0: ("正常认知全脑功能连接整合度良好且三大核心网络DMN-SN-CEN内外部耦合模式正常，"
        "长程与短程功能连接分布均衡无特定脑区显著性失连接，"
        "左右半球同源连接对称性保持良好，"
        "各网络间未见代偿性异常激活或病理性抑制模式"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Knowledge Anchor Contrastive Loss Module
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeAnchorLoss(nn.Module):
    """Supervised contrastive loss aligning COT hidden states to class anchors.

    Pipeline:
      1. Pre-encode 4 knowledge-anchor texts → mean-pooled LLM embeddings
         → stored as a frozen buffer (4, llm_dim).
      2. At each forward: project both the (mean-pooled) COT hidden states
         AND the anchor embeddings through the same learnable projection head
         → (proj_dim).  L2-normalize both.
      3. Cosine-similarity matrix (B, 4) → cross-entropy with labels.

    The label is used ONLY to select the positive anchor column in the
    similarity matrix.  It never appears in the generation target text,
    so there is no label leakage into the autoregressive CE loss.

    Args:
        llm_dim:      LLM hidden dimension (ChatGLM-6B = 4096).
        proj_dim:     projection bottleneck dimension.
        temperature:  contrastive temperature.
        tokenizer:    ChatGLM tokenizer (for pre-encoding anchors).
        wte:          LLM word-token-embeddings table.
        anchor_texts: dict {class_idx: anchor_text_str}.
    """

    def __init__(self, llm_dim=4096, proj_dim=256, temperature=0.07,
                 tokenizer=None, wte=None, anchor_texts=None):
        super().__init__()
        self.llm_dim = llm_dim
        self.proj_dim = proj_dim
        self.temperature = temperature

        # ── Projection head: llm_dim → llm_dim//4 → proj_dim ──
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 4),
            nn.ReLU(),
            nn.Linear(llm_dim // 4, proj_dim),
        )

        # ── Pre-encode knowledge anchors ──
        if tokenizer is not None and wte is not None and anchor_texts is not None:
            self._init_anchors(tokenizer, wte, anchor_texts)

    def _init_anchors(self, tokenizer, wte, anchor_texts):
        """Tokenize each anchor text, mean-pool its LLM embeddings, store as buffer."""
        anchor_embs = []
        for cls_idx in sorted(anchor_texts.keys()):
            text = anchor_texts[cls_idx]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            with torch.no_grad():
                emb = wte(torch.tensor(token_ids, dtype=torch.long))
                emb = emb.mean(dim=0)                     # mean-pool → (llm_dim,)
            anchor_embs.append(emb)
        anchors = torch.stack(anchor_embs)                # (4, llm_dim)
        self.register_buffer('anchor_embeddings', anchors)

    def forward(self, text_hidden, labels):
        """Compute supervised contrastive loss.

        Args:
            text_hidden: (B, T, llm_dim)  hidden states at COT target positions.
            labels:      (B,)             ground-truth class indices [0..3].

        Returns:
            scalar loss.
        """
        # ── Mean-pool over COT token positions ──
        text_vec = text_hidden.mean(dim=1)                     # (B, llm_dim)

        # ── Project + L2-normalize ──
        text_vec = self.proj(text_vec)                         # (B, proj_dim)
        text_vec = F.normalize(text_vec, dim=-1)

        # ── Project anchors through the same head ──
        anchors = self.anchor_embeddings.to(
            dtype=text_vec.dtype, device=text_vec.device)
        anchors_proj = self.proj(anchors)                      # (4, proj_dim)
        anchors_proj = F.normalize(anchors_proj, dim=-1)

        # ── Cosine similarity + InfoNCE ──
        sim = (text_vec @ anchors_proj.T) / self.temperature   # (B, 4)
        loss = F.cross_entropy(sim, labels)
        return loss
