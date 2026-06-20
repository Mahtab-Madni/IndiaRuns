
#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon Candidate Ranking
Usage:
    python rank.py --candidates candidates.jsonl --team-id NoBlackBox --out NoBlackBox.csv
"""
import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import date
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
TODAY = date.today()

# ── JD Signals ──────────────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform
Pune Noida India hybrid 5 to 9 years experience applied machine learning production
embeddings retrieval ranking LLM fine-tuning recommendation system search
sentence transformers BGE E5 OpenAI embeddings vector database hybrid search
Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS dense retrieval
BM25 hybrid retrieval ranking evaluation NDCG MRR MAP A/B testing
python production code quality retrieval quality embedding drift index refresh
learning to rank XGBoost neural ranking LLM integration fine-tuning LoRA QLoRA PEFT
product company startup early stage NLP information retrieval recommendation
candidate job description matching talent acquisition recruiting platform
evaluation framework offline benchmark online experiment recruiter feedback
ship fast scrappy product engineering shipper not researcher
distributed systems large scale inference optimization open source contributions
"""

REQUIRED_SKILLS = [
    "embedding", "embeddings", "vector", "retrieval", "ranking", "search",
    "recommendation", "nlp", "information retrieval", "sentence transformer",
    "bge", "e5", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "bm25", "dense retrieval", "hybrid search",
    "machine learning", "deep learning", "pytorch", "tensorflow", "transformers",
    "llm", "fine-tuning", "rag", "lora", "qlora", "peft",
    "ndcg", "mrr", "map", "a/b testing", "learning to rank", "xgboost",
    "python", "production ml", "applied ml"
]

NICE_TO_HAVE_SKILLS = [
    "open source", "github", "distributed systems", "inference optimization",
    "hr tech", "recruiting", "talent", "marketplace", "lora", "qlora"
]

RED_FLAG_TITLES = [
    "marketing manager", "hr manager", "content writer", "business analyst",
    "project manager", "product manager", "sales", "finance", "accountant",
    "computer vision", "speech", "robotics", "qa engineer", "tester"
]

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "niit"
]

PREFERRED_LOCATIONS = ["pune", "noida", "delhi", "ncr", "hyderabad", "mumbai", "bangalore", "bengaluru"]
YOE_IDEAL_MIN = 5
YOE_IDEAL_MAX = 9

WEIGHTS = {
    "semantic_score":          0.12,
    "retrieval_depth":         0.10,
    "required_skills_hit":     0.08,
    "eval_depth":              0.07,
    "llm_depth":               0.05,
    "python_strength":         0.04,
    "yoe_fit":                 0.06,
    "ml_career_fraction":      0.05,
    "product_company_fraction": 0.05,
    "current_role_ml":         0.04,
    "has_production_ml":       0.04,
    "tenure_stability":        0.02,
    "recency":                 0.05,
    "open_to_work":            0.03,
    "recruiter_response":      0.03,
    "notice_score":            0.02,
    "interview_reliability":   0.02,
    "recruiter_interest":      0.02,
    "github_score":            0.02,
    "nice_skills_hit":         0.01,
    "edu_tier":                0.01,
    "location_fit":            0.03,
    "work_mode_fit":           0.01,
    "salary_fit":              0.01,
    "completeness":            0.01,
    "verified":                0.01,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, f"Weights sum to {sum(WEIGHTS.values()):.3f}"

PENALTIES = {
    "red_flag_title":  0.80,
    "consulting_only": 0.50,
    "research_heavy":  0.40,
    "is_honeypot":     0.95,
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def text_lower(text: str) -> str:
    return str(text).lower()

def keyword_hit_rate(text: str, keywords: list) -> float:
    if not keywords or not text:
        return 0.0
    t = text_lower(text)
    return sum(1 for kw in keywords if kw in t) / len(keywords)

def days_since(date_str: str) -> int:
    try:
        return (TODAY - date.fromisoformat(date_str)).days
    except (ValueError, TypeError):
        return 9999

def yoe_score(yoe: float) -> float:
    if yoe < 2:            return 0.1
    if yoe < YOE_IDEAL_MIN: return 0.5 + 0.5 * (yoe - 2) / (YOE_IDEAL_MIN - 2)
    if yoe <= YOE_IDEAL_MAX: return 1.0
    if yoe <= 12:          return 1.0 - 0.3 * (yoe - YOE_IDEAL_MAX) / (12 - YOE_IDEAL_MAX)
    return 0.4

def location_score(c: dict) -> float:
    loc     = text_lower(c.get("profile", {}).get("location", ""))
    country = text_lower(c.get("profile", {}).get("country", ""))
    signals = c.get("redrob_signals", {})
    relocate = signals.get("willing_to_relocate", False)
    if any(city in loc for city in PREFERRED_LOCATIONS): return 1.0
    if country == "india" and relocate:                  return 0.75
    if country == "india":                               return 0.5
    if relocate:                                         return 0.3
    return 0.1

def build_candidate_text(c: dict) -> str:
    parts = []
    p = c.get("profile", {})
    parts += [p.get("headline", ""), p.get("summary", ""),
              p.get("current_title", ""), p.get("current_industry", "")]
    for job in c.get("career_history", []):
        parts += [job.get("title", ""), job.get("description", ""),
                  job.get("company", ""), job.get("industry", "")]
    for sk in c.get("skills", []):
        weight = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
            sk.get("proficiency", "beginner"), 1)
        parts.extend([sk.get("name", "")] * weight)
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(parts)

# ── Honeypot Detection ───────────────────────────────────────────────────────
def detect_honeypot(c: dict, honeypot_ids: set) -> tuple:
    reasons = []
    profile  = c.get("profile", {})
    history  = c.get("career_history", [])
    skills   = c.get("skills", [])
    signals  = c.get("redrob_signals", {})

    for job in history:
        try:
            start = date.fromisoformat(job["start_date"])
            duration_months = job.get("duration_months", 0)
            if duration_months > 96 and start.year > 2018:
                reasons.append(f"Implausibly long tenure ({duration_months}mo) starting {start}")
        except (ValueError, TypeError, KeyError):
            pass

    expert_zero = [s["name"] for s in skills
                   if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0]
    if len(expert_zero) >= 3:
        reasons.append(f"Expert proficiency in {len(expert_zero)} skills with 0 months use")

    yoe = profile.get("years_of_experience", 0)
    history_months = sum(j.get("duration_months", 0) for j in history)
    if history_months > 0 and abs(history_months / 12 - yoe) > 5:
        reasons.append(f"YOE mismatch: profile={yoe}y, history={history_months/12:.1f}y")

    expert_skills = [s["name"].lower() for s in skills if s.get("proficiency") == "expert"]
    all_desc = " ".join(j.get("description", "") for j in history).lower()
    if len(expert_skills) > 5 and all(sk not in all_desc for sk in expert_skills):
        reasons.append(f"All {len(expert_skills)} expert skills absent from descriptions")

    try:
        last_active = date.fromisoformat(signals.get("last_active_date", "2000-01-01"))
        if signals.get("open_to_work_flag") and (TODAY - last_active).days > 365:
            reasons.append("open_to_work=True but inactive >365 days")
    except (ValueError, TypeError):
        pass

    for job in history:
        try:
            if date.fromisoformat(job["start_date"]) > TODAY:
                reasons.append(f"Future start date: {job['start_date']}")
        except (ValueError, TypeError, KeyError):
            pass

    return len(reasons) > 0, reasons

# ── Feature Extraction ───────────────────────────────────────────────────────
def extract_features(c: dict, honeypot_ids: set) -> dict:
    p       = c.get("profile", {})
    history = c.get("career_history", [])
    skills  = c.get("skills", [])
    edu     = c.get("education", [])
    signals = c.get("redrob_signals", {})
    cid     = c.get("candidate_id", "")

    full_text = build_candidate_text(c).lower()
    title_lc  = text_lower(p.get("current_title", ""))
    yoe       = float(p.get("years_of_experience", 0))
    feats     = {}

    # Hard disqualifiers
    feats["red_flag_title"] = 1.0 if any(rf in title_lc for rf in RED_FLAG_TITLES) else 0.0
    companies = [text_lower(j.get("company", "")) for j in history]
    consulting_count = sum(1 for co in companies if any(cf in co for cf in CONSULTING_FIRMS))
    feats["consulting_only"] = 1.0 if (len(companies) > 0 and consulting_count == len(companies)) else 0.0

    production_kws = ["production", "deployed", "shipped", "launched", "release",
                      "real users", "live", "scale", "serving", "inference"]
    feats["has_production_ml"] = min(1.0, keyword_hit_rate(full_text, production_kws) * 5)

    research_titles = ["research scientist", "research engineer", "phd intern",
                       "postdoc", "research associate", "academic"]
    research_count = sum(1 for j in history
                         if any(rt in text_lower(j.get("title", "")) for rt in research_titles))
    feats["research_heavy"] = min(1.0, research_count / max(len(history), 1))

    # Skills
    skill_names_lc = [text_lower(s.get("name", "")) for s in skills]
    feats["required_skills_hit"] = keyword_hit_rate(" ".join(skill_names_lc), REQUIRED_SKILLS)
    feats["nice_skills_hit"]     = keyword_hit_rate(full_text, NICE_TO_HAVE_SKILLS)
    expert_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    feats["expert_skill_count_norm"] = min(1.0, expert_count / 10)

    retrieval_terms = ["embedding", "vector", "retrieval", "faiss", "pinecone",
                       "weaviate", "qdrant", "milvus", "elasticsearch", "bm25",
                       "dense retrieval", "hybrid search", "sentence transformer",
                       "semantic search", "ann", "approximate nearest neighbor"]
    feats["retrieval_depth"] = min(1.0, keyword_hit_rate(full_text, retrieval_terms) * 3)

    eval_terms = ["ndcg", "mrr", "map", "a/b test", "ab test", "learning to rank",
                  "ltr", "xgboost rank", "ranknet", "lambdamart", "evaluation", "benchmark"]
    feats["eval_depth"] = min(1.0, keyword_hit_rate(full_text, eval_terms) * 4)

    llm_terms = ["llm", "rag", "fine-tun", "lora", "qlora", "peft", "instruction tuning",
                 "gpt", "bert", "transformers", "huggingface", "langchain"]
    feats["llm_depth"] = min(1.0, keyword_hit_rate(full_text, llm_terms) * 3)

    python_skills = [s for s in skills if "python" in text_lower(s.get("name", ""))]
    if python_skills:
        prof_map = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
        feats["python_strength"] = max(prof_map.get(s.get("proficiency", "beginner"), 0.1) for s in python_skills)
    else:
        feats["python_strength"] = 0.3 if "python" in full_text else 0.0

    # Experience
    feats["yoe_fit"] = yoe_score(yoe)
    ml_keywords = ["machine learning", "ml engineer", "ai engineer", "data scientist",
                   "nlp", "deep learning", "applied ml", "research engineer"]
    total_months = sum(j.get("duration_months", 0) for j in history)
    ml_months = sum(j.get("duration_months", 0) for j in history
                    if any(kw in text_lower(j.get("title", "") + " " + j.get("description", "")) for kw in ml_keywords))
    feats["ml_career_fraction"] = ml_months / max(total_months, 1)

    product_industries = ["technology", "software", "internet", "saas", "fintech",
                          "edtech", "healthtech", "e-commerce", "marketplace"]
    product_months = sum(j.get("duration_months", 0) for j in history
                         if any(pi in text_lower(j.get("industry", "")) for pi in product_industries))
    feats["product_company_fraction"] = product_months / max(total_months, 1)

    if len(history) >= 2:
        avg_tenure = total_months / len(history)
        feats["tenure_stability"] = min(1.0, avg_tenure / 36) if avg_tenure < 36 else max(0.5, 1.0 - (avg_tenure - 48) / 100)
    else:
        feats["tenure_stability"] = 0.5

    current_jobs = [j for j in history if j.get("is_current")]
    if current_jobs:
        curr = current_jobs[0]
        curr_ml = any(kw in text_lower(curr.get("title", "") + " " + curr.get("description", "")) for kw in ml_keywords)
        feats["current_role_ml"] = 1.0 if curr_ml else 0.2
    else:
        feats["current_role_ml"] = 0.3

    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}
    edu_score = max((tier_map.get(e.get("tier", "unknown"), 0.5) for e in edu), default=0.4)
    feats["edu_tier"] = edu_score

    # Behavioral
    last_active_days = days_since(signals.get("last_active_date", "2000-01-01"))
    if last_active_days < 7:     feats["recency"] = 1.0
    elif last_active_days < 30:  feats["recency"] = 0.85
    elif last_active_days < 90:  feats["recency"] = 0.6
    elif last_active_days < 180: feats["recency"] = 0.35
    else:                        feats["recency"] = 0.1

    feats["open_to_work"]         = 1.0 if signals.get("open_to_work_flag") else 0.4
    feats["recruiter_response"]   = float(signals.get("recruiter_response_rate", 0.0))
    notice = int(signals.get("notice_period_days", 90))
    if notice <= 15:   feats["notice_score"] = 1.0
    elif notice <= 30: feats["notice_score"] = 0.9
    elif notice <= 60: feats["notice_score"] = 0.6
    elif notice <= 90: feats["notice_score"] = 0.4
    else:              feats["notice_score"] = 0.2

    gh = float(signals.get("github_activity_score", -1))
    feats["github_score"]          = gh / 100 if gh >= 0 else 0.3
    feats["completeness"]          = float(signals.get("profile_completeness_score", 50)) / 100
    feats["recruiter_interest"]    = min(1.0, int(signals.get("saved_by_recruiters_30d", 0)) / 10)
    feats["interview_reliability"] = float(signals.get("interview_completion_rate", 0.5))

    pref = signals.get("preferred_work_mode", "flexible")
    feats["work_mode_fit"] = {"hybrid": 1.0, "flexible": 0.9, "onsite": 0.7, "remote": 0.5}.get(pref, 0.6)
    feats["location_fit"]  = location_score(c)

    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_mid = (float(sal.get("min", 0)) + float(sal.get("max", 999))) / 2
    if 20 <= sal_mid <= 70:   feats["salary_fit"] = 1.0
    elif 15 <= sal_mid < 20:  feats["salary_fit"] = 0.7
    elif 70 < sal_mid <= 100: feats["salary_fit"] = 0.6
    else:                     feats["salary_fit"] = 0.3

    feats["verified"]    = 1.0 if (signals.get("verified_email") and signals.get("verified_phone")) else 0.5
    feats["is_honeypot"] = 1.0 if cid in honeypot_ids else 0.0
    return feats

# ── Reasoning Generation ─────────────────────────────────────────────────────
def build_reasoning(c: dict, feats: dict, rank: int, score: float) -> str:
    p       = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    skills  = c.get("skills", [])

    title   = p.get("current_title", "Engineer")
    yoe     = p.get("years_of_experience", 0)
    company = p.get("current_company", "")
    loc     = p.get("location", "")

    top_skills = sorted(
        [s for s in skills if s.get("proficiency") in ("expert", "advanced")],
        key=lambda s: s.get("endorsements", 0), reverse=True
    )[:3]
    top_skill_names = ", ".join(s["name"] for s in top_skills) if top_skills else "general ML"

    notice         = signals.get("notice_period_days", 90)
    rr             = signals.get("recruiter_response_rate", 0)
    last_active_days = days_since(signals.get("last_active_date", "2000-01-01"))

    strengths, concerns = [], []
    if feats.get("retrieval_depth", 0) > 0.5:     strengths.append("strong retrieval/embedding background")
    if feats.get("eval_depth", 0) > 0.3:          strengths.append("ranking evaluation experience (NDCG/MRR)")
    if feats.get("product_company_fraction", 0) > 0.6: strengths.append("product-company career")
    if feats.get("has_production_ml", 0) > 0.5:   strengths.append("production ML deployment")
    if feats.get("github_score", 0) > 0.5:        strengths.append("active open-source contributions")
    if feats.get("llm_depth", 0) > 0.4:           strengths.append("LLM/fine-tuning experience")

    if notice > 60:            concerns.append(f"long notice period ({notice}d)")
    if rr < 0.3:               concerns.append(f"low recruiter response rate ({rr:.0%})")
    if last_active_days > 90:  concerns.append(f"inactive for {last_active_days} days")
    if feats.get("red_flag_title", 0) > 0:  concerns.append(f"current title ({title}) is outside AI/ML domain")
    if feats.get("consulting_only", 0) > 0: concerns.append("career exclusively at consulting firms")
    if feats.get("research_heavy", 0) > 0.5: concerns.append("research-heavy background without clear production deployments")
    if feats.get("is_honeypot", 0) > 0:     concerns.append("profile has data inconsistencies (honeypot signal)")

    s1 = f"{yoe:.0f}-year {title} at {company}" if company else f"{yoe:.0f}-year {title}"
    if loc:         s1 += f", {loc}"
    if top_skills:  s1 += f"; top skills: {top_skill_names}"

    parts2 = []
    if rank <= 10:
        if strengths: parts2.append("Strong fit: " + ", ".join(strengths[:3]))
        if concerns:  parts2.append("Concern: " + "; ".join(concerns[:1]))
    elif rank <= 30:
        if strengths: parts2.append("Fit signals: " + ", ".join(strengths[:2]))
        if concerns:  parts2.append("Gap: " + "; ".join(concerns[:2]))
    elif rank <= 60:
        if strengths: parts2.append("Partial match: " + ", ".join(strengths[:1]))
        if concerns:  parts2.append("Notable gaps: " + "; ".join(concerns[:2]))
    else:
        if concerns:  parts2.append("Significant gaps: " + "; ".join(concerns[:3]))
        elif strengths: parts2.append("Adjacent skills only — weak JD fit overall")
        else:         parts2.append("Profile lacks core JD requirements")

    if not parts2:
        parts2.append("Limited alignment with JD requirements")
    return f"{s1}. {'; '.join(parts2)}."

# ── Validation ───────────────────────────────────────────────────────────────
def validate_submission(df: pd.DataFrame, all_candidate_ids: set, honeypot_ids: set) -> bool:
    errors, warnings_list = [], []

    if len(df) != 100:
        errors.append(f"Row count = {len(df)}, expected 100")

    ranks = sorted(df["rank"].tolist())
    if ranks != list(range(1, 101)):
        errors.append(f"Ranks not exactly 1-100 (got {len(set(ranks))} unique, min={min(ranks)}, max={max(ranks)})")

    if df["candidate_id"].nunique() != 100:
        errors.append("Duplicate candidate_ids detected")

    unknown = set(df["candidate_id"]) - all_candidate_ids
    if unknown:
        errors.append(f"Unknown candidate_ids: {unknown}")

    req_cols = ["candidate_id", "rank", "score", "reasoning"]
    for col in req_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if list(df.columns)[:4] != req_cols:
        errors.append(f"Column order wrong: got {list(df.columns)}, expected {req_cols}")

    sorted_by_rank = df.sort_values("rank")
    score_diffs = sorted_by_rank["score"].diff().dropna()
    if (score_diffs > 1e-9).any():
        errors.append(f"Scores not non-increasing: {(score_diffs > 1e-9).sum()} violation(s)")

    if df["score"].nunique() == 1:
        errors.append("All scores identical — model not differentiating")

    top10_honeypots = df[df["rank"] <= 10]["candidate_id"].apply(lambda x: x in honeypot_ids).sum()
    if top10_honeypots > 0:
        warnings_list.append(f"{top10_honeypots} honeypots in top 10 — risk of Stage 3 disqualification")

    if "reasoning" in df.columns:
        empty = df["reasoning"].isna().sum() + (df["reasoning"] == "").sum()
        if empty > 10:
            warnings_list.append(f"{empty} empty reasoning entries")
        if df["reasoning"].nunique() < 50:
            warnings_list.append(f"Low reasoning diversity: only {df['reasoning'].nunique()} unique values")

    if not errors:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        print(f"{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  * {e}")
    if warnings_list:
        print(f"{len(warnings_list)} WARNING(S):")
        for w in warnings_list:
            print(f"  * {w}")
    return len(errors) == 0

# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--team-id",   default="NoBlackBox", help="Team ID (used for output filename)")
    parser.add_argument("--out",       default=None, help="Output CSV path (default: <team-id>.csv)")
    args = parser.parse_args()

    out_path = args.out or f"{args.team_id}.csv"
    t0 = time.time()

    # 1. Load candidates
    candidates = []
    errors = 0
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                if "candidate_id" in c:
                    candidates.append(c)
                else:
                    errors += 1
            except json.JSONDecodeError:
                errors += 1
    print(f"Loaded {len(candidates):,} candidates ({errors} parse errors) in {time.time()-t0:.1f}s")

    # 2. Honeypot detection
    honeypot_ids = set()
    for c in tqdm(candidates, desc="Honeypot detection"):
        is_hp, _ = detect_honeypot(c, honeypot_ids)
        if is_hp:
            honeypot_ids.add(c["candidate_id"])
    print(f"Honeypots detected: {len(honeypot_ids)}")

    # 3. Feature extraction
    feature_rows, candidate_texts = [], []
    for c in tqdm(candidates, desc="Extracting features"):
        feats = extract_features(c, honeypot_ids)
        feats["candidate_id"] = c["candidate_id"]
        feature_rows.append(feats)
        candidate_texts.append(build_candidate_text(c))
    feat_df = pd.DataFrame(feature_rows).set_index("candidate_id")

    # 4. Semantic scoring (TF-IDF + BM25)
    tfidf = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, strip_accents="unicode", analyzer="word",
        token_pattern=r"(?u)[a-z][a-z+#/\.]{1,}"
    )
    candidate_tfidf = tfidf.fit_transform(candidate_texts)
    jd_vec = tfidf.transform([JD_TEXT])
    tfidf_scores = cosine_similarity(jd_vec, candidate_tfidf).flatten()

    tokenized = [text.lower().split() for text in candidate_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores_raw = bm25.get_scores(JD_TEXT.lower().split())
    bm25_max = bm25_scores_raw.max()
    bm25_norm = bm25_scores_raw / bm25_max if bm25_max > 0 else bm25_scores_raw
    semantic_scores = 0.6 * tfidf_scores + 0.4 * bm25_norm

    cand_ids = [c["candidate_id"] for c in candidates]
    semantic_df = pd.DataFrame({
        "candidate_id": cand_ids,
        "tfidf_score":  tfidf_scores,
        "bm25_score":   bm25_norm,
        "semantic_score": semantic_scores
    }).set_index("candidate_id")

    # 5. Composite score
    all_df = feat_df.join(semantic_df, how="left")
    raw_score = sum(all_df[col].fillna(0) * w for col, w in WEIGHTS.items() if col in all_df.columns)
    penalty_mult = pd.Series(1.0, index=all_df.index)
    for pen_col, pen_factor in PENALTIES.items():
        if pen_col in all_df.columns:
            penalty_mult *= (1 - all_df[pen_col].fillna(0) * pen_factor)
    composite = raw_score * penalty_mult
    c_min, c_max = composite.min(), composite.max()
    composite_norm = (composite - c_min) / (c_max - c_min) if c_max > c_min else composite.copy()
    all_df["composite_score"] = composite_norm

    ranked = all_df.sort_values(["composite_score", "candidate_id"], ascending=[False, True]).reset_index()
    top100 = ranked.head(100).copy()
    top100["rank"] = range(1, 101)

    # 6. Reasoning
    cand_map = {c["candidate_id"]: c for c in candidates}
    feat_map = {row["candidate_id"]: row.to_dict() for _, row in feat_df.reset_index().iterrows()}
    reasonings = []
    for _, row in tqdm(top100.iterrows(), total=100, desc="Generating reasoning"):
        cid   = row["candidate_id"]
        r = build_reasoning(cand_map.get(cid, {}), feat_map.get(cid, {}), int(row["rank"]), float(row["composite_score"]))
        reasonings.append(r)
    top100["reasoning"] = reasonings

    # 7. Build submission DataFrame — exact column order per spec
    submission = top100[["candidate_id", "rank", "composite_score", "reasoning"]].copy()
    submission = submission.rename(columns={"composite_score": "score"})
    submission = submission.sort_values("rank").reset_index(drop=True)

    # Enforce non-increasing scores (fix floating-point ties)
    scores = submission["score"].values.copy()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            scores[i] = scores[i - 1]
    submission["score"] = scores

    # 8. Validate
    all_ids = {c["candidate_id"] for c in candidates}
    is_valid = validate_submission(submission, all_ids, honeypot_ids)

    # 9. Save
    if is_valid:
        submission.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Submission saved to: {out_path}")
        print(f"Rows: {len(submission)} | Score range: {submission['score'].min():.4f} - {submission['score'].max():.4f}")
    else:
        print("Submission NOT saved — fix errors above first.")

    total_time = time.time() - t0
    print(f"Total pipeline time: {total_time:.1f}s ({total_time/60:.2f} min)")
    if total_time >= 300:
        print("WARNING: Exceeded 5-minute compute budget!")

if __name__ == "__main__":
    main()
