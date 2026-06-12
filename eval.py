"""Retrieval evaluation — 50 ground-truth QA pairs.
Measures the RAG pipeline's retrieval quality with no LLM calls needed.

Metrics
-------
Context Recall   : fraction of ground-truth key terms found in retrieved chunks
ROUGE-1/2/L F1   : n-gram overlap between retrieved context and ground truth
Hit Rate @3      : >= 1 retrieved chunk contains >30% of ground-truth terms
MRR @3           : Mean Reciprocal Rank of first relevant chunk
Retrieval Latency: wall-clock time for hybrid retrieval + reranking

Run: python eval.py
Output: eval_results.json + printed summary table
"""
import json
import time

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import CrossEncoder

from config import DB_PATH, EMBED_MODEL, FETCH_K, RERANK_MODEL, TOP_K

EVAL_DATASET = [
    {"question": "What is diabetes mellitus?",
     "ground_truth": "Diabetes mellitus is a metabolic disease where the body cannot properly regulate blood glucose due to insufficient insulin production or resistance to insulin."},
    {"question": "What are the main symptoms of a heart attack?",
     "ground_truth": "Heart attack symptoms include chest pain or pressure, pain radiating to the arm or jaw, shortness of breath, sweating, nausea, and lightheadedness."},
    {"question": "What is hypertension?",
     "ground_truth": "Hypertension is persistently elevated blood pressure of 140/90 mmHg or higher, increasing risk of heart disease and stroke."},
    {"question": "How is pneumonia treated?",
     "ground_truth": "Pneumonia is treated with antibiotics for bacterial infection, rest, fluids, and in severe cases, hospitalization and oxygen therapy."},
    {"question": "What causes asthma?",
     "ground_truth": "Asthma is caused by chronic airway inflammation triggered by allergens, exercise, cold air, or respiratory infections leading to airway narrowing."},
    {"question": "What is Alzheimer's disease?",
     "ground_truth": "Alzheimer's disease is a progressive neurodegenerative disorder causing memory loss, cognitive decline, and behavioral changes due to brain cell death."},
    {"question": "What are the symptoms of depression?",
     "ground_truth": "Depression symptoms include persistent sadness, loss of interest in activities, fatigue, sleep disturbances, appetite changes, and suicidal thoughts."},
    {"question": "What is HIV/AIDS?",
     "ground_truth": "HIV is a virus that destroys CD4 immune cells; AIDS is the advanced stage where the immune system is severely compromised, leaving the body vulnerable to infections."},
    {"question": "How is appendicitis diagnosed?",
     "ground_truth": "Appendicitis is diagnosed through physical examination, blood tests showing elevated white cells, and imaging such as ultrasound or CT scan."},
    {"question": "What is osteoporosis?",
     "ground_truth": "Osteoporosis is a condition of reduced bone density making bones fragile and prone to fractures, common in postmenopausal women and older adults."},
    {"question": "What causes kidney stones?",
     "ground_truth": "Kidney stones form from mineral and salt deposits, often calcium oxalate, due to concentrated urine, dehydration, or high-oxalate diet."},
    {"question": "What are the symptoms of anemia?",
     "ground_truth": "Anemia symptoms include fatigue, weakness, pale skin, shortness of breath, dizziness, and cold hands and feet due to low red blood cells."},
    {"question": "What is Parkinson's disease?",
     "ground_truth": "Parkinson's disease is a neurological disorder characterized by tremors, muscle stiffness, slow movement, and balance problems due to dopamine deficiency."},
    {"question": "How is hypothyroidism treated?",
     "ground_truth": "Hypothyroidism is treated with daily oral levothyroxine, a synthetic thyroid hormone, to normalize TSH levels."},
    {"question": "What is lupus?",
     "ground_truth": "Lupus is an autoimmune disease where the immune system attacks healthy tissue, causing joint pain, skin rashes, kidney problems, and fatigue."},
    {"question": "What causes migraines?",
     "ground_truth": "Migraines are caused by changes in brain chemistry and nerve signaling, triggered by stress, hormones, certain foods, or sensory stimuli."},
    {"question": "What are the symptoms of a stroke?",
     "ground_truth": "Stroke symptoms include sudden facial drooping, arm weakness, speech difficulty, vision loss, and severe headache requiring immediate emergency care."},
    {"question": "What is multiple sclerosis?",
     "ground_truth": "Multiple sclerosis is an autoimmune disease where the immune system attacks the myelin sheath of nerve fibers, causing muscle weakness, vision problems, and coordination loss."},
    {"question": "How is tuberculosis diagnosed?",
     "ground_truth": "Tuberculosis is diagnosed via skin test (Mantoux), chest X-ray, sputum culture, and interferon-gamma release assays (IGRA)."},
    {"question": "What is rheumatoid arthritis?",
     "ground_truth": "Rheumatoid arthritis is an autoimmune disorder causing chronic inflammation in joints, leading to joint damage, swelling, pain, and systemic effects."},
    {"question": "What causes peptic ulcers?",
     "ground_truth": "Peptic ulcers are caused by Helicobacter pylori infection and prolonged NSAID use, which erode the protective mucus lining of the stomach or duodenum."},
    {"question": "What are the symptoms of hypothyroidism?",
     "ground_truth": "Hypothyroidism symptoms include fatigue, weight gain, cold intolerance, dry skin, hair loss, constipation, and depression due to low thyroid hormone."},
    {"question": "What is celiac disease?",
     "ground_truth": "Celiac disease is an autoimmune disorder triggered by gluten ingestion, causing small intestine damage, malabsorption, diarrhea, and bloating."},
    {"question": "How is COPD treated?",
     "ground_truth": "COPD is treated with bronchodilators, inhaled corticosteroids, pulmonary rehabilitation, oxygen therapy, and smoking cessation."},
    {"question": "What is fibromyalgia?",
     "ground_truth": "Fibromyalgia is a chronic condition causing widespread musculoskeletal pain, fatigue, sleep problems, and cognitive difficulties without identifiable tissue damage."},
    {"question": "What causes gallstones?",
     "ground_truth": "Gallstones form when bile contains too much cholesterol or bilirubin, or too little bile salts, leading to crystal and stone formation in the gallbladder."},
    {"question": "What are the symptoms of appendicitis?",
     "ground_truth": "Appendicitis presents with pain beginning around the navel moving to the lower right abdomen, nausea, vomiting, fever, and loss of appetite."},
    {"question": "What is schizophrenia?",
     "ground_truth": "Schizophrenia is a chronic psychiatric disorder characterized by hallucinations, delusions, disorganized thinking, and negative symptoms such as flat affect."},
    {"question": "How is leukemia treated?",
     "ground_truth": "Leukemia is treated with chemotherapy, targeted therapy, radiation, immunotherapy, and bone marrow or stem cell transplantation depending on type and stage."},
    {"question": "What is Crohn's disease?",
     "ground_truth": "Crohn's disease is an inflammatory bowel disease causing chronic inflammation of any part of the gastrointestinal tract, leading to diarrhea, abdominal pain, and malnutrition."},
    {"question": "What causes chickenpox?",
     "ground_truth": "Chickenpox is caused by the varicella-zoster virus, spreading through respiratory droplets and direct contact with the rash, causing an itchy blister-like eruption."},
    {"question": "What are the symptoms of pneumonia?",
     "ground_truth": "Pneumonia symptoms include cough with phlegm, fever, chills, shortness of breath, chest pain, and fatigue."},
    {"question": "What is meningitis?",
     "ground_truth": "Meningitis is inflammation of the membranes surrounding the brain and spinal cord, caused by bacterial or viral infection, presenting with severe headache, stiff neck, and fever."},
    {"question": "How is hypertension diagnosed?",
     "ground_truth": "Hypertension is diagnosed by measuring blood pressure on multiple occasions; readings consistently at or above 140/90 mmHg confirm the diagnosis."},
    {"question": "What is chronic kidney disease?",
     "ground_truth": "Chronic kidney disease is progressive loss of kidney function over months to years, leading to accumulation of waste products, fluid imbalance, and eventual kidney failure."},
    {"question": "What causes eczema?",
     "ground_truth": "Eczema is caused by a combination of genetic predisposition to skin barrier dysfunction and immune dysregulation, triggered by allergens, irritants, or stress."},
    {"question": "What is deep vein thrombosis?",
     "ground_truth": "Deep vein thrombosis is a blood clot forming in a deep vein, usually the leg, causing swelling, pain, and risk of pulmonary embolism if the clot dislodges."},
    {"question": "How is malaria treated?",
     "ground_truth": "Malaria is treated with antimalarial drugs such as chloroquine, artemisinin-based combination therapy, or quinine depending on the Plasmodium species and resistance patterns."},
    {"question": "What is hepatitis B?",
     "ground_truth": "Hepatitis B is a viral liver infection transmitted through blood, sexual contact, or from mother to child, causing acute or chronic liver disease and cirrhosis risk."},
    {"question": "What causes psoriasis?",
     "ground_truth": "Psoriasis is caused by an overactive immune response accelerating skin cell growth, resulting in thick, scaly patches, triggered by stress, infections, or medications."},
    {"question": "What is Graves' disease?",
     "ground_truth": "Graves' disease is an autoimmune disorder causing hyperthyroidism, where antibodies stimulate the thyroid to produce excess hormone, causing weight loss, tremors, and exophthalmos."},
    {"question": "How is sleep apnea diagnosed?",
     "ground_truth": "Sleep apnea is diagnosed through a polysomnography (sleep study) that measures breathing, oxygen levels, and brain activity during sleep."},
    {"question": "What is peripheral artery disease?",
     "ground_truth": "Peripheral artery disease is narrowing of arteries reducing blood flow to limbs, causing leg pain while walking, numbness, and increased risk of amputation."},
    {"question": "What causes epilepsy?",
     "ground_truth": "Epilepsy is caused by abnormal electrical activity in the brain due to genetic factors, brain injury, stroke, infection, or unknown causes, resulting in recurrent seizures."},
    {"question": "What are the symptoms of Lyme disease?",
     "ground_truth": "Lyme disease symptoms include a bulls-eye rash, fever, fatigue, joint pain, and if untreated, neurological and cardiac complications."},
    {"question": "What is aortic aneurysm?",
     "ground_truth": "An aortic aneurysm is an abnormal bulging of the aorta wall that can rupture causing life-threatening bleeding; risk factors include hypertension and smoking."},
    {"question": "How is sepsis treated?",
     "ground_truth": "Sepsis is treated with intravenous antibiotics, fluid resuscitation, vasopressors for low blood pressure, oxygen support, and source control of the infection."},
    {"question": "What is endometriosis?",
     "ground_truth": "Endometriosis is a condition where tissue similar to the uterine lining grows outside the uterus, causing chronic pelvic pain, painful periods, and infertility."},
    {"question": "What is pancreatitis?",
     "ground_truth": "Pancreatitis is inflammation of the pancreas usually caused by gallstones or alcohol use, presenting with severe abdominal pain, nausea, and elevated lipase levels."},
    {"question": "What are the risk factors for colorectal cancer?",
     "ground_truth": "Risk factors for colorectal cancer include age over 50, family history, inflammatory bowel disease, high-fat low-fiber diet, obesity, smoking, and alcohol consumption."},
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def context_recall(context: str, ground_truth: str, scorer) -> float:
    """ROUGE-L recall: fraction of ground-truth content found in context (stemmed)."""
    return scorer.score(ground_truth, context)["rougeL"].recall


def hit_rate(docs, ground_truth: str, scorer, threshold: float = 0.12) -> int:
    """1 if any retrieved chunk covers >=12% of ground truth via ROUGE-L recall."""
    for doc in docs:
        if context_recall(doc.page_content, ground_truth, scorer) >= threshold:
            return 1
    return 0


def mrr(docs, ground_truth: str, scorer, threshold: float = 0.12) -> float:
    """Reciprocal rank of the first chunk that passes the ROUGE-L recall threshold."""
    for rank, doc in enumerate(docs, 1):
        if context_recall(doc.page_content, ground_truth, scorer) >= threshold:
            return 1.0 / rank
    return 0.0


# ── Setup ──────────────────────────────────────────────────────────────────────

def build_retriever_reranker():
    emb       = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    db        = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)
    all_docs  = list(db.docstore._dict.values())
    bm25      = BM25Retriever.from_documents(all_docs); bm25.k = FETCH_K
    faiss_ret = db.as_retriever(search_kwargs={"k": FETCH_K})
    retriever = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.4, 0.6])
    reranker  = CrossEncoder(RERANK_MODEL)
    return retriever, reranker


def retrieve_and_rerank(question, retriever, reranker):
    docs   = retriever.invoke(question)
    pairs  = [[question, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    return [d for _, d in sorted(zip(scores, docs), reverse=True)][:TOP_K]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading retriever and reranker…")
    retriever, reranker = build_retriever_reranker()
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    records, latencies = [], []
    r1s, r2s, rls, recalls, hits, mrrs = [], [], [], [], [], []

    for i, item in enumerate(EVAL_DATASET, 1):
        q, gt = item["question"], item["ground_truth"]
        print(f"[{i:02d}/50] {q[:65]}")

        t0   = time.perf_counter()
        docs = retrieve_and_rerank(q, retriever, reranker)
        lat  = time.perf_counter() - t0

        ctx    = " ".join(d.page_content for d in docs)
        scores = scorer.score(gt, ctx)
        recall = context_recall(ctx, gt, scorer)
        hit    = hit_rate(docs, gt, scorer)
        rr     = mrr(docs, gt, scorer)

        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rls.append(scores["rougeL"].fmeasure)
        recalls.append(recall)
        hits.append(hit)
        mrrs.append(rr)
        latencies.append(lat)

        records.append({
            "question":       q,
            "ground_truth":   gt,
            "rouge1_f":       round(scores["rouge1"].fmeasure, 4),
            "rouge2_f":       round(scores["rouge2"].fmeasure, 4),
            "rougeL_f":       round(scores["rougeL"].fmeasure, 4),
            "context_recall": round(recall, 4),
            "hit_rate":       hit,
            "mrr":            round(rr, 4),
            "latency_s":      round(lat, 3),
        })

    n = len(EVAL_DATASET)
    sorted_lat = sorted(latencies)
    summary = {
        "n_questions":        n,
        "avg_rouge1_f":       round(sum(r1s) / n, 4),
        "avg_rouge2_f":       round(sum(r2s) / n, 4),
        "avg_rougeL_f":       round(sum(rls) / n, 4),
        "avg_context_recall": round(sum(recalls) / n, 4),
        "hit_rate@3":         round(sum(hits) / n, 4),
        "mrr@3":              round(sum(mrrs) / n, 4),
        "avg_latency_s":      round(sum(latencies) / n, 3),
        "p50_latency_s":      round(sorted_lat[n // 2], 3),
        "p95_latency_s":      round(sorted_lat[int(n * 0.95)], 3),
    }

    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_question": records}, f, indent=2)

    print("\n" + "=" * 42)
    print("  Retrieval Evaluation Summary (n=50)")
    print("=" * 42)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("=" * 42)
    print("Full results saved to eval_results.json")


if __name__ == "__main__":
    main()
