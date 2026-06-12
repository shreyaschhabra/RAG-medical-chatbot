"""Offline evaluation — 50 ground-truth QA pairs.
Run: python eval.py
Outputs: eval_results.json  (metrics) + prints a summary table.
"""
import json
import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import CrossEncoder

from config import REPO_ID, EMBED_MODEL, RERANK_MODEL, DB_PATH, FETCH_K, TOP_K, TEMPERATURE, MAX_TOKENS

load_dotenv()

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
     "ground_truth": "Peripheral artery disease is narrowing of arteries reducing blood flow to limbs, causing leg pain while walking (claudication), numbness, and increased risk of amputation."},
    {"question": "What causes epilepsy?",
     "ground_truth": "Epilepsy is caused by abnormal electrical activity in the brain due to genetic factors, brain injury, stroke, infection, or unknown causes, resulting in recurrent seizures."},
    {"question": "What are the symptoms of Lyme disease?",
     "ground_truth": "Lyme disease symptoms include a characteristic bulls-eye rash, fever, fatigue, joint pain, and if untreated, neurological and cardiac complications."},
    {"question": "What is aortic aneurysm?",
     "ground_truth": "An aortic aneurysm is an abnormal bulging of the aorta wall that can rupture, causing life-threatening internal bleeding; risk factors include hypertension and smoking."},
    {"question": "How is sepsis treated?",
     "ground_truth": "Sepsis is treated with intravenous antibiotics, fluid resuscitation, vasopressors for low blood pressure, oxygen support, and source control of the infection."},
    {"question": "What is endometriosis?",
     "ground_truth": "Endometriosis is a condition where tissue similar to the uterine lining grows outside the uterus, causing chronic pelvic pain, painful periods, and infertility."},
    {"question": "What is pancreatitis?",
     "ground_truth": "Pancreatitis is inflammation of the pancreas, usually caused by gallstones or excessive alcohol use, presenting with severe abdominal pain, nausea, and elevated lipase levels."},
    {"question": "What are the risk factors for colorectal cancer?",
     "ground_truth": "Risk factors for colorectal cancer include age over 50, family history, inflammatory bowel disease, high-fat low-fiber diet, obesity, smoking, and alcohol consumption."},
    {"question": "What is carpal tunnel syndrome?",
     "ground_truth": "Carpal tunnel syndrome is compression of the median nerve at the wrist causing hand numbness, tingling, and weakness, worsened by repetitive motions."},
]


def build_retriever_reranker():
    emb       = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    db        = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)
    all_docs  = list(db.docstore._dict.values())
    bm25      = BM25Retriever.from_documents(all_docs); bm25.k = FETCH_K
    faiss_ret = db.as_retriever(search_kwargs={"k": FETCH_K})
    retriever = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.4, 0.6])
    reranker  = CrossEncoder(RERANK_MODEL)
    return retriever, reranker


def build_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN", ""), task="text-generation",
    )
    return ChatHuggingFace(llm=endpoint)


def get_answer(question, retriever, reranker, llm):
    docs   = retriever.invoke(question)
    pairs  = [[question, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    top    = [d for _, d in sorted(zip(scores, docs), reverse=True)][:TOP_K]
    ctx    = "\n\n---\n\n".join(d.page_content for d in top)
    msgs   = [
        SystemMessage(content=f"Answer ONLY from the context below.\n\nContext:\n{ctx}"),
        HumanMessage(content=question),
    ]
    return llm.invoke(msgs).content, top


def context_recall(context: str, ground_truth: str) -> float:
    gt_words  = set(ground_truth.lower().split())
    ctx_words = set(context.lower().split())
    return len(gt_words & ctx_words) / len(gt_words) if gt_words else 0.0


def main():
    print("Loading models…")
    retriever, reranker = build_retriever_reranker()
    llm = build_llm()
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results, latencies = [], []
    r1s, r2s, rls, ctx_recalls = [], [], [], []

    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"[{i:02d}/50] {item['question'][:60]}…")
        t0 = time.time()
        answer, docs = get_answer(item["question"], retriever, reranker, llm)
        latency = time.time() - t0

        scores   = scorer.score(item["ground_truth"], answer)
        ctx_text = " ".join(d.page_content for d in docs)
        recall   = context_recall(ctx_text, item["ground_truth"])

        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rls.append(scores["rougeL"].fmeasure)
        ctx_recalls.append(recall)
        latencies.append(latency)

        results.append({
            "question":       item["question"],
            "ground_truth":   item["ground_truth"],
            "answer":         answer,
            "rouge1_f":       round(scores["rouge1"].fmeasure, 4),
            "rouge2_f":       round(scores["rouge2"].fmeasure, 4),
            "rougeL_f":       round(scores["rougeL"].fmeasure, 4),
            "context_recall": round(recall, 4),
            "latency_s":      round(latency, 2),
        })
        time.sleep(0.5)

    summary = {
        "n_questions":          50,
        "avg_rouge1_f":         round(sum(r1s) / 50, 4),
        "avg_rouge2_f":         round(sum(r2s) / 50, 4),
        "avg_rougeL_f":         round(sum(rls) / 50, 4),
        "avg_context_recall":   round(sum(ctx_recalls) / 50, 4),
        "avg_latency_s":        round(sum(latencies) / 50, 2),
        "p50_latency_s":        round(sorted(latencies)[24], 2),
        "p95_latency_s":        round(sorted(latencies)[47], 2),
    }

    output = {"summary": summary, "results": results}
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()
