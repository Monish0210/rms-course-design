import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from loguru import logger
from config.settings import EMBEDDING_MODEL


def load_reference_clos(filepath: str) -> list[str]:
    """
    Load expert reference CLOs from a text file.
    One CLO per line.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Reference file not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        clos = [line.strip() for line in f
                if line.strip() and not line.startswith("#")]

    logger.info(f"Loaded {len(clos)} reference CLOs from {filepath}")
    return clos


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def evaluate_clos(generated_clos: list[str],
                  reference_clos: list[str],
                  model: SentenceTransformer = None) -> dict:
    """
    Evaluate generated CLOs against expert reference CLOs
    using cosine similarity.

    For each generated CLO, find the best matching reference CLO
    and record the similarity score.

    Target: average similarity > 0.75

    Args:
        generated_clos: CLOs from the pipeline
        reference_clos: Expert-written reference CLOs
        model: Embedding model (loaded if not provided)

    Returns:
        Dict with scores and evaluation summary
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(
        f"Evaluating {len(generated_clos)} generated CLOs "
        f"against {len(reference_clos)} reference CLOs"
    )

    # Embed all CLOs
    gen_embeddings = model.encode(generated_clos)
    ref_embeddings = model.encode(reference_clos)

    results = []
    for i, (gen_clo, gen_emb) in enumerate(
            zip(generated_clos, gen_embeddings)):

        # Find best matching reference CLO
        best_score = 0.0
        best_ref = ""

        for ref_clo, ref_emb in zip(reference_clos, ref_embeddings):
            score = cosine_similarity(
                gen_emb.tolist(), ref_emb.tolist()
            )
            if score > best_score:
                best_score = score
                best_ref = ref_clo

        results.append({
            "generated": gen_clo,
            "best_match": best_ref,
            "score": round(best_score, 4)
        })

        logger.info(
            f"CLO{i+1} score: {best_score:.4f} | "
            f"Match: '{best_ref[:40]}...'"
        )

    avg_score = sum(r["score"] for r in results) / len(results)
    target_met = avg_score >= 0.75

    logger.info(
        f"Average similarity: {avg_score:.4f} | "
        f"Target (>0.75): {'MET' if target_met else 'NOT MET'}"
    )

    return {
        "results": results,
        "average_score": round(avg_score, 4),
        "target_score": 0.75,
        "target_met": target_met,
        "num_generated": len(generated_clos),
        "num_reference": len(reference_clos)
    }


def print_evaluation_report(eval_result: dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "="*60)
    print("CLO EVALUATION REPORT")
    print("="*60)
    print(f"Generated CLOs : {eval_result['num_generated']}")
    print(f"Reference CLOs : {eval_result['num_reference']}")
    print(f"Average Score  : {eval_result['average_score']}")
    print(f"Target (>0.75) : "
          f"{'MET' if eval_result['target_met'] else 'NOT MET'}")

    print("\nPer-CLO Results:")
    print("-"*60)
    for i, r in enumerate(eval_result["results"], 1):
        status = "PASS" if r["score"] >= 0.75 else "BELOW"
        print(f"CLO{i} [{status}] Score: {r['score']}")
        print(f"  Generated : {r['generated'][:55]}")
        print(f"  Best match: {r['best_match'][:55]}")
    print("="*60)


if __name__ == "__main__":
    from workflow.graph import run_workflow

    # Step 1 — Run pipeline to get generated CLOs
    print("Running pipeline to generate CLOs...")
    state = run_workflow(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        num_modules=5
    )
    generated_clos = state.course.clos
    print(f"Generated {len(generated_clos)} CLOs")

    # Step 2 — Load or create reference CLOs
    ref_path = "data/references/sdn_reference_clos.txt"

    if not os.path.exists(ref_path):
        # Create reference CLOs if file does not exist
        print(f"\nReference file not found. Creating: {ref_path}")
        os.makedirs("data/references", exist_ok=True)
        reference_clos = [
            "Explain the fundamental concepts and architecture of Software Defined Networking",
            "Design and implement SDN network topologies using OpenFlow protocol",
            "Implement SDN controllers and configure network devices programmatically",
            "Analyze and evaluate the performance characteristics of SDN-based networks",
            "Evaluate security vulnerabilities and design secure SDN deployments",
            "Compare traditional networking with SDN approaches and justify architectural choices",
            "Apply SDN principles to solve complex network management problems",
            "Describe the role of SDN controllers in managing network infrastructure"
        ]
        with open(ref_path, "w") as f:
            f.write("# SDN Expert Reference CLOs\n")
            for clo in reference_clos:
                f.write(clo + "\n")
        print(f"Created reference file with {len(reference_clos)} CLOs")
    else:
        reference_clos = load_reference_clos(ref_path)

    # Step 3 — Evaluate
    model = SentenceTransformer(EMBEDDING_MODEL)
    eval_result = evaluate_clos(generated_clos, reference_clos, model)

    # Step 4 — Print report
    print_evaluation_report(eval_result)