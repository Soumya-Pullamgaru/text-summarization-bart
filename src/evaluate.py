from rouge import Rouge
import matplotlib.pyplot as plt

def evaluate_rouge(model_output, reference):
    """
    Computes ROUGE scores between model output and reference summary.
    """
    rouge = Rouge()
    scores = rouge.get_scores(model_output, reference, avg=True)
    return scores

def plot_rouge(scores):
    """
    Plots ROUGE f-scores as a graph.
    """
    rouge_types = ["rouge-1", "rouge-2", "rouge-l"]
    f_scores = [
        scores['rouge-1']['f'] * 100,
        scores['rouge-2']['f'] * 100,
        scores['rouge-l']['f'] * 100
    ]

    plt.plot(rouge_types, f_scores, marker='o')
    plt.title('Rouge Graph')
    plt.xlabel('Rouge Type')
    plt.ylabel('F-Scores')
    plt.ylim(0, 100)
    plt.savefig('../results/rouge_graph.png')  # saves the graph
    plt.show()


# Test it
if __name__ == "__main__":
    model_out = ['About 10 men armed with pistols raided a casino in Switzerland.']
    reference = ['About 10 men armed with pistols and machine guns raided a casino in Basel.']

    scores = evaluate_rouge(model_out, reference)
    print("ROUGE Scores:", scores)
    plot_rouge(scores)