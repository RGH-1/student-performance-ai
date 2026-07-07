import matplotlib.pyplot as plt

def plot_score(r2_scores, title):
    outputs = [
        "Focus Index",
        "Burnout Level",
        "Productivity Score",
        "Exam Score"
    ]

    bars = plt.bar(outputs, r2_scores)
    plt.xlabel("Output")
    plt.ylabel("Score")
    plt.title("R2 Outputs for " + title)

    plt.ylim(0,1)

    plt.show()