def generate_credit_score(prob):

    score = int(
        850 - (prob * 550)
    )

    return score
score = generate_credit_score(0.12)

print(score)