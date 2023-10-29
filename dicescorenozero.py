# %%
# Read Dice Scores from the text file
dice_scores = []

with open("C:\calhacks_hackathon\AI_Model\dicescores_9500.txt", "r") as file:
    for line in file:
        if line.startswith("Dice Score:"):
            parts = line.split()
            dice_score = float(parts[-1])
            # Replace 0 scores with 1
            if dice_score == 0:
                dice_score = 1
            dice_scores.append(dice_score)

# Calculate the average
total_score = sum(dice_scores)
average_dice_score = total_score / len(dice_scores)

print(f"Average Dice Score: {average_dice_score:.4f}")


# %%